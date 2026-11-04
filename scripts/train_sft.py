"""SFT warm-start on oracle trajectories, then merge for GRPO.

Why this exists: the Nov 2 GRPO smoke on the base model logged
`frac_reward_zero_std=1` — every completion group got the same reward, so GRPO
had no advantage to learn from. This script installs the output format and the
ledger-copy behaviour first; `train_grpo.py` then optimises success/efficiency.

Run (GPU, after `pip install -r requirements.txt` and `hf auth login`):

    python scripts/train_sft.py --check
    python scripts/train_sft.py --limit 20 --epochs 1 --out runs/sft_smoke
    python scripts/train_sft.py --tasks data/tasks_train.jsonl --epochs 3 \
        --out runs/sft --merge-out runs/sft_merged
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hinglish_agent.ledger import empty_ledger  # noqa: E402
from hinglish_agent.parser import format_call  # noqa: E402
from hinglish_agent.prompts import build_messages  # noqa: E402
from hinglish_agent.runner import LEDGER_TOOLS  # noqa: E402
from hinglish_agent.tools import execute_tool  # noqa: E402


def load_tasks(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_conversations(tasks: list[dict]) -> list[dict]:
    """Replay the oracle and record the exact messages the agent saw."""
    rows = []
    for task in tasks:
        ledger = empty_ledger()
        messages = build_messages(task["task"])
        targets: list[int] = []

        for action in task["actions"]:
            arguments = dict(action["parameters"])
            if action["tool_name"] in LEDGER_TOOLS:
                arguments["ledger"] = ledger
            result = execute_tool(action["tool_name"], arguments)
            if not result.get("ok"):
                raise RuntimeError(f"{task['id']}: {result}")

            messages.append(
                {
                    "role": "assistant",
                    "content": format_call(
                        {"tool_name": action["tool_name"], "parameters": arguments}
                    ),
                }
            )
            targets.append(len(messages) - 1)

            if isinstance(result.get("ledger"), dict):
                ledger = result["ledger"]
            messages.append(
                {
                    "role": "user",
                    "content": json.dumps(
                        {"tool_results": [{"tool_name": action["tool_name"], "result": result}]},
                        ensure_ascii=False,
                    ),
                }
            )

        messages.append(
            {
                "role": "assistant",
                "content": format_call(
                    {"tool_name": "finish", "parameters": {"answer": task["answer_text"]}}
                ),
            }
        )
        targets.append(len(messages) - 1)
        rows.append({"id": task["id"], "messages": messages, "targets": targets})
    return rows


def _template_ids(tokenizer, messages: list[dict], add_generation_prompt: bool) -> list[int]:
    """apply_chat_template returns a BatchEncoding by default in transformers 5."""
    output = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        return_dict=False,
    )
    if isinstance(output, dict):
        output = output["input_ids"]
    if hasattr(output, "tolist"):
        output = output.tolist()
    if output and isinstance(output[0], list):
        output = output[0]
    return list(output)


def encode_row(row: dict, tokenizer, max_len: int):
    """Tokenise once; mask everything except the assistant spans."""
    messages = row["messages"]
    full_ids = _template_ids(tokenizer, messages, add_generation_prompt=False)
    labels = [-100] * len(full_ids)

    for index in row["targets"]:
        prefix_ids = _template_ids(tokenizer, messages[:index], add_generation_prompt=True)
        upto_ids = _template_ids(tokenizer, messages[: index + 1], add_generation_prompt=False)
        if full_ids[: len(prefix_ids)] != prefix_ids:
            raise ValueError(
                f"{row['id']}: chat template is not prefix-stable at assistant turn {index}"
            )
        start, end = len(prefix_ids), len(upto_ids)
        if end <= start:
            raise ValueError(f"{row['id']}: empty assistant span at turn {index}")
        labels[start:end] = full_ids[start:end]

    if len(full_ids) > max_len:
        raise ValueError(
            f"{row['id']}: {len(full_ids)} tokens > --max-len {max_len}; shorten the task"
        )
    if all(label == -100 for label in labels):
        raise ValueError(f"{row['id']}: no assistant tokens to train on")
    return {"input_ids": full_ids, "labels": labels}


class TokenDataset:
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return self.rows[index]


def make_collator(pad_token_id: int):
    import torch

    def collate(batch: list[dict]) -> dict:
        width = max(len(item["input_ids"]) for item in batch)
        input_ids, labels, attention = [], [], []
        for item in batch:
            pad = width - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [pad_token_id] * pad)
            labels.append(item["labels"] + [-100] * pad)
            attention.append([1] * len(item["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
        }

    return collate


def masked_causal_lm_loss(
    logits,
    labels,
    vocab_size: int | None = None,
    num_items_in_batch=None,
    ignore_index: int = -100,
    shift_labels=None,
    **kwargs,
):
    """Cross-entropy on supervised tokens only.

    transformers' ForCausalLMLoss upcasts every logit to fp32. At 2670 tokens x
    256k vocab that is ~2.7 GB and OOMs a 14.5 GB T4. Only ~450 tokens per row
    carry labels, so select them before the upcast and skip the rest.
    """
    import torch
    import torch.nn.functional as F

    vocab_size = vocab_size or logits.size(-1)
    if shift_labels is None:
        labels = F.pad(labels, (0, 1), value=ignore_index)
        shift_labels = labels[..., 1:].contiguous()

    flat_logits = logits.view(-1, vocab_size)
    flat_labels = shift_labels.view(-1).to(flat_logits.device)
    keep = flat_labels != ignore_index
    if not keep.any():
        return flat_logits.sum() * 0.0

    loss = F.cross_entropy(
        flat_logits[keep].float(), flat_labels[keep], ignore_index=ignore_index, reduction="sum"
    )
    if num_items_in_batch is not None:
        denom = (
            num_items_in_batch.to(loss.device)
            if torch.is_tensor(num_items_in_batch)
            else num_items_in_batch
        )
        return loss / denom
    return loss / keep.sum()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="CohereLabs/tiny-aya-global")
    parser.add_argument("--tasks", default=str(ROOT / "data" / "tasks.jsonl"))
    parser.add_argument("--out", default=str(ROOT / "runs" / "sft"))
    parser.add_argument("--merge-out", default=str(ROOT / "runs" / "sft_merged"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-len", type=int, default=4096)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="seeds LoRA init and the data order; run 2 seeds to report training variance",
    )
    parser.add_argument("--no-merge", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    tasks = load_tasks(Path(args.tasks))
    if args.limit:
        tasks = tasks[: args.limit]
    conversations = build_conversations(tasks)

    assistant_turns = sum(len(row["targets"]) for row in conversations)
    longest = max(len(json.dumps(row["messages"], ensure_ascii=False)) for row in conversations)
    print(f"rows: {len(conversations)}  assistant turns: {assistant_turns}  longest chars: {longest}")
    print(f"first id: {conversations[0]['id']}  targets: {conversations[0]['targets']}")

    if args.check:
        return

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

    if not torch.cuda.is_available():
        raise SystemExit("SFT needs CUDA; run this on Kaggle/Colab, not the local Windows shell.")

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    encoded, skipped = [], []
    for row in conversations:
        try:
            encoded.append(encode_row(row, tokenizer, args.max_len))
        except ValueError as exc:
            print(f"skip: {exc}")
            skipped.append(row["id"])
    if not encoded:
        raise SystemExit(f"no trainable rows at --max-len {args.max_len}; skipped {skipped}")
    if skipped:
        print(f"warning: skipped {len(skipped)} overlong rows: {skipped}")
    lengths = [len(item["input_ids"]) for item in encoded]
    print(f"tokens: min={min(lengths)} mean={sum(lengths) // len(lengths)} max={max(lengths)}")

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16).to("cuda")
    model.loss_function = masked_causal_lm_loss
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    expected_steps = max(1, int(len(encoded) * args.epochs) // args.grad_accum)
    warmup_steps = max(1, int(expected_steps * 0.03))
    model.print_trainable_parameters()
    print("loss: assistant-token cross-entropy (skips masked logits)")
    print(f"plan: ~{expected_steps} optimizer steps, warmup {warmup_steps}")

    training_args = TrainingArguments(
        output_dir=str(args.out),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_steps=warmup_steps,
        seed=args.seed,
        data_seed=args.seed,
        logging_steps=2,
        save_steps=25,
        save_total_limit=2,
        fp16=True,
        optim="adamw_torch",
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=TokenDataset(encoded),
        data_collator=make_collator(tokenizer.pad_token_id),
    )
    result = trainer.train(resume_from_checkpoint=True if args.resume else None)
    print(f"done: {result}")

    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"adapter saved -> {args.out}")

    if not args.no_merge:
        model = model.to("cpu")
        torch.cuda.empty_cache()
        merged = model.merge_and_unload()
        merged.save_pretrained(args.merge_out, safe_serialization=True)
        tokenizer.save_pretrained(args.merge_out)
        print(f"merged model saved -> {args.merge_out}")

    print("\nNext: GRPO from the SFT model")
    print(
        f"python scripts/train_grpo.py --model {args.merge_out} "
        f"--tasks {args.tasks} --max-steps 60 --efficiency"
    )


if __name__ == "__main__":
    main()
