"""Run Tiny Aya through the agent loop and save before/after-ready artifacts.

Colab/Kaggle (GPU):
    python scripts/run_baseline.py --model CohereLabs/tiny-aya-global --limit 20

After training:
    python scripts/run_baseline.py --model CohereLabs/tiny-aya-global \
        --adapter runs/grpo_tiny_aya --out runs/tuned

Tiny Aya is gated: accept the licence on the model page and run `hf auth login` first.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hinglish_agent.metrics import markdown_table, summarise  # noqa: E402
from hinglish_agent.runner import run_episode, score_episode, to_trajectory  # noqa: E402
from hinglish_agent.safety import by_family, score_safety, summarise_safety  # noqa: E402


def load_tasks(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_model(model_id: str, adapter: str | None, four_bit: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    kwargs = {"device_map": "auto"}
    if torch.cuda.is_available():
        kwargs["dtype"] = torch.float16
        if four_bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
    else:
        kwargs["dtype"] = torch.float32
        print("WARNING: no CUDA device. This will be very slow; use Colab or Kaggle for real runs.")

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tokenizer


def make_model_call(model, tokenizer, max_new_tokens: int):
    import torch

    def call(messages: list[dict]) -> str:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        generated = output[0][inputs["input_ids"].shape[1] :]
        return tokenizer.decode(generated, skip_special_tokens=True).strip()

    return call


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="CohereLabs/tiny-aya-global")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--tasks", default=str(ROOT / "data" / "tasks.jsonl"))
    parser.add_argument("--out", default=str(ROOT / "runs" / "baseline"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--safety", action="store_true", help="score injection/underspecification tasks")
    parser.add_argument(
        "--hardened-prompt",
        action="store_true",
        help="add the tool-results-are-data rule to the system prompt",
    )
    args = parser.parse_args()

    tasks = load_tasks(Path(args.tasks))
    if args.limit:
        tasks = tasks[: args.limit]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model(args.model, args.adapter, four_bit=not args.no_4bit)
    call_model = make_model_call(model, tokenizer, args.max_new_tokens)

    scores = []
    started = time.time()
    with (out_dir / "trajectories.jsonl").open("w", encoding="utf-8") as trajectories, (
        out_dir / "results.jsonl"
    ).open("w", encoding="utf-8") as results:
        for index, task in enumerate(tasks, start=1):
            episode = run_episode(
                call_model, task, max_steps=args.max_steps, hardened=args.hardened_prompt
            )
            task_scores = score_safety(episode, task) if args.safety else score_episode(episode, task)
            scores.append(task_scores)
            trajectories.write(json.dumps(to_trajectory(episode, task_scores), ensure_ascii=False) + "\n")
            results.write(json.dumps(task_scores, ensure_ascii=False) + "\n")
            print(
                f"[{index}/{len(tasks)}] {task['id']} success={task_scores['success']:.0f} "
                f"calls={task_scores['calls']} errors={task_scores['errors']} "
                f"done={1 - task_scores['lcf']:.0f}",
                flush=True,
            )

    summary = summarise_safety(scores) if args.safety else summarise(scores)
    summary["model"] = args.model
    summary["adapter"] = args.adapter
    summary["hardened_prompt"] = bool(args.hardened_prompt)
    summary["seconds"] = round(time.time() - started, 1)
    if args.safety:
        summary["families"] = by_family(scores, tasks)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print(json.dumps(summary, indent=2))
    print()
    if not args.safety:
        print(markdown_table(summary))
    print(f"\nwrote {len(scores)} episodes -> {out_dir}")


if __name__ == "__main__":
    main()
