"""GRPO fine-tuning of Tiny Aya on the Hinglish expense-splitting tasks.

Colab/Kaggle (Python 3.12, CUDA, AgentTune installed):
    python scripts/train_grpo.py --check          # validate data only, no GPU needed
    python scripts/train_grpo.py --max-steps 10   # smoke test
    python scripts/train_grpo.py --max-steps 120  # real starter run

Tiny Aya is gated: accept the licence and run `hf auth login` first.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

# Must be set before torch initialises CUDA; reduces fragmentation on a 16 GB T4.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hinglish_agent.parser import parse_calls  # noqa: E402
from hinglish_agent.prompts import SYSTEM_PROMPT  # noqa: E402
from hinglish_agent.runner import answer_contains  # noqa: E402
from hinglish_agent.tools import TOOL_FUNCTIONS  # noqa: E402


def load_tasks(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_dataset(tasks: list[dict]):
    from datasets import Dataset

    rows = [
        {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": task["task"]},
            ],
            "answer": task["answer_text"],
            "expected_tools": task["expected_tools"],
        }
        for task in tasks
    ]
    return Dataset.from_list(rows)


def _text(value) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def format_reward(completions, **kwargs):
    return [1.0 if parse_calls(_text(item)) else 0.0 for item in completions]


def tool_reward(completions, expected_tools=None, **kwargs):
    if expected_tools is None:
        return [0.0 for _ in completions]
    scores = []
    for item, expected in zip(completions, expected_tools):
        names = {call["tool_name"] for call in (parse_calls(_text(item)) or [])}
        scores.append(1.0 if expected and all(name in names for name in expected) else 0.0)
    return scores


def efficiency_reward(completions, **kwargs):
    """Reward turns whose calls are all distinct; duplicate calls get no credit."""
    scores = []
    for item in completions:
        calls = parse_calls(_text(item)) or []
        keys = [json.dumps([call["tool_name"], call["arguments"]], sort_keys=True, default=str) for call in calls]
        scores.append(1.0 if keys and len(keys) == len(set(keys)) else 0.0)
    return scores


def answer_reward(completions, answer=None, **kwargs):
    if answer is None:
        return [0.0 for _ in completions]
    scores = []
    for item, expected in zip(completions, answer):
        numbers = re.findall(r"-?\d+(?:\.\d+)?", str(expected))
        scores.append(1.0 if numbers and answer_contains(_text(item), numbers) else 0.0)
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="CohereLabs/tiny-aya-global")
    parser.add_argument("--tasks", default=str(ROOT / "data" / "tasks.jsonl"))
    parser.add_argument("--out", default=str(ROOT / "runs" / "grpo_tiny_aya"))
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--max-completion-length", type=int, default=256)
    parser.add_argument("--max-steps-per-turn", type=int, default=3)
    parser.add_argument("--efficiency", action="store_true", help="add a no-duplicate-calls reward")
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16"],
        default="float16",
        help="model dtype; TRL defaults to float32, which OOMs a 16 GB T4",
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    tasks = load_tasks(Path(args.tasks))
    print(f"dataset rows: {len(tasks)}")
    print(f"first task: {tasks[0]['task']}")
    print(f"first answer: {tasks[0]['answer_text']}")
    print(f"expected tools: {tasks[0]['expected_tools']}")

    if args.check:
        return

    dataset = build_dataset(tasks)

    import torch
    from agenttune.core.backend_factory import create_agentic_trainer

    reward_funcs = [format_reward, tool_reward, answer_reward]
    if args.efficiency:
        reward_funcs.append(efficiency_reward)

    trainer = create_agentic_trainer(
        algorithm="grpo",
        model=args.model,
        train_dataset=dataset,
        tools=TOOL_FUNCTIONS,
        reward_funcs=reward_funcs,
        model_init_kwargs={"dtype": getattr(torch, args.dtype)},
        gradient_checkpointing=True,
        output_dir=args.out,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_generations=args.generations,
        max_completion_length=args.max_completion_length,
        max_steps_per_turn=args.max_steps_per_turn,
        use_vllm=False,
        peft_config=dict(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM"),
        log_completions=True,
        report_to="none",
    )

    result = trainer.train()
    print(f"done: {result}")

    try:
        trainer.save_model(args.out)
        print(f"adapter saved -> {args.out}")
    except Exception as exc:  # noqa: BLE001 - trainer versions differ
        print(f"save_model failed ({exc}); the trainer output_dir should still hold the adapter")

    print("\nNext: evaluate the adapter")
    print(
        f"python scripts/run_baseline.py --model {args.model} "
        f"--adapter {args.out} --out runs/tuned"
    )


if __name__ == "__main__":
    main()
