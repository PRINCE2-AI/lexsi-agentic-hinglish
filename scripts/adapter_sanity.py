"""Does a Hub adapter actually change the model? Three tasks, one verdict.

Stage 1's `sft2` row came back at ~0/60 on hard-90 while the same adapter scored
29/30 on hard-30, which is the signature of a merge that silently did nothing.
This script loads base + adapter through the exact Hub path a job uses, then runs
three easy tasks and prints the tool calls, so "the adapter is applied" is a fact
instead of an assumption.

    python scripts/adapter_sanity.py --hub-adapter primce/tiny-aya-hinglish-lora:sft2_adapter
    python scripts/adapter_sanity.py --merged runs/sft4_merged
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from hinglish_agent.runner import run_episode, score_episode  # noqa: E402
from run_baseline import load_tasks, make_model_call  # noqa: E402


def load_merged(base: str, repo: str | None, subfolder: str | None, merged: str | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(merged or base)
    model = AutoModelForCausalLM.from_pretrained(
        merged or base,
        dtype=torch.float16,
        device_map="auto",
    )
    tag = merged or base
    if repo:
        from huggingface_hub import snapshot_download
        from peft import PeftModel

        with tempfile.TemporaryDirectory() as tmp:
            local = snapshot_download(
                repo,
                allow_patterns=[f"{subfolder}/*"],
                local_dir=f"{tmp}/repo",
            )
            model = PeftModel.from_pretrained(model, local, subfolder=subfolder)
            model = model.merge_and_unload()
        tag = f"{base} + {repo}:{subfolder} (merged)"
    model.eval()
    return model, tokenizer, tag


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="CohereLabs/tiny-aya-global")
    parser.add_argument("--hub-adapter", default=None, help="REPO:SUBFOLDER")
    parser.add_argument("--merged", default=None, help="local merged model dir")
    parser.add_argument("--tasks", default=str(ROOT / "data" / "tasks_eval.jsonl"))
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    repo = subfolder = None
    if args.hub_adapter:
        repo, subfolder = args.hub_adapter.split(":")

    model, tokenizer, tag = load_merged(args.base, repo, subfolder, args.merged)
    call = make_model_call(model, tokenizer, 256)
    tasks = load_tasks(Path(args.tasks))[: args.limit]

    print(f"model: {tag}")
    hits = 0
    for task in tasks:
        episode = run_episode(call, task, max_steps=8)
        scores = score_episode(episode, task)
        hits += int(scores["success"])
        print(
            f"{task['id']} success={scores['success']:.0f} calls={scores['calls']} "
            f"errors={scores['errors']} strict={scores['strict']:.0f}"
        )
        print("  calls:", [step["name"] for step in episode["steps"]])
        print("  answer:", (episode["final_response"] or "")[:160].replace("\n", " "))
    print(f"\n{tag}: {hits}/{len(tasks)} on {Path(args.tasks).name}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps({"model": tag, "hits": hits, "n": len(tasks)}, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
