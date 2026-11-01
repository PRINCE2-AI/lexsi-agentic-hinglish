"""Verify the dataset with the oracle and print the metric table.

Run: python scripts/run_oracle_eval.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hinglish_agent.metrics import markdown_table, summarise  # noqa: E402
from hinglish_agent.runner import run_oracle, score_episode, to_trajectory  # noqa: E402


def load_tasks(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default=str(ROOT / "data" / "tasks.jsonl"))
    parser.add_argument("--out", default=str(ROOT / "runs" / "oracle"))
    args = parser.parse_args()

    tasks = load_tasks(Path(args.tasks))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    scores = []
    with (out_dir / "trajectories.jsonl").open("w", encoding="utf-8") as handle:
        for task in tasks:
            episode = run_oracle(task)
            task_scores = score_episode(episode, task)
            scores.append(task_scores)
            handle.write(json.dumps(to_trajectory(episode, task_scores), ensure_ascii=False) + "\n")

    summary = summarise(scores)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print()
    print(markdown_table(summary))
    print(f"\nwrote {len(scores)} trajectories -> {out_dir}")


if __name__ == "__main__":
    main()
