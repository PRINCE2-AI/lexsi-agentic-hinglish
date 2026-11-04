"""Split a run's hard-90 score by "was this wording in training?".

hard-90 is held-out *instances* of the six families, so the same template can appear
in both splits with a different name and amount. That is fine for a task-success
number, but it is not evidence that the model generalises to new wording. Two of
the five `empty_balance` phrasings exist only in hard-90, so this script separates
them.

    python scripts/template_split.py --run sft2_hard90 --train data/tasks_train_mix.jsonl
    python scripts/template_split.py --run sft4_hard90 --train data/tasks_train_mix3.jsonl

A task counts as "seen wording" if its text, with the person names masked, matches
the masked text of any training row.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = ("Rahul", "Priya", "Amit", "Neha", "Sara", "Vikram")


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def mask(text: str) -> str:
    masked = text
    for name in NAMES:
        masked = masked.replace(name, "{person}")
    return re.sub(r"\s+", " ", masked.strip().lower())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="run folder under runs/")
    parser.add_argument("--tasks", default=str(ROOT / "data" / "tasks_hard90_eval.jsonl"))
    parser.add_argument("--train", required=True, help="training jsonl used by the adapter")
    args = parser.parse_args()

    tasks = {task["id"]: task for task in load(Path(args.tasks))}
    seen = {mask(row["task"]) for row in load(Path(args.train))}

    rows = load(ROOT / "runs" / args.run / "trajectories.jsonl")
    buckets: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in rows:
        score = row["metadata"]["scores"]
        task = tasks[row["metadata"]["task_id"]]
        wording = "seen" if mask(task["task"]) in seen else "unseen"
        buckets[(task["family"], wording)].append(int(score["success"]))

    print(f"run: {args.run}  train: {Path(args.train).name}")
    print("| family | seen wording | unseen wording |")
    print("|---|---:|---:|")
    for family in sorted({key[0] for key in buckets}):
        cells = []
        for wording in ("seen", "unseen"):
            hits = buckets.get((family, wording), [])
            cells.append(f"{sum(hits)}/{len(hits)}" if hits else "-")
        print(f"| {family} | {cells[0]} | {cells[1]} |")

    for wording in ("seen", "unseen"):
        hits = [value for (family, kind), values in buckets.items() if kind == wording for value in values]
        if hits:
            print(f"{wording} wording total: {sum(hits)}/{len(hits)} ({100 * sum(hits) / len(hits):.2f}%)")
        else:
            print(f"{wording} wording total: -")


if __name__ == "__main__":
    main()
