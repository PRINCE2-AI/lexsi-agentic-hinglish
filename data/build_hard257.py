"""Build a 257-task held-out hard split, for a tighter confidence interval.

hard-90 puts 93.33% in an 86.2–96.9% Wilson interval. This split is the same six families
sampled again from fresh seeds, at 50 tasks per family instead of 15, so the interval on the
same behaviour narrows and the two families that still fail get 50 graded rows instead of 15.

`empty_balance` contributes only 7: its pool is 5 phrasings x 6 names, and every other split
already took a share of those 30 strings. The cap is declared in the plan rather than hidden by
reusing text or by inventing phrasings for this one split, so the family definitions stay
identical to hard-90 and the per-family rates are directly comparable.

    python data/build_hard273.py
    -> data/tasks_hard257_eval.jsonl  (257, frozen by data/hard257_eval.sha256)

Text is deduplicated against hard-30, hard-90, both hard train files, the easy splits, the
abstention rows and all six injection splits, so no row here repeats a string the model or the
guard has ever seen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.build_hard import verify  # noqa: E402
from data.build_hard90 import BUILDERS90  # noqa: E402
from data.task_builder import finalize  # noqa: E402

PREFIX = "h27-"

PLAN257 = [
    ("settle5", 50),
    ("two_query", 50),
    ("trap_combo", 50),
    ("paraphrase", 50),
    ("convert_sum", 50),
    ("empty_balance", 7),
]

OUTPUT = "tasks_hard257_eval.jsonl"


def known_texts_all() -> set:
    """Every task string in `data/`, so a new row cannot repeat anything ever trained on or graded.

    Globbing rather than a fixed list: an earlier version missed `tasks_train_contrast.jsonl`,
    whose rows were the training data for the shipped adapter, and let one empty-ledger string
    through into the eval split.
    """
    texts: set = set()
    here = Path(__file__).resolve()
    for path in sorted(ROOT.glob("data/tasks*.jsonl")):
        if path.resolve() == here.with_name(OUTPUT):
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    texts.add(json.loads(line)["task"])
    if not texts:
        raise SystemExit("no task files found under data/")
    return texts


def build(seed: int, plan: list[tuple[str, int]]) -> list[dict]:
    rng = random.Random(seed)
    seen_text = known_texts_all()
    used_expenses: set = set()
    tasks = []
    for index, (kind, count) in enumerate(plan):
        made = 0
        attempts = 0
        while made < count:
            attempts += 1
            if attempts > count * 200:
                raise RuntimeError(f"could not generate {count} unique {kind} tasks")
            scenario = BUILDERS90[kind](rng, used_expenses)
            if scenario["task"] in seen_text:
                continue
            seen_text.add(scenario["task"])
            task = finalize(scenario)
            task["family"] = kind
            task["id"] = f"{PREFIX}{len(tasks) + 1:03d}"
            tasks.append(task)
            made += 1
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20261113)
    parser.add_argument("--out-dir", default=str(ROOT / "data"))
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()

    tasks = build(args.seed, PLAN257)
    verify(tasks, "hard-257 eval")

    out_dir = Path(args.out_dir)
    path = out_dir / OUTPUT
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for task in tasks:
            handle.write(json.dumps(task, ensure_ascii=False) + "\n")
    print(f"wrote {len(tasks)} -> {path}")

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    pin_path = out_dir / "hard257_eval.sha256"
    if args.freeze or not pin_path.exists():
        pin_path.write_text(f"{digest}  {OUTPUT}\n", encoding="utf-8", newline="\n")
        print(f"hard-257 eval sha256: {digest} (frozen)")
    else:
        pinned = pin_path.read_text(encoding="utf-8").split()[0]
        if pinned != digest:
            raise SystemExit(f"hard-257 eval drifted: pinned {pinned}, regenerated {digest}")
        print(f"hard-257 eval sha256: {digest} (matches pin)")

    families: dict[str, int] = {}
    for task in tasks:
        families[task["family"]] = families.get(task["family"], 0) + 1
    print("families:", json.dumps(families, sort_keys=True))


if __name__ == "__main__":
    main()
