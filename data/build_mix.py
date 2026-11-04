"""Build the GRPO training mix with family reweighting.

SFT on the easy split left three hard families at 0% (two_query 0/6,
convert_sum 0/3, empty_balance 2/3) even though tool calls and the ledger were
correct: the failures are answer completeness, a missing second conversion and
an invented expense. GRPO samples rows uniformly, so those families are repeated
here to give them a learnable share of the rollouts.

    python data/build_mix.py
    -> data/tasks_train_mix.jsonl  (184 rows: easy 100 + hard 84)

    python data/build_mix.py --abstain --out data/tasks_train_mix2.jsonl
    -> adds the abstention split (`data/build_abstain.py`) so SFT also sees what
       "info missing, ask" looks like

    python data/build_mix.py --abstain --contrast --out data/tasks_train_mix3.jsonl
    -> also adds the minimal pairs from `data/build_contrast.py`: an empty ledger
       is answerable (0 INR), only a missing field is a reason to ask
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Copies per hard family, one entry per source row.
COPIES = {
    "two_query": 3,      # 0/6: settlement reported, balance dropped
    "convert_sum": 4,    # 0/3: one conversion, second value invented
    "empty_balance": 3,  # 2/3: invented an expense on an empty ledger
    "trap_combo": 2,     # 5/6: payer folded into participants once
    "settle5": 1,
    "paraphrase": 1,
}


def load(name: str) -> list[dict]:
    with (ROOT / "data" / name).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "data" / "tasks_train_mix.jsonl"))
    parser.add_argument(
        "--abstain",
        nargs="?",
        const=str(ROOT / "data" / "tasks_train_abstain.jsonl"),
        default=None,
        help="append the abstention split from data/build_abstain.py",
    )
    parser.add_argument("--abstain-copies", type=int, default=3)
    parser.add_argument(
        "--contrast",
        nargs="?",
        const=str(ROOT / "data" / "tasks_train_contrast.jsonl"),
        default=None,
        help="append the minimal-pair split from data/build_contrast.py",
    )
    parser.add_argument("--contrast-copies", type=int, default=2)
    parser.add_argument(
        "--no-easy",
        action="store_true",
        help="drop the 100 easy rows; GRPO samples uniformly, so rows the SFT already "
        "solves give every group the same reward and no gradient",
    )
    args = parser.parse_args()

    easy = load("tasks_train.jsonl")
    hard = load("tasks_hard_train.jsonl")
    hard_eval = {
        task["task"].strip().lower()
        for name in ("tasks_hard_eval.jsonl", "tasks_hard90_eval.jsonl")
        for task in load(name)
    }

    rows = [] if args.no_easy else list(easy)
    for task in hard:
        if task["task"].strip().lower() in hard_eval:
            raise SystemExit(f"train row leaks into hard eval: {task['id']}")
        rows.extend([task] * COPIES.get(task.get("family", ""), 1))

    if args.abstain:
        abstain = [
            json.loads(line)
            for line in Path(args.abstain).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for task in abstain:
            if task["task"].strip().lower() in hard_eval:
                raise SystemExit(f"abstention row leaks into hard eval: {task['id']}")
            repeats = args.abstain_copies if task.get("abstain") else 1
            rows.extend([task] * repeats)

    if args.contrast:
        contrast = [
            json.loads(line)
            for line in Path(args.contrast).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for task in contrast:
            if task["task"].strip().lower() in hard_eval:
                raise SystemExit(f"contrast row leaks into hard eval: {task['id']}")
            copies = args.contrast_copies if task.get("family") == "contrast_empty" else 1
            rows.extend([task] * copies)

    with Path(args.out).open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    families = Counter(row.get("family", "easy") for row in rows)
    share = {name: round(count / len(rows), 3) for name, count in families.most_common()}
    print(f"wrote {len(rows)} rows -> {args.out}")
    print("family share:", json.dumps(share, sort_keys=True))


if __name__ == "__main__":
    main()
