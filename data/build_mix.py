"""Build the GRPO training mix with family reweighting.

SFT on the easy split left three hard families at 0% (two_query 0/6,
convert_sum 0/3, empty_balance 2/3) even though tool calls and the ledger were
correct: the failures are answer completeness, a missing second conversion and
an invented expense. GRPO samples rows uniformly, so those families are repeated
here to give them a learnable share of the rollouts.

    python data/build_mix.py
    -> data/tasks_train_mix.jsonl  (184 rows: easy 100 + hard 84)
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
    args = parser.parse_args()

    easy = load("tasks_train.jsonl")
    hard = load("tasks_hard_train.jsonl")
    hard_eval = {task["task"].strip().lower() for task in load("tasks_hard_eval.jsonl")}

    rows = list(easy)
    for task in hard:
        if task["task"].strip().lower() in hard_eval:
            raise SystemExit(f"train row leaks into hard eval: {task['id']}")
        rows.extend([task] * COPIES.get(task.get("family", ""), 1))

    with Path(args.out).open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    families = Counter(row.get("family", "easy") for row in rows)
    share = {name: round(count / len(rows), 3) for name, count in families.most_common()}
    print(f"wrote {len(rows)} rows -> {args.out}")
    print("family share:", json.dumps(share, sort_keys=True))


if __name__ == "__main__":
    main()
