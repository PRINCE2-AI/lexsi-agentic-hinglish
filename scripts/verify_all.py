"""Run every check that does not need a GPU or a token, and print one verdict.

    python scripts/verify_all.py

A judge should be able to clone the repo and run this before reading anything else.
Exit code is 0 only if every step passes.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STEPS = [
    ("hard split pin + oracle", ["data/build_hard.py"]),
    ("mix leak check", ["data/build_mix.py"]),
    ("safety split pin + oracle", ["data/build_safety.py"]),
    ("abstention split oracle", ["data/build_abstain.py"]),
    ("hard-90 split pin + oracle", ["data/build_hard90.py"]),
    ("mix2 leak check", ["data/build_mix.py", "--abstain", "--out", "data/tasks_train_mix2.jsonl"]),
    ("hard-data tests", ["tests/test_hard_data.py"]),
    ("safety tests", ["tests/test_safety.py"]),
    ("reward tests", ["tests/test_rewards.py"]),
    ("stats tests", ["tests/test_stats.py"]),
    ("scaffold tests", ["tests/test_scaffold.py"]),
    ("auditkit adapter tests", ["tests/test_auditkit_adapter.py"]),
    (
        "oracle on hard-30",
        ["scripts/run_oracle_eval.py", "--tasks", "data/tasks_hard_eval.jsonl", "--out", "runs/oracle"],
    ),
]


def main() -> None:
    failures = []
    for label, args in STEPS:
        result = subprocess.run(
            [sys.executable, *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        status = "PASS" if result.returncode == 0 else "FAIL"
        print(f"[{status}] {label}")
        if result.returncode != 0:
            failures.append(label)
            print(result.stdout[-2000:])
            print(result.stderr[-2000:])

    print()
    if failures:
        print(f"{len(failures)} of {len(STEPS)} checks failed: {', '.join(failures)}")
        raise SystemExit(1)
    print(f"all {len(STEPS)} checks passed - dataset, pins and scoring are consistent")


if __name__ == "__main__":
    main()
