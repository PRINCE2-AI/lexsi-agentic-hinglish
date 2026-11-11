"""Run every check that does not need a GPU or a token, and print one verdict.

    python scripts/verify_all.py                # 28 offline checks
    python scripts/verify_all.py --with-tests   # the same checks plus the pytest suite

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
    (
        "safety2 split pin + oracle (72 attacks, second seed)",
        [
            "data/build_safety.py", "--seed", "20261108", "--name", "safety_eval2",
            "--plan", "inject_note:24,inject_skip:24,inject_settle:24,underspecified:8",
        ],
    ),
    (
        "payee split pin + oracle",
        ["data/build_safety.py", "--seed", "20261108", "--name", "safety_payee",
         "--plan", "inject_payee:16"],
    ),
    (
        "hinglish payload split pin + oracle (24 attacks in Hinglish/Devanagari)",
        ["data/build_safety_hinglish.py"],
    ),
    ("abstention split oracle", ["data/build_abstain.py"]),
    ("hard-90 split pin + oracle", ["data/build_hard90.py"]),
    ("devanagari split pin + oracle", ["data/build_devanagari.py"]),
    ("contrast split oracle", ["data/build_contrast.py"]),
    ("mix2 leak check", ["data/build_mix.py", "--abstain", "--out", "data/tasks_train_mix2.jsonl"]),
    (
        "mix3 leak check",
        ["data/build_mix.py", "--abstain", "--contrast", "--out", "data/tasks_train_mix3.jsonl"],
    ),
    (
        "mix4 leak check",
        [
            "data/build_mix.py", "--abstain", "--abstain-copies", "0",
            "--contrast", "--contrast-copies", "3", "--out", "data/tasks_train_mix4.jsonl",
        ],
    ),
    (
        "mix5 leak check",
        [
            "data/build_mix.py", "--abstain", "--abstain-copies", "1",
            "--contrast", "--contrast-copies", "3", "--out", "data/tasks_train_mix5.jsonl",
        ],
    ),
    (
        "grpo split leak check",
        [
            "data/build_mix.py", "--no-easy", "--abstain", "--abstain-copies", "1",
            "--contrast", "--contrast-copies", "3", "--out", "data/tasks_train_grpo.jsonl",
        ],
    ),
    ("hard-data tests", ["tests/test_hard_data.py"]),
    ("safety tests", ["tests/test_safety.py"]),
    ("guard tests", ["tests/test_guard.py"]),
    ("guard sandbox tests", ["tests/test_sandbox.py"]),
    ("contrast tests", ["tests/test_contrast.py"]),
    ("reward tests", ["tests/test_rewards.py"]),
    ("stats tests", ["tests/test_stats.py"]),
    ("scaffold tests", ["tests/test_scaffold.py"]),
    ("devanagari tests", ["tests/test_devanagari.py"]),
    ("schema-shift tests", ["tests/test_schema_shift.py"]),
    ("auditkit adapter tests", ["tests/test_auditkit_adapter.py"]),
    ("demo Space handlers", ["scripts/check_demo_space.py"]),
    (
        "oracle on hard-30",
        ["scripts/run_oracle_eval.py", "--tasks", "data/tasks_hard_eval.jsonl", "--out", "runs/oracle"],
    ),
]


def main() -> None:
    steps = STEPS
    if "--with-tests" in sys.argv[1:]:
        steps = [("unit tests (pytest)", ["-m", "pytest", "-q"]), *STEPS]
    failures = []
    for label, args in steps:
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
        print(f"{len(failures)} of {len(steps)} checks failed: {', '.join(failures)}")
        raise SystemExit(1)
    print(f"all {len(steps)} checks passed - dataset, pins and scoring are consistent")


if __name__ == "__main__":
    main()
