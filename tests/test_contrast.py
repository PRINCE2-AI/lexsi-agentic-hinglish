"""Contrast-split checks. Run: python tests/test_contrast.py

The contrast rows exist to teach one boundary: an empty ledger is answerable
(0 INR), a missing amount/currency/share is not. If a contrast row repeated an
eval text, or if the oracle stopped reproducing its own answer, the split would
teach the wrong thing silently.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hinglish_agent.runner import run_oracle, score_episode  # noqa: E402

DATA = ROOT / "data"
FAMILIES = {"contrast_empty", "contrast_settle", "contrast_convert"}
SPLITS = (
    "tasks.jsonl",
    "tasks_train.jsonl",
    "tasks_eval.jsonl",
    "tasks_hard_train.jsonl",
    "tasks_hard_eval.jsonl",
    "tasks_hard90_eval.jsonl",
    "tasks_train_abstain.jsonl",
    "tasks_safety_eval.jsonl",
)


def load(name: str) -> list[dict]:
    with (DATA / name).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_shapes() -> None:
    rows = load("tasks_train_contrast.jsonl")
    assert len(rows) >= 40, len(rows)
    assert {row["family"] for row in rows} == FAMILIES
    assert all(row["require_answer"] for row in rows)
    for row in rows:
        assert row["id"].startswith("c-t-")
        if row["family"] == "contrast_empty":
            assert row["expected_tools"] == ["get_balance"], row["id"]
            assert "0" in row["answer_contains"], row["id"]
            assert row["expected_balances"] == {}, row["id"]
        if row["family"] == "contrast_settle":
            assert row["expected_tools"] == ["settle_up"], row["id"]
            assert row["expected_settlements"] == [], row["id"]


def test_no_leakage() -> None:
    mine = {row["task"].strip().lower() for row in load("tasks_train_contrast.jsonl")}
    for name in SPLITS:
        other = {row["task"].strip().lower() for row in load(name)}
        overlap = mine & other
        assert not overlap, f"contrast overlaps {name}: {list(overlap)[:2]}"


def test_oracle_is_clean() -> None:
    for row in load("tasks_train_contrast.jsonl"):
        scores = score_episode(run_oracle(row), row)
        assert scores["success"] == 1.0, (row["id"], scores)
        assert scores["ter"] == 0.0, (row["id"], scores)
        assert scores["lcf"] == 0.0, (row["id"], scores)


def test_mix3_is_built_from_it() -> None:
    mix = load("tasks_train_mix3.jsonl")
    ids = {row["id"] for row in mix}
    contrast_ids = {row["id"] for row in load("tasks_train_contrast.jsonl")}
    assert contrast_ids <= ids, sorted(contrast_ids - ids)[:5]
    hard_eval = {
        row["task"].strip().lower()
        for name in ("tasks_hard_eval.jsonl", "tasks_hard90_eval.jsonl")
        for row in load(name)
    }
    assert not [row for row in mix if row["task"].strip().lower() in hard_eval]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"[PASS] {name}")
