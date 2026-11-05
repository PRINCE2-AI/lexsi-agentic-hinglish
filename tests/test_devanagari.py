"""Devanagari split checks. Run: python tests/test_devanagari.py

The write-up claims a Devanagari generation eval. That claim is only worth
something if the file is frozen, oracle-clean, and actually written in
Devanagari - a stray Latin sentence would quietly turn it into another
Hinglish split.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hinglish_agent.runner import run_oracle, score_episode  # noqa: E402

DATA = ROOT / "data"
PLAN = {"settle5": 6, "two_query": 6, "trap_combo": 6, "paraphrase": 6, "convert_sum": 3, "empty_balance": 3}
DEVANAGARI = re.compile(r"[\u0900-\u097F]")


def load() -> list[dict]:
    with (DATA / "tasks_devanagari_eval.jsonl").open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_frozen_hash() -> None:
    expected = (DATA / "devanagari_eval.sha256").read_text(encoding="utf-8").split()[0]
    actual = hashlib.sha256((DATA / "tasks_devanagari_eval.jsonl").read_bytes()).hexdigest()
    assert expected == actual, f"devanagari eval drifted: {expected} != {actual}"


def test_counts_and_families() -> None:
    tasks = load()
    assert len(tasks) == 30, len(tasks)
    counts: dict[str, int] = {}
    for task in tasks:
        counts[task["family"]] = counts.get(task["family"], 0) + 1
    assert counts == PLAN, counts
    assert len({task["id"] for task in tasks}) == 30
    assert len({task["task"] for task in tasks}) == 30


def test_every_prompt_is_devanagari() -> None:
    for task in load():
        assert DEVANAGARI.search(task["task"]), task["id"]
        latin_words = re.findall(r"[A-Za-z]{4,}", task["task"])
        # only the six names and the ISO currency codes may stay in Latin
        allowed = {"Rahul", "Priya", "Amit", "Neha", "Sara", "Vikram", "INR", "USD", "EUR"}
        assert set(latin_words) <= allowed, (task["id"], sorted(set(latin_words) - allowed))


def test_oracle_clean() -> None:
    for task in load():
        scores = score_episode(run_oracle(task), task)
        assert scores["success"] == 1.0, task["id"]
        assert scores["ter"] == 0.0, task["id"]
        assert scores["strict"] == 1.0, task["id"]


if __name__ == "__main__":
    test_frozen_hash()
    test_counts_and_families()
    test_every_prompt_is_devanagari()
    test_oracle_clean()
    print("devanagari checks passed")
