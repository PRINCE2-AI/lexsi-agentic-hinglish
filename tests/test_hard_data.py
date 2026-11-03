"""Hard-split checks. Run: python tests/test_hard_data.py

The hard split is the out-of-distribution benchmark, so it has to be frozen and
oracle-clean: a drifted eval file would silently change every reported number.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hinglish_agent.runner import run_oracle, score_episode  # noqa: E402

DATA = ROOT / "data"
FAMILIES = {"settle5", "two_query", "trap_combo", "paraphrase", "convert_sum", "empty_balance"}


def load(name: str) -> list[dict]:
    with (DATA / name).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_frozen_hash() -> None:
    expected = (DATA / "hard_eval.sha256").read_text(encoding="utf-8").split()[0]
    actual = hashlib.sha256((DATA / "tasks_hard_eval.jsonl").read_bytes()).hexdigest()
    assert expected == actual, f"hard eval drifted: {expected} != {actual}"


def test_counts_and_families() -> None:
    eval_tasks = load("tasks_hard_eval.jsonl")
    train_tasks = load("tasks_hard_train.jsonl")
    assert len(eval_tasks) == 30, len(eval_tasks)
    assert len(train_tasks) == 40, len(train_tasks)
    for tasks, label in ((eval_tasks, "eval"), (train_tasks, "train")):
        seen = {task["family"] for task in tasks}
        assert seen == FAMILIES, f"{label} families: {seen}"
        assert all(task["require_answer"] for task in tasks), f"{label} missing require_answer"


def test_no_leakage() -> None:
    def texts(name: str) -> set[str]:
        return {task["task"].strip().lower() for task in load(name)}

    hard = texts("tasks_hard_eval.jsonl") | texts("tasks_hard_train.jsonl")
    easy = texts("tasks_train.jsonl") | texts("tasks_eval.jsonl") | texts("tasks.jsonl")
    overlap = hard & easy
    assert not overlap, f"hard/easy overlap: {list(overlap)[:3]}"
    assert len(hard) == 70, f"hard duplicate texts: {70 - len(hard)}"


def test_oracle_is_clean() -> None:
    for name in ("tasks_hard_eval.jsonl", "tasks_hard_train.jsonl"):
        for task in load(name):
            scores = score_episode(run_oracle(task), task)
            assert scores["success"] == 1.0, (task["id"], scores)
            assert scores["ter"] == 0.0, (task["id"], scores)
            assert scores["strict"] == 1.0, (task["id"], scores)
            assert scores["overcall"] == 0.0, (task["id"], scores)
            assert scores["parse_failures"] == 0, (task["id"], scores)


def test_overcall_and_parse_failures() -> None:
    task = load("tasks_hard_eval.jsonl")[0]
    episode = run_oracle(task)

    extra = dict(
        episode,
        steps=episode["steps"] + [{"name": "get_balance", "arguments": {}, "result": {"ok": True}}],
    )
    assert score_episode(extra, task)["overcall"] > 0.0

    broken = dict(
        episode,
        steps=episode["steps"] + [{"name": None, "arguments": {}, "result": {"ok": False}}],
    )
    broken_scores = score_episode(broken, task)
    assert broken_scores["parse_failures"] == 1, broken_scores
    assert broken_scores["calls"] == len(episode["steps"]), broken_scores


def test_require_answer_gates_success() -> None:
    task = next(t for t in load("tasks_hard_eval.jsonl") if t["checks_ledger"])
    episode = run_oracle(task)

    right = score_episode(dict(episode), task)
    assert right["success"] == 1.0

    silent = dict(episode, final_response="Ho gaya.")
    wrong = score_episode(silent, task)
    assert wrong["ledger_ok"] == 1.0, wrong
    assert wrong["answer_ok"] == 0.0, wrong
    assert wrong["success"] == 0.0, "ledger-only success leaked past require_answer"


def main() -> None:
    test_frozen_hash()
    test_counts_and_families()
    test_no_leakage()
    test_oracle_is_clean()
    test_overcall_and_parse_failures()
    test_require_answer_gates_success()
    print("all hard-data tests passed")


if __name__ == "__main__":
    main()
