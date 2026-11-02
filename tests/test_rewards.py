"""Graded-reward checks. Run: python tests/test_rewards.py

These are the numbers GRPO optimises, so they are pinned: a correct call must
score 7.0, a fenced one loses only the format term, and the answer term is worth
2.0 on its own.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hinglish_agent.parser import format_call  # noqa: E402
from hinglish_agent.rewards import MAX_REWARD, grade_calls, grade_text  # noqa: E402
from hinglish_agent.runner import run_oracle  # noqa: E402

TASK = {
    "actions": [
        {"tool_name": "add_expense", "parameters": {
            "payer": "Rahul", "amount": 1500, "currency": "INR",
            "participants": ["Priya", "Amit", "Neha"], "description": "pizza",
        }},
        {"tool_name": "get_balance", "parameters": {"person": "Priya"}},
    ],
    "answer_contains": ["250.0"],
}


def single(call: dict) -> str:
    return json.dumps([call], ensure_ascii=False)


def test_exact_call_scores_full_tool_credit() -> None:
    text = single(TASK["actions"][0])
    parts = grade_text(text, TASK["actions"], TASK["answer_contains"])
    assert parts["format"] == 1.0, parts
    assert parts["name"] == 0.5, parts  # one of two oracle actions
    assert parts["param_name"] == 0.5, parts
    assert parts["param_value"] == 0.5, parts
    assert parts["total"] == 4.0, parts


def test_argument_value_is_worth_more_than_nothing() -> None:
    right = dict(TASK["actions"][0])
    wrong = dict(TASK["actions"][0], parameters=dict(TASK["actions"][0]["parameters"], amount=999))
    right_score = grade_calls([right], TASK["actions"][:1])
    wrong_score = grade_calls([wrong], TASK["actions"][:1])
    assert right_score["name"] == wrong_score["name"] == 1.0
    assert right_score["param_value"] == 1.0, right_score
    assert wrong_score["param_value"] == 0.8, wrong_score
    assert right_score["param_value"] > wrong_score["param_value"]


def test_wrong_tool_gets_no_parameter_credit() -> None:
    text = single({"tool_name": "settle_up", "parameters": {"ledger": {}}})
    parts = grade_text(text, TASK["actions"], TASK["answer_contains"])
    assert parts["name"] == 0.0, parts
    assert parts["total"] == 1.0, parts  # format only


def test_fenced_json_loses_only_format() -> None:
    body = single(TASK["actions"][0])
    parts = grade_text("```json\n" + body + "\n```", TASK["actions"], TASK["answer_contains"])
    assert parts["format"] == 0.5, parts
    assert parts["name"] == 0.5, parts


def test_prose_scores_zero() -> None:
    parts = grade_text("Rahul ka balance 250 rupees hai.", TASK["actions"], TASK["answer_contains"])
    assert parts["total"] == 0.0, parts


def test_answer_term_needs_the_number() -> None:
    finish_ok = single({"tool_name": "finish", "parameters": {"answer": "Priya ka balance 250.00 INR hai."}})
    finish_bad = single({"tool_name": "finish", "parameters": {"answer": "Ho gaya."}})
    assert grade_text(finish_ok, TASK["actions"], TASK["answer_contains"])["answer"] == 1.0
    assert grade_text(finish_bad, TASK["actions"], TASK["answer_contains"])["answer"] == 0.0


def test_full_oracle_trajectory_hits_the_ceiling() -> None:
    episode = run_oracle({
        "id": "unit",
        "task": "unit",
        "actions": TASK["actions"],
        "answer_text": "Priya ka balance 250.00 INR hai.",
        "answer_contains": ["250"],
    })
    text = "\n".join(
        message["content"] for message in episode["messages"] if message["role"] == "assistant"
    )
    parts = grade_text(text, TASK["actions"], ["250"])
    assert parts["name"] == 1.0, parts
    assert parts["param_value"] == 1.0, parts
    assert parts["answer"] == 1.0, parts
    assert parts["total"] == MAX_REWARD - 0.5, parts  # format 0.5: multi-turn text is not bare JSON


def main() -> None:
    test_exact_call_scores_full_tool_credit()
    test_argument_value_is_worth_more_than_nothing()
    test_wrong_tool_gets_no_parameter_credit()
    test_fenced_json_loses_only_format()
    test_prose_scores_zero()
    test_answer_term_needs_the_number()
    test_full_oracle_trajectory_hits_the_ceiling()
    print("all reward tests passed")


if __name__ == "__main__":
    main()
