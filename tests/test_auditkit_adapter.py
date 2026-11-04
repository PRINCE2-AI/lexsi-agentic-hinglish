"""The AuditKIT adapter builds the shapes AuditKIT expects, without needing it installed."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import auditkit_eval as ak  # noqa: E402


TASK = {
    "id": "h-e-001",
    "task": "Amit ne 2003 INR diye groceries ke liye (Amit, Priya). Settle up karo.",
    "family": "settle5",
    "expected_balances": {"Amit": 1001.5, "Priya": -1001.5},
    "answer_contains": ["1001.5"],
    "answer_text": "Settlement: Priya -> Amit 1001.50 INR",
    "actions": [
        {"tool_name": "add_expense", "parameters": {"payer": "Amit", "amount": 2003,
                                                    "currency": "INR", "participants": ["Amit", "Priya"],
                                                    "description": "groceries"}},
        {"tool_name": "settle_up", "parameters": {}},
    ],
}

ROW = {
    "task": TASK["task"],
    "final_response": "Priya Amit ko 1001.5 INR dega.",
    "steps": [
        {"id": 0, "name": "add_expense",
         "arguments": {"payer": "Amit", "amount": 2003, "currency": "INR",
                       "participants": ["Amit", "Priya"], "description": "groceries",
                       "ledger": {"balances_cents": {"Amit": 0, "Priya": 0}, "expense_count": 0}},
         "result": {"ok": True, "ledger": {"balances_cents": {"Amit": 200300, "Priya": -100150},
                                           "expense_count": 1}}},
        {"id": 1, "name": "finish",
         "arguments": {"answer": "Priya Amit ko 1001.5 INR dega."},
         "result": {"ok": True, "done": True}},
    ],
    "metadata": {"task_id": TASK["id"], "scores": {"success": 1.0}},
}


def test_reference_drops_the_ledger_argument():
    calls = ak._reference(TASK, include_finish=False)
    assert [call["name"] for call in calls] == ["add_expense", "settle_up"]
    assert all("ledger" not in call["arguments"] for call in calls)


def test_reference_can_include_the_answer_turn():
    calls = ak._reference(TASK, include_finish=True)
    assert calls[-1]["name"] == "finish"
    assert calls[-1]["arguments"]["answer"] == TASK["answer_text"]


def test_messages_pair_every_call_with_its_result():
    messages = ak._messages(TASK, ROW)
    assert messages[0] == {"role": "user", "content": TASK["task"]}
    calls = [m["tool_calls"][0] for m in messages if m.get("tool_calls")]
    results = [m for m in messages if m.get("role") == "tool"]
    assert [call["id"] for call in calls] == [m["tool_call_id"] for m in results]
    assert messages[-1]["role"] == "assistant"
    # a ledger snapshot must never leak into the model-facing arguments
    assert "ledger" not in json.loads(calls[0]["function"]["arguments"])


def test_final_ledger_prefers_the_last_result_snapshot():
    ledger = ak._final_ledger(ROW)
    assert ledger["balances_cents"] == {"Amit": 200300, "Priya": -100150}
