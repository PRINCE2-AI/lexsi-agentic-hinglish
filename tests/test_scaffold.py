"""No-dependency sanity tests. Run: python tests/test_scaffold.py"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hinglish_agent.parser import parse_calls  # noqa: E402
from hinglish_agent.runner import (  # noqa: E402
    answer_contains,
    run_episode,
    run_oracle,
    score_episode,
)


class Scripted:
    def __init__(self, outputs: list[str]):
        self.outputs = outputs
        self.index = 0

    def __call__(self, messages):
        output = self.outputs[min(self.index, len(self.outputs) - 1)]
        self.index += 1
        return output


def load_tasks() -> list[dict]:
    with (ROOT / "data" / "tasks.jsonl").open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_parser() -> None:
    fenced = '```json\n[{"tool_name":"finish","parameters":{"answer":"900"}}]\n```'
    assert parse_calls(fenced)[0]["tool_name"] == "finish"
    renamed = 'Sure: [{"name":"get_balance","arguments":{"person":"Rahul"}}]'
    assert parse_calls(renamed)[0]["parameters"]["person"] == "Rahul"
    assert parse_calls("no json here") is None
    assert answer_contains("Rahul ko 900.00 INR milne hain", ["900"])
    assert answer_contains("1,660 INR", ["1660"])
    assert not answer_contains("1900 INR", ["900"])


def test_agent_loop() -> None:
    task = load_tasks()[0]

    good = [
        '[{"tool_name":"add_expense","parameters":{"ledger":{"balances_cents":{},"expense_count":0},"payer":"Rahul","amount":1200,"currency":"INR","participants":["Rahul","Priya","Amit","Neha"],"description":"pizza"}}]',
        '[{"tool_name":"get_balance","parameters":{"ledger":{"balances_cents":{},"expense_count":0},"person":"Rahul"}}]',
        '[{"tool_name":"finish","parameters":{"answer":"Rahul ka balance 900.00 INR hai."}}]',
    ]
    scores = score_episode(run_episode(Scripted(good), task), task)
    assert scores["success"] == 1.0 and scores["tac"] == 1.0 and scores["ter"] == 0.0
    assert scores["strict"] == 1.0 and scores["strict_rate"] == 1.0

    bad = score_episode(run_episode(Scripted(["I cannot help with that."]), task), task)
    assert bad["success"] == 0.0 and bad["lcf"] == 1.0 and bad["ter"] == 1.0
    assert bad["strict"] == 0.0 and bad["strict_rate"] == 0.0

    fenced = score_episode(
        run_episode(Scripted(['```json\n[{"tool_name":"finish","parameters":{"answer":"900"}}]\n```']), task),
        task,
    )
    assert fenced["strict"] == 0.0 and fenced["strict_rate"] == 0.0

    oracle = score_episode(run_oracle(task), task)
    assert oracle["success"] == 1.0 and oracle["strict"] == 1.0

    wrong = [
        '[{"tool_name":"add_expense","parameters":{"ledger":{"balances_cents":{},"expense_count":0},"payer":"Rahul","amount":1200,"currency":"INR","participants":["Rahul","Priya"]}}]',
        '[{"tool_name":"finish","parameters":{"answer":"600"}}]',
    ]
    wrong_scores = score_episode(run_episode(Scripted(wrong), task), task)
    assert wrong_scores["success"] == 0.0


def test_sft_traces() -> None:
    from scripts.train_sft import build_conversations

    rows = build_conversations(load_tasks())
    assert len(rows) == 20
    for row in rows:
        for index in row["targets"]:
            message = row["messages"][index]
            assert message["role"] == "assistant"
            assert parse_calls(message["content"]) is not None
        assert parse_calls(row["messages"][row["targets"][-1]]["content"])[0]["tool_name"] == "finish"

    first = rows[0]
    balance_call = parse_calls(first["messages"][first["targets"][1]]["content"])[0]
    assert balance_call["tool_name"] == "get_balance"
    assert balance_call["parameters"]["ledger"]["expense_count"] == 1


def main() -> None:
    test_parser()
    test_agent_loop()
    test_sft_traces()
    print("all scaffold tests passed")


if __name__ == "__main__":
    main()
