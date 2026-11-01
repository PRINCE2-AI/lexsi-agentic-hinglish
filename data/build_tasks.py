"""Build data/tasks.jsonl and verify every task against the oracle.

Run: python data/build_tasks.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hinglish_agent.ledger import (  # noqa: E402
    balance_inr,
    empty_ledger,
    public_balances,
    settle_ledger,
)
from hinglish_agent.runner import run_oracle, score_episode  # noqa: E402
from hinglish_agent.tools import execute_tool  # noqa: E402


def A(name: str, **parameters) -> dict:
    return {"tool_name": name, "parameters": parameters}


SCENARIOS = [
    {
        "id": "t01",
        "task": "Rahul ne 1200 INR diye pizza ke liye. 4 log the: Rahul, Priya, Amit aur Neha. Rahul ka final balance kitna hai?",
        "actions": [
            A("add_expense", payer="Rahul", amount=1200, currency="INR", participants=["Rahul", "Priya", "Amit", "Neha"], description="pizza"),
            A("get_balance", person="Rahul"),
        ],
        "expected_tools": ["add_expense", "get_balance"],
        "query": {"type": "balance", "person": "Rahul"},
    },
    {
        "id": "t02",
        "task": "Priya ne 30 USD diye cab ke liye. 3 log the: Priya, Amit aur Sara. Priya ka balance batao.",
        "actions": [
            A("add_expense", payer="Priya", amount=30, currency="USD", participants=["Priya", "Amit", "Sara"], description="cab"),
            A("get_balance", person="Priya"),
        ],
        "expected_tools": ["add_expense", "get_balance"],
        "query": {"type": "balance", "person": "Priya"},
    },
    {
        "id": "t03",
        "task": "Amit ne 90 EUR ka dinner diya. 4 log the: Amit, Neha, Rahul aur Vikram. Amit ka balance kitna hai?",
        "actions": [
            A("add_expense", payer="Amit", amount=90, currency="EUR", participants=["Amit", "Neha", "Rahul", "Vikram"], description="dinner"),
            A("get_balance", person="Amit"),
        ],
        "expected_tools": ["add_expense", "get_balance"],
        "query": {"type": "balance", "person": "Amit"},
    },
    {
        "id": "t04",
        "task": "Rahul ne 1200 INR pizza ke liye diye (Rahul, Priya, Amit, Neha). Priya ne 30 USD cab ke liye diye (Priya, Amit, Sara). Rahul ka final balance kitna hai?",
        "actions": [
            A("add_expense", payer="Rahul", amount=1200, currency="INR", participants=["Rahul", "Priya", "Amit", "Neha"], description="pizza"),
            A("add_expense", payer="Priya", amount=30, currency="USD", participants=["Priya", "Amit", "Sara"], description="cab"),
            A("get_balance", person="Rahul"),
        ],
        "expected_tools": ["add_expense", "get_balance"],
        "query": {"type": "balance", "person": "Rahul"},
    },
    {
        "id": "t05",
        "task": "Rahul ne 1000 INR diye Rahul aur Priya ke liye. Priya ne 500 INR diye Rahul aur Priya ke liye. Settle up karo.",
        "actions": [
            A("add_expense", payer="Rahul", amount=1000, currency="INR", participants=["Rahul", "Priya"], description="groceries"),
            A("add_expense", payer="Priya", amount=500, currency="INR", participants=["Rahul", "Priya"], description="snacks"),
            A("settle_up"),
        ],
        "expected_tools": ["add_expense", "settle_up"],
        "query": {"type": "settle"},
    },
    {
        "id": "t06",
        "task": "30 USD kitne INR hote hain?",
        "actions": [A("convert_currency", amount=30, from_currency="USD", to_currency="INR")],
        "expected_tools": ["convert_currency"],
        "query": {"type": "convert", "amount": 30, "from": "USD", "to": "INR"},
    },
    {
        "id": "t07",
        "task": "25 EUR ko INR me convert karo.",
        "actions": [A("convert_currency", amount=25, from_currency="EUR", to_currency="INR")],
        "expected_tools": ["convert_currency"],
        "query": {"type": "convert", "amount": 25, "from": "EUR", "to": "INR"},
    },
    {
        "id": "t08",
        "task": "Rahul ne 900 INR diye Priya, Amit aur Neha ke liye. Rahul khud share nahi kar raha. Rahul ka balance kitna hai?",
        "actions": [
            A("add_expense", payer="Rahul", amount=900, currency="INR", participants=["Priya", "Amit", "Neha"], description="movie tickets"),
            A("get_balance", person="Rahul"),
        ],
        "expected_tools": ["add_expense", "get_balance"],
        "query": {"type": "balance", "person": "Rahul"},
    },
    {
        "id": "t09",
        "task": "Rahul ne 1000 INR diye Rahul, Priya aur Amit ke liye. Settle up karo.",
        "actions": [
            A("add_expense", payer="Rahul", amount=1000, currency="INR", participants=["Rahul", "Priya", "Amit"], description="petrol"),
            A("settle_up"),
        ],
        "expected_tools": ["add_expense", "settle_up"],
        "query": {"type": "settle"},
    },
    {
        "id": "t10",
        "task": "Trip ke expenses: Rahul ne 3000 INR diye (Rahul, Priya, Amit, Neha), Priya ne 60 USD diye (Rahul, Priya, Amit, Neha), Amit ne 50 EUR diye (Rahul, Priya, Amit, Neha). Settle up karo.",
        "actions": [
            A("add_expense", payer="Rahul", amount=3000, currency="INR", participants=["Rahul", "Priya", "Amit", "Neha"], description="hotel"),
            A("add_expense", payer="Priya", amount=60, currency="USD", participants=["Rahul", "Priya", "Amit", "Neha"], description="flights"),
            A("add_expense", payer="Amit", amount=50, currency="EUR", participants=["Rahul", "Priya", "Amit", "Neha"], description="food"),
            A("settle_up"),
        ],
        "expected_tools": ["add_expense", "settle_up"],
        "query": {"type": "settle"},
    },
    {
        "id": "t11",
        "task": "Sara ne 50 USD diye Sara aur Vikram ke liye. Vikram ne 25 EUR diye Sara aur Vikram ke liye. Settle up karo.",
        "actions": [
            A("add_expense", payer="Sara", amount=50, currency="USD", participants=["Sara", "Vikram"], description="concert"),
            A("add_expense", payer="Vikram", amount=25, currency="EUR", participants=["Sara", "Vikram"], description="dinner"),
            A("settle_up"),
        ],
        "expected_tools": ["add_expense", "settle_up"],
        "query": {"type": "settle"},
    },
    {
        "id": "t12",
        "task": "Rahul ne 1200 INR diye Priya, Amit aur Neha ke liye. Sara is trip me shamil nahi thi. Sara ka balance kya hai?",
        "actions": [
            A("add_expense", payer="Rahul", amount=1200, currency="INR", participants=["Priya", "Amit", "Neha"], description="pizza"),
            A("get_balance", person="Sara"),
        ],
        "expected_tools": ["add_expense", "get_balance"],
        "query": {"type": "balance", "person": "Sara"},
    },
    {
        "id": "t13",
        "task": "Rahul ne 500 INR sirf apne khane ke liye diye. Rahul ka balance kitna hai?",
        "actions": [
            A("add_expense", payer="Rahul", amount=500, currency="INR", participants=["Rahul"], description="lunch"),
            A("get_balance", person="Rahul"),
        ],
        "expected_tools": ["add_expense", "get_balance"],
        "query": {"type": "balance", "person": "Rahul"},
    },
    {
        "id": "t14",
        "task": "Neha ne 2000 INR diye (Neha, Rahul, Priya), Rahul ne 40 USD diye (Neha, Rahul, Priya), Priya ne 1500 INR diye (Neha, Rahul, Priya). Settle up karo.",
        "actions": [
            A("add_expense", payer="Neha", amount=2000, currency="INR", participants=["Neha", "Rahul", "Priya"], description="rent"),
            A("add_expense", payer="Rahul", amount=40, currency="USD", participants=["Neha", "Rahul", "Priya"], description="groceries"),
            A("add_expense", payer="Priya", amount=1500, currency="INR", participants=["Neha", "Rahul", "Priya"], description="electricity"),
            A("settle_up"),
        ],
        "expected_tools": ["add_expense", "settle_up"],
        "query": {"type": "settle"},
    },
    {
        "id": "t15",
        "task": "Vikram ne 80 EUR diye Vikram, Sara, Rahul aur Neha ke liye. Neha ka balance kitna hai?",
        "actions": [
            A("add_expense", payer="Vikram", amount=80, currency="EUR", participants=["Vikram", "Sara", "Rahul", "Neha"], description="resort"),
            A("get_balance", person="Neha"),
        ],
        "expected_tools": ["add_expense", "get_balance"],
        "query": {"type": "balance", "person": "Neha"},
    },
    {
        "id": "t16",
        "task": "8300 INR kitne USD hain?",
        "actions": [A("convert_currency", amount=8300, from_currency="INR", to_currency="USD")],
        "expected_tools": ["convert_currency"],
        "query": {"type": "convert", "amount": 8300, "from": "INR", "to": "USD"},
    },
    {
        "id": "t17",
        "task": "Priya ne 1000 INR diye Rahul aur Priya ke liye. Rahul ne 2000 INR diye Rahul aur Priya ke liye. Priya ka balance kitna hai?",
        "actions": [
            A("add_expense", payer="Priya", amount=1000, currency="INR", participants=["Rahul", "Priya"], description="wifi"),
            A("add_expense", payer="Rahul", amount=2000, currency="INR", participants=["Rahul", "Priya"], description="furniture"),
            A("get_balance", person="Priya"),
        ],
        "expected_tools": ["add_expense", "get_balance"],
        "query": {"type": "balance", "person": "Priya"},
    },
    {
        "id": "t18",
        "task": "Amit ne 1200 INR diye (Amit, Sara, Vikram), Sara ne 900 INR diye (Amit, Sara, Vikram), Vikram ne 30 USD diye (Amit, Sara, Vikram). Settle up karo.",
        "actions": [
            A("add_expense", payer="Amit", amount=1200, currency="INR", participants=["Amit", "Sara", "Vikram"], description="brunch"),
            A("add_expense", payer="Sara", amount=900, currency="INR", participants=["Amit", "Sara", "Vikram"], description="museum"),
            A("add_expense", payer="Vikram", amount=30, currency="USD", participants=["Amit", "Sara", "Vikram"], description="taxi"),
            A("settle_up"),
        ],
        "expected_tools": ["add_expense", "settle_up"],
        "query": {"type": "settle"},
    },
    {
        "id": "t19",
        "task": "Rahul ne 1500 INR diye (Rahul, Priya, Amit, Neha), Priya ne 45 USD diye (Priya, Amit, Neha), Amit ne 20 EUR diye (Rahul, Amit), Neha ne 600 INR diye (Neha, Rahul). Settle up karo.",
        "actions": [
            A("add_expense", payer="Rahul", amount=1500, currency="INR", participants=["Rahul", "Priya", "Amit", "Neha"], description="party"),
            A("add_expense", payer="Priya", amount=45, currency="USD", participants=["Priya", "Amit", "Neha"], description="decor"),
            A("add_expense", payer="Amit", amount=20, currency="EUR", participants=["Rahul", "Amit"], description="cake"),
            A("add_expense", payer="Neha", amount=600, currency="INR", participants=["Neha", "Rahul"], description="drinks"),
            A("settle_up"),
        ],
        "expected_tools": ["add_expense", "settle_up"],
        "query": {"type": "settle"},
    },
    {
        "id": "t20",
        "task": "Rahul ne 30 USD diye 3 logon ke liye: Rahul, Amit aur Sara. Rahul ka final balance kitna hai?",
        "actions": [
            A("add_expense", payer="Rahul", amount=30, currency="USD", participants=["Rahul", "Amit", "Sara"], description="tickets"),
            A("get_balance", person="Rahul"),
        ],
        "expected_tools": ["add_expense", "get_balance"],
        "query": {"type": "balance", "person": "Rahul"},
    },
]


def fmt(value: float) -> str:
    return f"{float(value):.2f}".rstrip("0").rstrip(".")


def build() -> list[dict]:
    tasks = []
    for scenario in SCENARIOS:
        ledger = empty_ledger()
        for action in scenario["actions"]:
            arguments = dict(action["parameters"])
            if action["tool_name"] in {"add_expense", "get_balance", "settle_up"}:
                arguments["ledger"] = ledger
            result = execute_tool(action["tool_name"], arguments)
            if not result.get("ok"):
                raise RuntimeError(f"{scenario['id']}: {result}")
            if isinstance(result.get("ledger"), dict):
                ledger = result["ledger"]

        query = scenario["query"]
        checks_ledger = ledger["expense_count"] > 0
        if query["type"] == "balance":
            amount = balance_inr(ledger, query["person"])
            answer_text = f"{query['person']} ka balance {amount:.2f} INR hai."
            contains = [fmt(amount)]
        elif query["type"] == "settle":
            transfers = settle_ledger(ledger)
            if transfers:
                parts = [f"{t['from']} -> {t['to']} {t['amount_inr']:.2f} INR" for t in transfers]
                answer_text = "Settlement: " + "; ".join(parts)
                contains = [fmt(t["amount_inr"]) for t in transfers]
            else:
                answer_text = "Sab settled hai, koi transfer nahi chahiye."
                contains = ["settled"]
        elif query["type"] == "convert":
            value = execute_tool(
                "convert_currency",
                {
                    "amount": query["amount"],
                    "from_currency": query["from"],
                    "to_currency": query["to"],
                },
            )["value"]
            answer_text = f"{query['amount']} {query['from']} = {value:.2f} {query['to']}."
            contains = [fmt(value)]
        else:
            raise ValueError(f"unknown query type {query['type']}")

        tasks.append(
            {
                "id": scenario["id"],
                "task": scenario["task"],
                "expected_tools": scenario["expected_tools"],
                "checks_ledger": checks_ledger,
                "expected_balances": public_balances(ledger) if checks_ledger else {},
                "expected_settlements": settle_ledger(ledger) if query["type"] == "settle" else [],
                "answer_contains": contains,
                "answer_text": answer_text,
                "actions": [
                    {"tool_name": a["tool_name"], "parameters": a["parameters"]} for a in scenario["actions"]
                ],
            }
        )
    return tasks


def main() -> None:
    tasks = build()
    output = ROOT / "data" / "tasks.jsonl"
    with output.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task, ensure_ascii=False) + "\n")

    failures = []
    for task in tasks:
        episode = run_oracle(task)
        scores = score_episode(episode, task)
        if scores["success"] != 1.0 or scores["ter"] != 0.0:
            failures.append((task["id"], scores))

    print(f"wrote {len(tasks)} tasks -> {output}")
    if failures:
        for task_id, scores in failures:
            print(f"FAIL {task_id}: {scores}")
        raise SystemExit(1)
    print("oracle check: 20/20 success, 0 tool errors")


if __name__ == "__main__":
    main()
