"""Shared scenario -> task conversion for every dataset builder.

Both `build_tasks.py` (starter-20) and `build_generated.py` (100/24 splits)
must compute expected balances, settlements and answer text through the same
code path; otherwise the oracle gate tests different logic than the dataset.
"""

from __future__ import annotations

from hinglish_agent.ledger import (
    balance_inr,
    empty_ledger,
    public_balances,
    settle_ledger,
)
from hinglish_agent.tools import execute_tool

LEDGER_TOOLS = {"add_expense", "get_balance", "settle_up"}


def A(name: str, **parameters) -> dict:
    return {"tool_name": name, "parameters": parameters}


def fmt(value: float) -> str:
    return f"{float(value):.2f}".rstrip("0").rstrip(".")


def finalize(scenario: dict) -> dict:
    """Replay the oracle actions and derive every expected field."""
    ledger = empty_ledger()
    for action in scenario["actions"]:
        arguments = dict(action["parameters"])
        if action["tool_name"] in LEDGER_TOOLS:
            arguments["ledger"] = ledger
        result = execute_tool(action["tool_name"], arguments)
        if not result.get("ok"):
            raise RuntimeError(f"oracle failed: {result}")
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
    elif query["type"] == "convert_sum":
        total = 0.0
        parts = []
        for item in query["items"]:
            value = execute_tool(
                "convert_currency",
                {
                    "amount": item["amount"],
                    "from_currency": item["from"],
                    "to_currency": query["to"],
                },
            )["value"]
            total += value
            parts.append(f"{item['amount']} {item['from']} = {value:.2f} {query['to']}")
        total = round(total, 2)
        answer_text = "; ".join(parts) + f". Total {total:.2f} {query['to']}."
        contains = [fmt(total)]
    elif query["type"] == "balance_and_settle":
        amount = balance_inr(ledger, query["person"])
        transfers = settle_ledger(ledger)
        head = f"{query['person']} ka balance {amount:.2f} INR hai."
        if transfers:
            parts = [f"{t['from']} -> {t['to']} {t['amount_inr']:.2f} INR" for t in transfers]
            answer_text = head + " Settlement: " + "; ".join(parts)
            contains = [fmt(amount)] + [fmt(t["amount_inr"]) for t in transfers]
        else:
            answer_text = head + " Sab settled hai, koi transfer nahi chahiye."
            contains = [fmt(amount), "settled"]
    else:
        raise ValueError(query["type"])

    settles = query["type"] in ("settle", "balance_and_settle")

    return {
        "task": scenario["task"],
        "expected_tools": scenario["expected_tools"],
        "checks_ledger": checks_ledger,
        "expected_balances": public_balances(ledger) if checks_ledger else {},
        "expected_settlements": settle_ledger(ledger) if settles else [],
        "answer_contains": contains,
        "answer_text": answer_text,
        "require_answer": bool(scenario.get("require_answer", False)),
        "actions": [
            {"tool_name": a["tool_name"], "parameters": a["parameters"]}
            for a in scenario["actions"]
        ],
    }
