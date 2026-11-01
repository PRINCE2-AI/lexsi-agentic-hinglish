"""Stateless tools for the Hinglish expense-splitting agent.

Every ledger tool takes and returns the ledger object itself, so parallel GRPO
rollouts can never leak state into each other.
"""

from __future__ import annotations

from .ledger import (
    add_expense_to_ledger,
    balance_inr,
    clean_ledger,
    convert_amount,
    empty_ledger,
    settle_ledger,
)


def add_expense(
    ledger: dict,
    payer: str,
    amount: float,
    currency: str,
    participants: list[str],
    description: str = "",
) -> dict:
    """Add one shared expense and return the updated ledger.

    Args:
        ledger: Current ledger object. For the first call use {"balances_cents": {}, "expense_count": 0}.
        payer: Person who paid. One of Rahul, Priya, Amit, Neha, Sara, Vikram.
        amount: Amount paid, in `currency`.
        currency: INR, USD or EUR.
        participants: People who share this expense.
        description: Short label such as "pizza".

    Returns:
        {"ok": true, "ledger": {...}, "total_inr": float, "share_inr": [...]}.
    """
    updated, info = add_expense_to_ledger(
        ledger=ledger,
        payer=payer,
        amount=amount,
        currency=currency,
        participants=participants,
        description=description,
    )
    return {"ok": True, "ledger": updated, **info}


def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
    """Convert money between INR, USD and EUR using fixed offline rates (1 USD = 83 INR, 1 EUR = 90 INR).

    Args:
        amount: Amount to convert.
        from_currency: INR, USD or EUR.
        to_currency: INR, USD or EUR.

    Returns:
        {"ok": true, "value": float, "unit": "INR"}.
    """
    value = convert_amount(amount, from_currency, to_currency)
    return {"ok": True, "value": value, "unit": to_currency}


def get_balance(ledger: dict, person: str) -> dict:
    """Return one person's net balance in INR. Positive means the person should receive money.

    Args:
        ledger: Current ledger object.
        person: One of Rahul, Priya, Amit, Neha, Sara, Vikram.

    Returns:
        {"ok": true, "person": str, "balance_inr": float}.
    """
    clean_ledger(ledger)
    value = balance_inr(ledger, person)
    return {"ok": True, "person": person, "balance_inr": value}


def settle_up(ledger: dict) -> dict:
    """Compute the transfers that settle all balances.

    Args:
        ledger: Current ledger object.

    Returns:
        {"ok": true, "transfers": [{"from": str, "to": str, "amount_inr": float}]}.
    """
    clean_ledger(ledger)
    transfers = settle_ledger(ledger)
    return {"ok": True, "transfers": transfers}


def finish(answer: str) -> dict:
    """Finish the task and return the final answer to the user.

    Args:
        answer: Short answer that includes the rupee amounts, e.g. "Rahul ko 900 INR milne hain."

    Returns:
        {"ok": true, "done": true, "answer": str}.
    """
    if not str(answer).strip():
        raise ValueError("answer must not be empty")
    return {"ok": True, "done": True, "answer": str(answer).strip()}


TOOL_FUNCTIONS = [add_expense, convert_currency, get_balance, settle_up, finish]
TOOL_NAMES = tuple(fn.__name__ for fn in TOOL_FUNCTIONS)

_DISPATCH = {fn.__name__: fn for fn in TOOL_FUNCTIONS}


def execute_tool(name: str, arguments: dict) -> dict:
    """Run one tool call and never raise: failures become {"ok": false, "error": ...}."""
    fn = _DISPATCH.get(str(name))
    if fn is None:
        return {"ok": False, "error": f"unknown tool {name!r}; available: {', '.join(TOOL_NAMES)}"}
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        return {"ok": False, "error": "parameters must be a JSON object"}
    try:
        result = fn(**arguments)
    except TypeError as exc:
        return {"ok": False, "error": f"bad parameters: {exc}"}
    except Exception as exc:  # noqa: BLE001 - tools must degrade to observations
        return {"ok": False, "error": str(exc)}
    return result if isinstance(result, dict) else {"ok": True, "result": result}


LEDGER_SCHEMA = {
    "type": "object",
    "description": "Ledger returned by the previous ledger tool call.",
    "properties": {
        "balances_cents": {"type": "object", "additionalProperties": {"type": "integer"}},
        "expense_count": {"type": "integer"},
    },
}

TOOL_SCHEMAS = [
    {
        "name": "add_expense",
        "description": "Add one shared expense and return the updated ledger.",
        "parameters": {
            "type": "object",
            "properties": {
                "ledger": LEDGER_SCHEMA,
                "payer": {"type": "string", "enum": ["Rahul", "Priya", "Amit", "Neha", "Sara", "Vikram"]},
                "amount": {"type": "number"},
                "currency": {"type": "string", "enum": ["INR", "USD", "EUR"]},
                "participants": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["Rahul", "Priya", "Amit", "Neha", "Sara", "Vikram"]},
                },
                "description": {"type": "string"},
            },
            "required": ["ledger", "payer", "amount", "currency", "participants"],
        },
    },
    {
        "name": "convert_currency",
        "description": "Convert between INR, USD and EUR. Fixed rates: 1 USD = 83 INR, 1 EUR = 90 INR.",
        "parameters": {
            "type": "object",
            "properties": {
                "amount": {"type": "number"},
                "from_currency": {"type": "string", "enum": ["INR", "USD", "EUR"]},
                "to_currency": {"type": "string", "enum": ["INR", "USD", "EUR"]},
            },
            "required": ["amount", "from_currency", "to_currency"],
        },
    },
    {
        "name": "get_balance",
        "description": "Return one person's net balance in INR. Positive means they should receive money.",
        "parameters": {
            "type": "object",
            "properties": {"ledger": LEDGER_SCHEMA, "person": {"type": "string"}},
            "required": ["ledger", "person"],
        },
    },
    {
        "name": "settle_up",
        "description": "Compute the transfers that settle all balances.",
        "parameters": {
            "type": "object",
            "properties": {"ledger": LEDGER_SCHEMA},
            "required": ["ledger"],
        },
    },
    {
        "name": "finish",
        "description": "Finish the task and return the final answer with rupee amounts.",
        "parameters": {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        },
    },
]

INITIAL_LEDGER = empty_ledger()
