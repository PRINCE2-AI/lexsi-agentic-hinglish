"""Deterministic expense-splitting ledger.

All money is stored as integer paise (INR cents) so splits are exact and the
same task always produces the same expected answer.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

PEOPLE = ("Rahul", "Priya", "Amit", "Neha", "Sara", "Vikram")

# Fixed, offline rates: 1 USD = 83 INR, 1 EUR = 90 INR.
CURRENCY_CENTS = {"INR": 100, "USD": 8300, "EUR": 9000}


def empty_ledger() -> dict:
    return {"balances_cents": {}, "expense_count": 0}


def to_cents(amount, currency: str) -> int:
    if currency not in CURRENCY_CENTS:
        raise ValueError(f"unsupported currency: {currency}; use INR, USD or EUR")
    try:
        value = Decimal(str(amount))
    except Exception as exc:  # noqa: BLE001 - surface a clean tool error
        raise ValueError(f"amount must be a number, got {amount!r}") from exc
    if value <= 0:
        raise ValueError("amount must be positive")
    if value > Decimal("10000000"):
        raise ValueError("amount is unreasonably large")
    return int((value * CURRENCY_CENTS[currency]).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def split_cents(total: int, n: int) -> list[int]:
    """Split `total` paise across `n` people; leftover paise go to the first people."""
    base, remainder = divmod(total, n)
    return [base + (1 if i < remainder else 0) for i in range(n)]


def clean_ledger(ledger) -> dict:
    if ledger is None:
        return empty_ledger()
    if not isinstance(ledger, dict):
        raise ValueError("ledger must be a JSON object like {\"balances_cents\": {}, \"expense_count\": 0}")
    raw = ledger.get("balances_cents") or {}
    if not isinstance(raw, dict):
        raise ValueError("ledger.balances_cents must be an object")
    balances = {}
    for person, cents in raw.items():
        if person not in PEOPLE:
            raise ValueError(f"unknown person in ledger: {person}")
        try:
            balances[person] = int(cents)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"balance for {person} must be an integer number of paise") from exc
    count = int(ledger.get("expense_count") or 0)
    return {"balances_cents": balances, "expense_count": count}


def add_expense_to_ledger(ledger, payer: str, amount, currency: str, participants, description: str = ""):
    state = clean_ledger(ledger)
    if payer not in PEOPLE:
        raise ValueError(f"unknown payer {payer!r}; use one of {', '.join(PEOPLE)}")
    if not isinstance(participants, (list, tuple)) or not participants:
        raise ValueError("participants must be a non-empty list")
    parts = [str(p) for p in participants]
    if len(set(parts)) != len(parts):
        raise ValueError("participants contains a duplicate")
    for person in parts:
        if person not in PEOPLE:
            raise ValueError(f"unknown participant {person!r}; use one of {', '.join(PEOPLE)}")

    total = to_cents(amount, currency)
    shares = split_cents(total, len(parts))
    balances = dict(state["balances_cents"])
    balances[payer] = balances.get(payer, 0) + total
    for person, share in zip(parts, shares):
        balances[person] = balances.get(person, 0) - share

    updated = {"balances_cents": balances, "expense_count": state["expense_count"] + 1}
    info = {
        "total_inr": round(total / 100, 2),
        "share_inr": [round(s / 100, 2) for s in shares],
        "description": description,
    }
    return updated, info


def convert_amount(amount, from_currency: str, to_currency: str) -> float:
    if from_currency not in CURRENCY_CENTS or to_currency not in CURRENCY_CENTS:
        raise ValueError("currencies must be INR, USD or EUR")
    cents = to_cents(amount, from_currency)
    if from_currency == to_currency:
        return round(cents / 100, 2)
    inr = Decimal(cents) / 100
    value = inr / Decimal(CURRENCY_CENTS[to_currency]) * 100
    return float(value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def balance_inr(ledger, person: str) -> float:
    state = clean_ledger(ledger)
    if person not in PEOPLE:
        raise ValueError(f"unknown person {person!r}")
    return round(state["balances_cents"].get(person, 0) / 100, 2)


def public_balances(ledger) -> dict[str, float]:
    state = clean_ledger(ledger)
    out = {}
    for person, cents in state["balances_cents"].items():
        if cents:
            out[person] = round(cents / 100, 2)
    return dict(sorted(out.items()))


def settle_ledger(ledger) -> list[dict]:
    """Greedy minimal-ish settlement. Returns transfers in INR."""
    state = clean_ledger(ledger)
    creditors = sorted(
        ([name, cents] for name, cents in state["balances_cents"].items() if cents > 0),
        key=lambda pair: (-pair[1], pair[0]),
    )
    debtors = sorted(
        ([name, -cents] for name, cents in state["balances_cents"].items() if cents < 0),
        key=lambda pair: (-pair[1], pair[0]),
    )
    transfers = []
    i = j = 0
    while i < len(debtors) and j < len(creditors):
        amount = min(debtors[i][1], creditors[j][1])
        if amount > 0:
            transfers.append(
                {
                    "from": debtors[i][0],
                    "to": creditors[j][0],
                    "amount_inr": round(amount / 100, 2),
                }
            )
        debtors[i][1] -= amount
        creditors[j][1] -= amount
        if debtors[i][1] == 0:
            i += 1
        if creditors[j][1] == 0:
            j += 1
    return transfers


def ledgers_match(ledger, expected: dict) -> bool:
    got = public_balances(ledger)
    keys = set(got) | set(expected)
    return all(round(float(got.get(k, 0.0)), 2) == round(float(expected.get(k, 0.0)), 2) for k in keys)


def settlements_match(got: list[dict], expected: list[dict]) -> bool:
    def key(item):
        return (str(item["from"]), str(item["to"]), round(float(item["amount_inr"]), 2))

    return sorted(map(key, got)) == sorted(map(key, expected))
