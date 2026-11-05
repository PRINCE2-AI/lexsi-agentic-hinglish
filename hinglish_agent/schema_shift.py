"""Aliased tool schema, Hammer-style: same task, different tool and parameter names.

The shipped adapter learned the canonical vocabulary (`add_expense`, `amount`,
`participants`). Renaming the surface at inference time separates "learned the
task" from "learned the schema string": the ledger semantics, the user text and
the scoring code stay untouched.

Nothing here changes the canonical prompt or the tools; it is opt-in.
"""

from __future__ import annotations

import copy

from .tools import TOOL_SCHEMAS

TOOL_ALIASES = {
    "add_expense": "record_split",
    "convert_currency": "fx_convert",
    "get_balance": "member_balance",
    "settle_up": "settle",
    "finish": "done",
}

PARAM_ALIASES = {
    "add_expense": {
        "ledger": "book",
        "payer": "paid_by",
        "amount": "value",
        "currency": "unit",
        "participants": "shared_by",
        "description": "label",
    },
    "convert_currency": {"amount": "value", "from_currency": "src", "to_currency": "dst"},
    "get_balance": {"ledger": "book", "person": "member"},
    "settle_up": {"ledger": "book"},
    "finish": {"answer": "text"},
}

REVERSE_TOOL = {aliased: canonical for canonical, aliased in TOOL_ALIASES.items()}
REVERSE_PARAM = {
    canonical: {aliased: param for param, aliased in mapping.items()}
    for canonical, mapping in PARAM_ALIASES.items()
}


def aliased_schemas() -> list[dict]:
    """The same schemas with every tool and parameter name renamed."""
    out = copy.deepcopy(TOOL_SCHEMAS)
    for schema in out:
        canonical = schema["name"]
        schema["name"] = TOOL_ALIASES[canonical]
        rename = PARAM_ALIASES[canonical]
        schema["parameters"]["properties"] = {
            rename.get(key, key): value
            for key, value in schema["parameters"]["properties"].items()
        }
        schema["parameters"]["required"] = [
            rename.get(key, key) for key in schema["parameters"]["required"]
        ]
    return out


def encode_call(name: str, parameters: dict) -> dict:
    """Canonical action -> aliased call, for the oracle and for tests."""
    rename = PARAM_ALIASES[name]
    return {
        "tool_name": TOOL_ALIASES[name],
        "parameters": {rename.get(key, key): value for key, value in parameters.items()},
    }


def decode_call(name: str, parameters: dict) -> tuple[str, dict] | None:
    """Aliased call -> canonical (name, parameters), or None if the model used the old schema."""
    canonical = REVERSE_TOOL.get(str(name))
    if canonical is None:
        return None
    rename = REVERSE_PARAM[canonical]
    if not isinstance(parameters, dict):
        return None
    decoded = {}
    for key, value in parameters.items():
        original = rename.get(key, key)
        if original in decoded:
            return None
        decoded[original] = value
    return canonical, decoded


def aliased_names() -> tuple[str, ...]:
    return tuple(TOOL_ALIASES[name] for name in TOOL_ALIASES)
