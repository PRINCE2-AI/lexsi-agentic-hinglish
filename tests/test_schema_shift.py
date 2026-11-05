"""Schema-shift checks. Run: python tests/test_schema_shift.py

The shipped adapter learned the canonical tool vocabulary. The shift eval renames
every tool and parameter at inference time, so the harness must (a) render a
prompt that never mentions the canonical names, (b) decode aliased calls back to
the real tools, and (c) reject canonical calls exactly like an unknown tool.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hinglish_agent.prompts import CANONICAL_EXAMPLE, SCHEMA_SHIFT_PROMPT, SYSTEM_PROMPT  # noqa: E402
from hinglish_agent.runner import run_episode, run_oracle, score_episode  # noqa: E402
from hinglish_agent.schema_shift import (  # noqa: E402
    TOOL_ALIASES,
    aliased_schemas,
    decode_call,
    encode_call,
)

CANONICAL_NAMES = tuple(TOOL_ALIASES)


def load_tasks() -> list[dict]:
    with (ROOT / "data" / "tasks_hard_eval.jsonl").open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_prompt_has_no_canonical_names() -> None:
    for name in CANONICAL_NAMES:
        assert name not in SCHEMA_SHIFT_PROMPT, f"{name} leaked into the shifted prompt"
    assert "record_split" in SCHEMA_SHIFT_PROMPT
    assert "add_expense" in SYSTEM_PROMPT


def test_schemas_are_renamed_not_changed() -> None:
    aliased = {schema["name"]: schema for schema in aliased_schemas()}
    assert set(aliased) == set(TOOL_ALIASES.values())
    add = aliased["record_split"]["parameters"]
    assert set(add["required"]) == {"book", "paid_by", "value", "unit", "shared_by"}
    assert add["properties"]["unit"]["enum"] == ["INR", "USD", "EUR"]


def test_decode_round_trip() -> None:
    call = encode_call("add_expense", {"payer": "Rahul", "amount": 5, "currency": "INR", "participants": ["Rahul"]})
    assert call["tool_name"] == "record_split"
    decoded = decode_call(call["tool_name"], call["parameters"])
    assert decoded == (
        "add_expense",
        {"payer": "Rahul", "amount": 5, "currency": "INR", "participants": ["Rahul"]},
    )


def test_canonical_call_is_rejected_under_shift() -> None:
    assert decode_call("add_expense", {"payer": "Rahul", "amount": 5}) is None
    assert decode_call("record_splitt", {"book": {}}) is None


def test_oracle_survives_the_shift() -> None:
    for task in load_tasks():
        episode = run_oracle(task, aliases=True)
        scores = score_episode(episode, task)
        assert scores["success"] == 1.0, task["id"]
        assert scores["ter"] == 0.0, task["id"]


def _scripted(calls: list[dict]):
    """A model that returns the given calls in order, then repeats the last one."""
    queue = list(calls)

    def call(_messages: list[dict]) -> str:
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        return json.dumps([item])

    return call


def test_aliased_model_succeeds_and_canonical_model_fails() -> None:
    task = next(t for t in load_tasks() if t["family"] == "settle5")
    oracle = run_oracle(task, aliases=True)
    actions = [encode_call(step["name"], step["arguments"]) for step in oracle["steps"]]

    shifted = run_episode(_scripted(actions), task, aliases=True)
    assert score_episode(shifted, task)["success"] == 1.0
    assert shifted["steps"][0]["name"] == "add_expense"
    assert shifted["steps"][0]["raw_name"] == "record_split"

    canonical = run_episode(
        _scripted([{"tool_name": s["name"], "parameters": s["arguments"]} for s in run_oracle(task)["steps"]]),
        task,
        aliases=True,
    )
    scores = score_episode(canonical, task)
    assert scores["success"] == 0.0
    assert scores["errors"] >= 1


if __name__ == "__main__":
    test_prompt_has_no_canonical_names()
    test_schemas_are_renamed_not_changed()
    test_decode_round_trip()
    test_canonical_call_is_rejected_under_shift()
    test_oracle_survives_the_shift()
    test_aliased_model_succeeds_and_canonical_model_fails()
    print("schema shift checks passed")
