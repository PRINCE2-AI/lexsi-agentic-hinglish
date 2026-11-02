"""Graded reward for tool-calling rollouts.

The binary reward used in the first GRPO attempt (`1.0` or `0.0`) had zero
variance: every completion in a group scored the same, so GRPO had no advantage
to learn from (`frac_reward_zero_std=1`). This module grades a completion on the
parts the task actually needs, using the oracle actions as the reference:

    format       1.0  bare JSON array of tool calls
    name         3.0  right tool picked
    param_name   1.5  right parameter keys (Jaccard against the oracle keys)
    param_value  1.5  right values for the keys that overlap
    answer       2.0  finish.answer contains every expected number

Matching is order-agnostic and normalises numbers, lists and case, so a correct
call that swaps two participants still scores full marks.
"""

from __future__ import annotations

import json

from .parser import parse_calls
from .runner import answer_contains, is_strict_json

WEIGHTS = {"format": 1.0, "name": 3.0, "param_name": 1.5, "param_value": 1.5, "answer": 2.0}
MAX_REWARD = sum(WEIGHTS.values())


def _norm(value):
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, (int, float)):
        return ("num", round(float(value), 6))
    if isinstance(value, str):
        try:
            return ("num", round(float(value.strip()), 6))
        except ValueError:
            return ("str", value.strip().lower())
    if isinstance(value, (list, tuple)):
        return ("seq", tuple(sorted(_norm(item) for item in value)))
    return ("str", str(value))


def _params(call: dict) -> dict:
    params = call.get("parameters")
    if not isinstance(params, dict):
        return {}
    return {key: value for key, value in params.items() if key != "ledger"}


def _extract_calls(text: str) -> list[dict]:
    """Every tool call in a completion, including multi-turn concatenations.

    `parse_calls` returns only the first JSON array, which is fine for one turn
    but would grade only turn 1 of a whole trajectory.
    """
    decoder = json.JSONDecoder()
    calls: list[dict] = []
    index = 0
    while True:
        start = text.find("[", index)
        if start == -1:
            break
        try:
            payload, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            index = start + 1
            continue
        items = payload if isinstance(payload, list) else [payload]
        calls.extend(
            item for item in items
            if isinstance(item, dict) and item.get("tool_name")
            and isinstance(item.get("parameters"), dict)
        )
        index = start + end
    return calls or (parse_calls(text) or [])


def score_call(predicted: dict, oracle: dict) -> tuple[float, float, float]:
    """(name, param_name, param_value) for one predicted call against one action."""
    if not isinstance(predicted, dict):
        return (0.0, 0.0, 0.0)
    if str(predicted.get("tool_name", "")).strip().lower() != str(oracle["tool_name"]).lower():
        return (0.0, 0.0, 0.0)

    want = {key: value for key, value in oracle["parameters"].items() if key != "ledger"}
    got = _params(predicted)
    union = set(want) | set(got)
    names = len(set(want) & set(got)) / len(union) if union else 1.0
    shared = set(want) & set(got)
    values = (
        sum(1 for key in shared if _norm(got[key]) == _norm(want[key])) / len(want)
        if want
        else 1.0
    )
    return (1.0, names, values)


def grade_calls(predicted: list[dict], oracle_actions: list[dict]) -> dict:
    """Average best-match score over the oracle actions (0..1 per component)."""
    if not oracle_actions:
        return {"name": 0.0, "param_name": 0.0, "param_value": 0.0}
    used: set[int] = set()
    totals = [0.0, 0.0, 0.0]
    for oracle in oracle_actions:
        best = (0.0, 0.0, 0.0)
        best_index = None
        for index, call in enumerate(predicted):
            if index in used:
                continue
            scored = score_call(call, oracle)
            if sum(scored) > sum(best):
                best, best_index = scored, index
        totals = [total + part for total, part in zip(totals, best)]
        if best_index is not None:
            used.add(best_index)
    n = len(oracle_actions)
    return {
        "name": totals[0] / n,
        "param_name": totals[1] / n,
        "param_value": totals[2] / n,
    }


def _answer(calls: list[dict]) -> str:
    for call in calls:
        if str(call.get("tool_name", "")).strip().lower() == "finish":
            return str(_params(call).get("answer", ""))
    return ""


def _actions(value) -> list[dict]:
    if isinstance(value, str):
        return json.loads(value)
    return list(value or [])


def _needles(value) -> list:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return [value]
    return list(value)


def grade_text(text: str, oracle_actions, answer_needles=None) -> dict:
    """Grade one completion (a single turn or a whole trajectory)."""
    calls = _extract_calls(text)
    parts = grade_calls(calls, _actions(oracle_actions))
    answer = _answer(calls)
    needs = _needles(answer_needles)
    if is_strict_json(text):
        parts["format"] = 1.0
    else:
        parts["format"] = 0.5 if calls else 0.0
    parts["answer"] = 1.0 if (needs and answer_contains(answer, needs)) else 0.0
    parts["total"] = round(sum(WEIGHTS[key] * parts[key] for key in WEIGHTS), 4)
    return parts


def grade_episode(episode: dict, task: dict) -> dict:
    """Greedy-eval diagnostic: grade every assistant turn, then the final answer."""
    turns = [message.get("content", "") for message in episode.get("messages", [])
             if message.get("role") == "assistant"]
    scored = [grade_text(turn, task["actions"], task.get("answer_contains")) for turn in turns]
    best = max((row["total"] for row in scored), default=0.0)
    return {
        "turns": len(scored),
        "best_turn": round(best, 4),
        "mean_turn": round(sum(row["total"] for row in scored) / len(scored), 4) if scored else 0.0,
        "final_answer": 1.0
        if answer_contains(episode.get("final_response", ""), task.get("answer_contains") or [])
        else 0.0,
    }
