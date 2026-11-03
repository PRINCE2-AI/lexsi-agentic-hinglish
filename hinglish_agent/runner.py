"""Run one agent episode and score it with deterministic, programmatic metrics."""

from __future__ import annotations

import json
import re

from .ledger import clean_ledger, empty_ledger, ledgers_match, settle_ledger, settlements_match
from .parser import format_call, parse_calls
from .prompts import CORRECTION_PROMPT, build_messages
from .tools import execute_tool

LEDGER_TOOLS = {"add_expense", "get_balance", "settle_up"}


def is_strict_json(text: str) -> bool:
    """True only for a bare JSON array of well-formed tool calls, no fences, no prose."""
    if not isinstance(text, str):
        return False
    text = text.strip()
    if not text or text[0] not in "[{" or text[-1] not in "]}":
        return False
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return False
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list) or not payload:
        return False
    return all(
        isinstance(item, dict)
        and item.get("tool_name")
        and isinstance(item.get("parameters"), dict)
        for item in payload
    )


def run_episode(call_model, task: dict, max_steps: int = 8, hardened: bool = False) -> dict:
    """Drive a model through the tool loop. `call_model(messages) -> str`.

    A task with an `injection` block appends that payload to the named tool's result, so the
    model sees attacker text arriving as tool output rather than as user input.
    """
    messages = build_messages(task["task"], hardened=hardened)
    injection = task.get("injection") or {}
    injections_fired = 0
    steps: list[dict] = []
    ledger = empty_ledger()
    done = False
    final_response = ""

    for _ in range(max_steps):
        raw = call_model(messages)
        calls = parse_calls(raw)

        if not calls:
            steps.append(
                {
                    "name": None,
                    "arguments": {},
                    "result": {"ok": False, "error": "unparseable model output"},
                    "raw": raw,
                }
            )
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": CORRECTION_PROMPT})
            continue

        observations = []
        for call in calls[:3]:
            result = execute_tool(call["tool_name"], call["parameters"])
            steps.append({"name": call["tool_name"], "arguments": call["parameters"], "result": result})
            shown = result
            if injection and call["tool_name"] == injection.get("after_tool"):
                shown = dict(result, note=injection["payload"])
                injections_fired += 1
            observations.append({"tool_name": call["tool_name"], "result": shown})
            if isinstance(result.get("ledger"), dict):
                ledger = result["ledger"]
            if result.get("done"):
                done = True
                final_response = str(result.get("answer", ""))
                break

        messages.append({"role": "assistant", "content": raw})
        if done:
            break
        messages.append(
            {
                "role": "user",
                "content": json.dumps({"tool_results": observations}, ensure_ascii=False),
            }
        )

    return {
        "task_id": task.get("id"),
        "task": task["task"],
        "messages": messages,
        "steps": steps,
        "final_response": final_response,
        "done": done,
        "ledger": clean_ledger(ledger),
        "injections_fired": injections_fired,
    }


def run_oracle(task: dict) -> dict:
    """Execute the ground-truth actions. Used to verify the dataset itself."""
    steps = []
    messages = build_messages(task["task"])
    ledger = empty_ledger()

    for action in task.get("actions", []):
        arguments = dict(action["parameters"])
        if action["tool_name"] in LEDGER_TOOLS:
            arguments["ledger"] = ledger
        result = execute_tool(action["tool_name"], arguments)
        if not result.get("ok"):
            raise RuntimeError(f"oracle failed on {task['id']}: {result}")
        if isinstance(result.get("ledger"), dict):
            ledger = result["ledger"]
        steps.append({"name": action["tool_name"], "arguments": arguments, "result": result})
        messages.append(
            {
                "role": "assistant",
                "content": format_call({"tool_name": action["tool_name"], "parameters": arguments}),
            }
        )
        messages.append(
            {
                "role": "user",
                "content": json.dumps(
                    {"tool_results": [{"tool_name": action["tool_name"], "result": result}]},
                    ensure_ascii=False,
                ),
            }
        )

    answer = task["answer_text"]
    result = execute_tool("finish", {"answer": answer})
    steps.append({"name": "finish", "arguments": {"answer": answer}, "result": result})
    messages.append(
        {
            "role": "assistant",
            "content": format_call({"tool_name": "finish", "parameters": {"answer": answer}}),
        }
    )

    return {
        "task_id": task.get("id"),
        "task": task["task"],
        "messages": messages,
        "steps": steps,
        "final_response": answer,
        "done": True,
        "ledger": clean_ledger(ledger),
    }


def answer_contains(answer: str, needles) -> bool:
    text = str(answer or "").replace(",", "")
    numbers = [float(match) for match in re.findall(r"-?\d+(?:\.\d+)?", text)]
    for needle in needles or []:
        try:
            value = float(needle)
        except (TypeError, ValueError):
            if str(needle).lower() not in text.lower():
                return False
            continue
        if not any(abs(found - value) < 0.005 for found in numbers):
            return False
    return True


def score_episode(episode: dict, task: dict) -> dict:
    calls = [step for step in episode["steps"] if step.get("name")]
    errors = [step for step in calls if not step["result"].get("ok")]
    n_calls = len(calls)

    ter = len(errors) / n_calls if n_calls else 1.0
    arr = 1.0 - ter if n_calls else 0.0

    called_names = {step["name"] for step in calls}
    expected_tools = task.get("expected_tools") or []
    tac = 1.0 if expected_tools and all(name in called_names for name in expected_tools) else 0.0

    keys = [json.dumps([step["name"], step["arguments"]], sort_keys=True, default=str) for step in calls]
    rad = (len(keys) - len(set(keys))) / len(keys) if keys else 0.0
    lcf = 0.0 if episode.get("done") else 1.0

    oracle_calls = len(task.get("actions") or []) + 1
    overcall = max(0, n_calls - oracle_calls) / oracle_calls if oracle_calls else 0.0
    parse_failures = sum(1 for step in episode["steps"] if not step.get("name"))

    expected_balances = task.get("expected_balances") or {}
    expected_settlements = task.get("expected_settlements") or []
    ledger_ok = ledgers_match(episode["ledger"], expected_balances)
    settlement_ok = (
        settlements_match(settle_ledger(episode["ledger"]), expected_settlements)
        if expected_settlements
        else True
    )
    answer_ok = answer_contains(episode.get("final_response", ""), task.get("answer_contains") or [])

    assistant_turns = [m for m in episode.get("messages", []) if m.get("role") == "assistant"]
    strict_turns = sum(1 for m in assistant_turns if is_strict_json(m.get("content", "")))
    strict = 1.0 if assistant_turns and strict_turns == len(assistant_turns) else 0.0
    strict_rate = strict_turns / len(assistant_turns) if assistant_turns else 0.0

    if task.get("checks_ledger"):
        success = bool(ledger_ok and settlement_ok)
    else:
        success = bool(answer_ok)
    if task.get("require_answer"):
        success = bool(success and answer_ok)

    return {
        "task_id": task.get("id"),
        "success": float(success),
        "tac": tac,
        "arr": round(arr, 4),
        "ter": round(ter, 4),
        "rad": round(rad, 4),
        "overcall": round(overcall, 4),
        "lcf": lcf,
        "answer_ok": float(answer_ok),
        "ledger_ok": float(ledger_ok),
        "settlement_ok": float(settlement_ok),
        "strict": strict,
        "strict_rate": round(strict_rate, 4),
        "calls": n_calls,
        "errors": len(errors),
        "parse_failures": parse_failures,
        "steps": len(episode["steps"]),
    }


def to_trajectory(episode: dict, scores: dict) -> dict:
    """AgentTune-compatible trajectory row for AuditKIT."""
    return {
        "task": episode["task"],
        "steps": [
            {
                "id": index,
                "name": step.get("name"),
                "arguments": step.get("arguments", {}),
                "result": step.get("result", {}),
            }
            for index, step in enumerate(episode["steps"])
        ],
        "final_response": episode.get("final_response", ""),
        "metadata": {
            "task_id": episode.get("task_id"),
            "conversation": episode.get("messages", []),
            "scores": scores,
        },
    }
