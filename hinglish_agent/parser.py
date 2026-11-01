"""Parse the model's JSON tool-call output."""

from __future__ import annotations

import json
import re


def _balanced_slice(text: str, start: int) -> str | None:
    opener = text[start]
    closer = {"[": "]", "{": "}"}[opener]
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _extract_json(text: str):
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for index, char in enumerate(text):
        if char not in "[{":
            continue
        chunk = _balanced_slice(text, index)
        if not chunk:
            continue
        try:
            return json.loads(chunk)
        except json.JSONDecodeError:
            continue
    return None


def parse_calls(text: str) -> list[dict] | None:
    """Return [{"tool_name": str, "parameters": dict}] or None if nothing parses."""
    if not text or not isinstance(text, str):
        return None
    payload = _extract_json(text)
    if payload is None:
        return None
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        return None

    calls = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        name = item.get("tool_name") or item.get("name") or item.get("tool")
        parameters = item.get("parameters") or item.get("arguments") or {}
        if name and isinstance(parameters, dict):
            calls.append({"tool_name": str(name), "parameters": parameters})
    return calls or None


def format_call(call: dict) -> str:
    return json.dumps(
        [{"tool_name": call.get("tool_name"), "parameters": call.get("parameters", {})}],
        ensure_ascii=False,
    )
