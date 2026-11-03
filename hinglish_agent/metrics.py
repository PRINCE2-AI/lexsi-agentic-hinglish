"""Aggregate episode scores into the before/after table."""

from __future__ import annotations

METRIC_KEYS = (
    "success",
    "tac",
    "arr",
    "ter",
    "rad",
    "overcall",
    "lcf",
    "answer_ok",
    "strict",
    "strict_rate",
)
COUNT_KEYS = ("calls", "errors", "steps", "parse_failures")


def summarise(scores: list[dict]) -> dict:
    if not scores:
        return {
            "n": 0,
            **{key: 0.0 for key in METRIC_KEYS},
            **{key: 0.0 for key in COUNT_KEYS},
        }
    summary = {"n": len(scores)}
    for key in METRIC_KEYS:
        summary[key] = round(sum(float(row.get(key, 0.0)) for row in scores) / len(scores), 4)
    for key in COUNT_KEYS:
        summary[key] = round(sum(float(row.get(key, 0.0)) for row in scores) / len(scores), 2)
    return summary


def markdown_table(before: dict, after: dict | None = None) -> str:
    labels = {
        "success": "Task success",
        "tac": "Tool accuracy (tac)",
        "arr": "Argument reliability (arr)",
        "ter": "Tool error rate (ter)",
        "rad": "Redundant actions (rad)",
        "overcall": "Unnecessary calls (overcall)",
        "lcf": "Loop collapse (lcf)",
        "answer_ok": "Answer contains amount",
        "strict": "Strict JSON (all turns)",
        "strict_rate": "Strict JSON turns",
        "calls": "Avg tool calls / task",
        "errors": "Tool errors / task",
        "steps": "Avg steps / task",
        "parse_failures": "Parse failures / task",
    }
    header = "| Metric | Baseline | Tuned |\n|---|---:|---:|"
    lines = [header]
    for key in METRIC_KEYS:
        left = f"{before.get(key, 0.0):.2%}"
        right = f"{after.get(key, 0.0):.2%}" if after else "-"
        lines.append(f"| {labels[key]} | {left} | {right} |")
    for key in COUNT_KEYS:
        left = f"{before.get(key, 0.0):.2f}"
        right = f"{after.get(key, 0.0):.2f}" if after else "-"
        lines.append(f"| {labels[key]} | {left} | {right} |")
    return "\n".join(lines)
