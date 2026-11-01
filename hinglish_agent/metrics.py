"""Aggregate episode scores into the before/after table."""

from __future__ import annotations

METRIC_KEYS = ("success", "tac", "arr", "ter", "rad", "lcf", "answer_ok")


def summarise(scores: list[dict]) -> dict:
    if not scores:
        return {"n": 0, **{key: 0.0 for key in METRIC_KEYS}}
    summary = {"n": len(scores)}
    for key in METRIC_KEYS:
        summary[key] = round(sum(float(row.get(key, 0.0)) for row in scores) / len(scores), 4)
    return summary


def markdown_table(before: dict, after: dict | None = None) -> str:
    labels = {
        "success": "Task success",
        "tac": "Tool accuracy (tac)",
        "arr": "Argument reliability (arr)",
        "ter": "Tool error rate (ter)",
        "rad": "Redundant actions (rad)",
        "lcf": "Loop collapse (lcf)",
        "answer_ok": "Answer contains amount",
    }
    header = "| Metric | Baseline | Tuned |\n|---|---:|---:|"
    lines = [header]
    for key in METRIC_KEYS:
        left = f"{before.get(key, 0.0):.2%}"
        right = f"{after.get(key, 0.0):.2%}" if after else "-"
        lines.append(f"| {labels[key]} | {left} | {right} |")
    return "\n".join(lines)
