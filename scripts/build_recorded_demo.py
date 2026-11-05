"""Freeze a handful of real episodes so the Space can show a full trace without a GPU.

ZeroGPU gives each visitor a few GPU-minutes per day, so a judge who runs out of quota
currently sees nothing. These episodes come from the frozen evaluations, so what the Space
replays is exactly what the scoring code produced:

    python scripts/build_recorded_demo.py     # writes space/recorded.json

The app re-runs its own `_verdict` and `_trace_markdown` over the stored episode, so the
recorded panel cannot drift from the live one.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hinglish_agent.ledger import empty_ledger  # noqa: E402

OUT = ROOT / "space" / "recorded.json"

TASK_FILES = {
    "hard": ROOT / "data" / "tasks_hard_eval.jsonl",
    "safety": ROOT / "data" / "tasks_safety_eval.jsonl",
}


def load_tasks(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8") as handle:
        return {row["id"]: row for row in (json.loads(line) for line in handle)}


def load_run(run: str) -> list[dict]:
    path = ROOT / "runs" / run / "trajectories.jsonl"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def rebuild_episode(row: dict) -> dict:
    """Turn a stored trajectory back into the dict `run_episode` returns."""
    scores = row["metadata"].get("scores") or {}
    ledger = empty_ledger()
    for step in row["steps"]:
        result = step.get("result") or {}
        if isinstance(result.get("ledger"), dict):
            ledger = result["ledger"]
    return {
        "steps": row["steps"],
        "final_response": row.get("final_response") or "",
        "ledger": ledger,
        "done": bool(row.get("final_response")),
        "injections_fired": int(scores.get("injections_fired") or 0),
        "messages": row["metadata"].get("conversation") or [],
    }


def pick(rows: list[dict], where) -> dict | None:
    for row in rows:
        if where(row["metadata"].get("scores") or {}):
            return row
    return None


def main() -> None:
    hard = load_tasks(TASK_FILES["hard"])
    safety = load_tasks(TASK_FILES["safety"])

    cases = [
        (
            "Shipped adapter — correct split",
            "sft4_hard30",
            hard,
            lambda s: s.get("success") == 1 and s.get("calls", 99) <= 5,
        ),
        (
            "Base model — duplicate expenses",
            "base_hard30",
            hard,
            lambda s: not s.get("success") and (s.get("rad") or 0) > 0,
        ),
        (
            "Shipped adapter — injection resisted",
            "safety_sft4",
            safety,
            lambda s: s.get("injections_fired") == 1
            and not s.get("attack_ledger")
            and not s.get("attack_answer"),
        ),
        (
            "Base model — injection succeeded",
            "safety_base",
            safety,
            lambda s: bool(s.get("attack_ledger")),
        ),
        (
            "Shipped adapter — asks for missing names",
            "safety_sft4",
            safety,
            lambda s: bool(s.get("asked")),
        ),
        (
            "Base model — invents a split",
            "safety_base",
            safety,
            lambda s: bool(s.get("invented_expense")),
        ),
    ]

    recorded = []
    for label, run, tasks, where in cases:
        row = pick(load_run(run), where)
        if row is None:
            print(f"skip (no episode): {label} from {run}")
            continue
        task_id = row["metadata"]["task_id"]
        task = tasks.get(task_id)
        if task is None:
            print(f"skip (task {task_id} missing from the frozen split): {label}")
            continue
        recorded.append({"label": label, "task": task, "episode": rebuild_episode(row)})
        print(f"{label}: {run}/{task_id}")

    if not recorded:
        raise SystemExit("no episodes recorded; run the evaluations first")

    OUT.write_text(json.dumps(recorded, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB, {len(recorded)} cases)")


if __name__ == "__main__":
    main()
