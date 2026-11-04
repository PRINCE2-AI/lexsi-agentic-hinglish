"""Score a recorded run with AuditKIT's agent layer (Track 3's second repo).

AgentTune trains the policy; this script gives the *same* saved trajectories to
AuditKIT and reports its own tool-call metrics, so the write-up's tool numbers
come from the organiser's evaluator instead of from our scoring code.

    pip install "git+https://github.com/Lexsi-Labs/AuditKIT"
    python scripts/auditkit_eval.py --run runs/sft2_hard30 --tasks data/tasks_hard_eval.jsonl

No judge model and no GPU: the verified outcome is a final-state assertion on the
ledger, and the tool metrics compare the recorded calls against the oracle route.
Tasks without a balance oracle (empty-ledger cases) score `unknown`, which is
reported as such rather than counted as a pass.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hinglish_agent.runner import answer_contains  # noqa: E402


TOOLS = ["add_expense", "convert_currency", "get_balance", "settle_up", "finish"]
# Two route comparisons, because our loop has two kinds of call:
#   ledger route      add_expense / convert_currency / get_balance / settle_up,
#                     compared with AuditKIT's exact-argument matcher
#   full route        the same calls plus finish, whose "answer" argument is free
#                     text, so it is compared on the tool name only
LEDGER_SCORERS = [
    "tool_call_f1",
    "tool_call_validity",
    "redundant_tool_calls",
    "agent_loop_detection",
]
FULL_SCORERS = ["tool_call_f1_name"]


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _slim(arguments: dict) -> dict:
    return {key: value for key, value in arguments.items() if key != "ledger"}


def _call(case_id: str, index: int, name: str, arguments: dict) -> dict:
    return {
        "id": f"{case_id}-{index}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(_slim(arguments), ensure_ascii=False)},
    }


def _messages(task: dict, row: dict) -> list[dict]:
    messages: list[dict] = [{"role": "user", "content": task["task"]}]
    for index, step in enumerate(row.get("steps") or []):
        call = _call(task["id"], index, step["name"], step.get("arguments") or {})
        messages.append({"role": "assistant", "content": "", "tool_calls": [call]})
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call["id"],
                "content": json.dumps(step.get("result"), ensure_ascii=False, default=str)[:4000],
            }
        )
    if row.get("final_response"):
        messages.append({"role": "assistant", "content": row["final_response"]})
    return messages


def _final_ledger(row: dict) -> dict | None:
    """The last ledger the run produced: a result snapshot, else the last argument one."""
    ledger = None
    for step in row.get("steps") or []:
        result = step.get("result") or {}
        if isinstance(result.get("ledger"), dict):
            ledger = result["ledger"]
        elif isinstance((step.get("arguments") or {}).get("ledger"), dict):
            ledger = step["arguments"]["ledger"]
    return ledger


def _answer_oracle(CustomPredicate, needles: list):
    """Answer covers every expected number; used only where no ledger oracle exists."""

    def check(episode, case):
        answer = getattr(episode, "final_answer", "") or ""
        return "success" if answer_contains(answer, needles) else "failure"

    return CustomPredicate(check, version="1", name="answer_contains")


def _reference(task: dict, include_finish: bool) -> list[dict]:
    calls = [
        {"name": action["tool_name"], "arguments": _slim(action["parameters"])}
        for action in task.get("actions") or []
    ]
    if include_finish:
        calls.append({"name": "finish", "arguments": {"answer": task["answer_text"]}})
    return calls


def build_case(AgentCase, FinalStateAssertion, CustomPredicate, task: dict,
               include_finish: bool = False):
    reference = _reference(task, include_finish)
    expected = task.get("expected_balances") or {}
    if expected:
        outcome = FinalStateAssertion("balances", equals=expected)
    elif task.get("answer_contains"):
        outcome = _answer_oracle(CustomPredicate, list(task["answer_contains"]))
    else:
        outcome = None
    return AgentCase(
        id=task["id"],
        task=task["task"],
        category=task.get("family", "generic"),
        allowed_tools=list(TOOLS),
        reference_turns=reference,
        outcome=outcome,
        metadata={"split": task.get("family"), "expected_balances": expected},
    )


def _drop_finish(episode):
    """The same episode without the answer turn, for argument-exact route scoring."""
    import dataclasses

    finish_ids = set()
    for event in episode.events:
        payload = event.payload or {}
        name = (payload.get("function") or {}).get("name") if isinstance(payload, dict) else None
        if event.type == "tool_call" and (name or payload.get("name")) == "finish":
            finish_ids.add(event.call_id)
    events = [
        event for event in episode.events
        if not (event.type == "tool_call" and event.call_id in finish_ids)
        and not (event.type == "tool_result" and event.call_id in finish_ids)
    ]
    counters = dict(episode.counters)
    counters["n_tool_calls"] = sum(1 for event in events if event.type == "tool_call")
    return dataclasses.replace(episode, events=events, counters=counters)


def build_episode(episode_from_openai_messages, OBSERVED, task: dict, row: dict):
    episode = episode_from_openai_messages(
        _messages(task, row), case_id=task["id"], final_answer=row.get("final_response")
    )
    ledger = _final_ledger(row)
    if ledger is not None:
        # Raw cents, not balance_inr(): a model that writes a junk key into its
        # ledger ("Total") must fail the assertion, not crash the evaluator.
        cents = ledger.get("balances_cents") or {}
        episode.final_state = {
            "balances": {person: round(value / 100, 2) for person, value in cents.items()},
            "expense_count": ledger.get("expense_count", 0),
        }
        episode.coverage["final_state"] = OBSERVED
    return episode


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _aggregate(result, own: dict[str, float], cases) -> dict:
    verdicts: dict[str, int] = {}
    families: dict[str, dict] = {}
    buckets: dict[str, list[float]] = {}
    agreed = decided = 0
    for case, case_row in zip(cases, result.rows):
        data = case_row.to_dict()
        verdict = (data.get("outcome") or {}).get("verdict")
        verdicts[verdict] = verdicts.get(verdict, 0) + 1
        family = families.setdefault(case.category, {"n": 0, "success": 0, "unknown": 0})
        family["n"] += 1
        family["success"] += int(verdict == "success")
        family["unknown"] += int(verdict == "unknown")
        if verdict in ("success", "failure"):
            decided += 1
            agreed += int((verdict == "success") == (own.get(case.id, 0.0) >= 0.5))
        for key, entry in (data.get("diagnostics") or {}).items():
            value = (entry or {}).get("value")
            if isinstance(value, (int, float)):
                buckets.setdefault(key, []).append(float(value))
    return {
        "outcome": verdicts,
        "success": round(verdicts.get("success", 0) / len(cases), 4) if cases else None,
        "agreement_with_own_scorer": round(agreed / decided, 4) if decided else None,
        "diagnostics": {key: _mean(values) for key, values in sorted(buckets.items())},
        "families": families,
    }


def score_run(run_dir: Path, tasks_path: Path, out_dir: Path) -> dict:
    from auditkit import __version__ as auditkit_version
    from auditkit.agent_eval import (
        AgentCase,
        AgentEvalRunner,
        AgentEvalSpec,
        CustomPredicate,
        FinalStateAssertion,
        episode_from_openai_messages,
    )
    from auditkit.agent_eval.types import OBSERVED

    tasks = {task["id"]: task for task in load_jsonl(tasks_path)}
    rows = load_jsonl(run_dir / "trajectories.jsonl")

    ledger_cases, full_cases, episodes, ledger_episodes, own = [], [], [], [], {}
    for row in rows:
        task_id = (row.get("metadata") or {}).get("task_id")
        task = tasks.get(task_id)
        if task is None:
            continue
        ledger_cases.append(build_case(AgentCase, FinalStateAssertion, CustomPredicate, task))
        full_cases.append(
            build_case(AgentCase, FinalStateAssertion, CustomPredicate, task, include_finish=True)
        )
        episode = build_episode(episode_from_openai_messages, OBSERVED, task, row)
        episodes.append(episode)
        ledger_episodes.append(_drop_finish(episode))
        own[task["id"]] = float(((row.get("metadata") or {}).get("scores") or {}).get("success") or 0.0)

    runner = AgentEvalRunner()
    ledger_result = runner.run(
        AgentEvalSpec(cases=ledger_cases, mode="recorded", episodes=ledger_episodes,
                      scorers=list(LEDGER_SCORERS), name=f"{run_dir.name}-ledger-route")
    )
    full_result = runner.run(
        AgentEvalSpec(cases=full_cases, mode="recorded", episodes=episodes,
                      scorers=list(FULL_SCORERS), name=f"{run_dir.name}-full-route")
    )

    ledger = _aggregate(ledger_result, own, ledger_cases)
    full = _aggregate(full_result, own, full_cases)
    summary = {
        "run": run_dir.name,
        "tasks": tasks_path.name,
        "auditkit_version": auditkit_version,
        "n": len(ledger_cases),
        "outcome": ledger["outcome"],
        "success": ledger["success"],
        "agreement_with_own_scorer": ledger["agreement_with_own_scorer"],
        "ledger_route": {"scorers": list(LEDGER_SCORERS), **ledger["diagnostics"]},
        "full_route": {"scorers": list(FULL_SCORERS), **full["diagnostics"]},
        "families": ledger["families"],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    ledger_result.to_json(str(out_dir / "auditkit_ledger_route.json"))
    full_result.to_json(str(out_dir / "auditkit_full_route.json"))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, help="run folder under runs/")
    parser.add_argument("--tasks", default=str(ROOT / "data" / "tasks_hard_eval.jsonl"))
    parser.add_argument("--out", default=None, help="exact output folder (default runs/auditkit_<run>)")
    args = parser.parse_args()

    for name in args.run:
        run_dir = Path(name)
        if not run_dir.is_absolute():
            run_dir = ROOT / "runs" / name
        out_dir = Path(args.out) if args.out else ROOT / "runs" / f"auditkit_{run_dir.name}"
        summary = score_run(run_dir, Path(args.tasks), out_dir)
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
