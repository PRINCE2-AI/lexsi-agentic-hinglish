"""Does the answer agree with the ledger the same episode produced?

`success` compares the final ledger with the task's expected balances. This asks a different
question, and one a user would ask: of the amounts the reply states, how many can the ledger in
that same conversation produce? An answer that names a balance the tools never returned is a
wrong answer even when the ledger happens to be right, and an answer that stays inside the ledger
is worth quoting on its own.

    python scripts/answer_ledger_audit.py
    python scripts/answer_ledger_audit.py --run sft4_hard90 --json out.json

Candidates for a legitimate amount: every balance the tools returned, every expense amount, every
per-person share the tools returned, and every transfer `hinglish_agent.ledger.settle_ledger`
derives from that final ledger. Anything else the reply states as money is counted as invented.
Responses that name transfers (`A -> B 123.45 INR`) are also checked as a set against the
settlement the ledger implies.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hinglish_agent.ledger import public_balances, settle_ledger  # noqa: E402

DEFAULT_RUNS = ["base_hard90", "sft4_hard90"]
TRANSFER = re.compile(
    r"([A-Z][a-zA-Z]+)\s*(?:->|→|to|ko)\s*([A-Z][a-zA-Z]+)[^0-9₹]{0,16}(?:₹|Rs\.?|INR)?\s*([0-9][0-9,]*(?:\.[0-9]+)?)",
)
AMOUNT = re.compile(r"(?:₹|Rs\.?|INR)\s*([0-9][0-9,]*(?:\.[0-9]+)?)|([0-9][0-9,]*\.[0-9]+)")


def to_float(text: str) -> float:
    return round(float(text.replace(",", "")), 2)


def final_ledger(steps: list[dict]) -> dict | None:
    for step in reversed(steps):
        result = step.get("result")
        if isinstance(result, dict) and isinstance(result.get("ledger"), dict):
            return result["ledger"]
    return None


def candidate_amounts(steps: list[dict], ledger: dict) -> set[float]:
    values: set[float] = {0.0}  # a settled ledger can always be quoted as zero

    def take(number) -> None:
        try:
            values.add(round(abs(float(number)), 2))
        except (TypeError, ValueError):
            return

    for person_amount in public_balances(ledger).values():
        take(person_amount)
    for transfer in settle_ledger(ledger):
        take(transfer["amount_inr"])
    totals = []
    for step in steps:
        result = step.get("result")
        if not isinstance(result, dict):
            continue
        for share in result.get("share_inr") or []:
            take(share)
        if "total_inr" in result:
            take(result["total_inr"])
            totals.append(result["total_inr"])
        for key in ("balances_cents",):
            for cents in (result.get("ledger") or {}).get(key, {}).values():
                take(cents / 100)
    if totals:
        take(sum(totals))
    arguments = [step.get("arguments") for step in steps if step.get("name") == "add_expense"]
    for args in arguments:
        if isinstance(args, dict):
            take(args.get("amount"))
    return values


def audit_run(run: str, json_only: bool = False) -> dict:
    folder = ROOT / "runs" / run
    episodes = [
        json.loads(line)
        for line in (folder / "trajectories.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    scored = {}
    results_path = folder / "results.jsonl"
    if results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                scored[row["task_id"]] = row

    rows = []
    for episode in episodes:
        task_id = (episode.get("metadata") or {}).get("task_id") or ""
        answer = episode.get("final_response") or ""
        ledger = final_ledger(episode.get("steps") or [])
        entry = {
            "task_id": task_id,
            "success": bool(scored.get(task_id, {}).get("success")),
            "has_answer": bool(answer.strip()),
            "transfers_stated": 0,
            "transfers_match_ledger": None,
            "amounts_stated": 0,
            "amounts_not_in_ledger": 0,
            "invented": [],
        }
        if answer.strip() and ledger is not None:
            candidates = candidate_amounts(episode["steps"], ledger)
            stated = [to_float(a or b) for a, b in AMOUNT.findall(answer)]
            entry["amounts_stated"] = len(stated)
            invented = [value for value in stated if not any(abs(value - c) < 0.005 for c in candidates)]
            entry["amounts_not_in_ledger"] = len(invented)
            entry["invented"] = invented[:6]
            pairs = {(giver, taker, to_float(amount)) for giver, taker, amount in TRANSFER.findall(answer)}
            if pairs:
                implied = {(t["from"], t["to"], round(float(t["amount_inr"]), 2)) for t in settle_ledger(ledger)}
                entry["transfers_stated"] = len(pairs)
                entry["transfers_match_ledger"] = pairs == implied
        rows.append(entry)

    n = len(rows)
    answered = sum(1 for row in rows if row["has_answer"])
    clean = sum(1 for row in rows if row["has_answer"] and row["amounts_not_in_ledger"] == 0)
    with_transfers = [row for row in rows if row["transfers_stated"]]
    transfer_ok = sum(1 for row in with_transfers if row["transfers_match_ledger"])
    summary = {
        "run": run,
        "n": n,
        "answered": answered,
        "answers_without_an_invented_amount": clean,
        "answers_with_transfers": len(with_transfers),
        "transfers_match_the_ledger": transfer_ok,
        "invented_amount_total": sum(row["amounts_not_in_ledger"] for row in rows),
        "tasks_with_invented_amounts": [row["task_id"] for row in rows if row["amounts_not_in_ledger"]],
        "success_rate": round(sum(1 for row in rows if row["success"]) / n, 3) if n else 0.0,
    }
    if not json_only:
        print(f"\n{run}: n={n} success={summary['success_rate']:.1%} answered={answered}")
        print(
            f"  answers with no amount outside the ledger: {clean}/{answered}  "
            f"(invented amounts: {summary['invented_amount_total']} across {len(summary['tasks_with_invented_amounts'])} tasks)"
        )
        print(f"  answers stating transfers: {len(with_transfers)}, matching the ledger's settlement: {transfer_ok}")
        worst = sorted(rows, key=lambda row: -row["amounts_not_in_ledger"])[:5]
        for row in worst:
            if row["amounts_not_in_ledger"]:
                print(f"    {row['task_id']}: {row['invented']}")
    return {"summary": summary, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", default=[])
    parser.add_argument("--json", default="")
    args = parser.parse_args()

    runs = args.run or DEFAULT_RUNS
    out = [audit_run(run) for run in runs]
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
