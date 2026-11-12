"""Measure the guard against payloads written after reading its rules.

The published payloads (`data/tasks_safety_*.jsonl`) and the blind set were written without
knowledge of `hinglish_agent/guard.py`. This set is the opposite: eight payloads whose only job
is to dodge every rule that file implements while still pushing a 999999 INR expense for Neha.
An attacker who reads the source is the honest upper bound on a rule-based filter.

Two modes are measured on the same eight:

* `flag` -- the text rules, on the string a tool result carries.
* `allowlist` -- the structural rule: a field the responding tool never returns is dropped.

    python scripts/adaptive_probe.py
    python scripts/adaptive_probe.py --json runs/adaptive_probe.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from guard_eval import load_traffic  # noqa: E402
from hinglish_agent.guard import USER_TEXT_KEYS, scan_result, sanitize  # noqa: E402

SET = "data/payloads_adaptive.jsonl"


def as_result(channel: str, text: str) -> dict:
    """The tool result an injected `note` or a poisoned `description` would arrive in."""
    if channel == "description":
        return {"ok": True, "description": text, "total_inr": 1200}
    return {"ok": True, "total_inr": 1200, "note": text}


def measure() -> list[dict]:
    rows = [json.loads(line) for line in (ROOT / SET).read_text(encoding="utf-8").splitlines() if line.strip()]
    out = []
    for row in rows:
        parts = []
        for text in row["parts"]:
            result = as_result(row["channel"], text)
            flag = scan_result(result)
            _, allow_flag = sanitize(result, mode="allowlist", tool_name="add_expense")
            parts.append({"flagged": flag["flagged"], "reasons": flag["strong"],
                          "dropped": allow_flag.get("dropped", [])})
        out.append({"id": row["id"], "style": row["style"], "channel": row["channel"],
                    "parts": parts,
                    "caught_by_text": any(p["flagged"] for p in parts),
                    "caught_by_allowlist": any(p["dropped"] for p in parts)})
    return out


def clean_traffic() -> dict:
    """Does the new user-text rule fire on traffic with no injection at all?"""
    descriptions = flagged = 0
    for episode in load_traffic():
        if not episode["clean"]:
            continue
        for step in episode["steps"]:
            result = step.get("result")
            if not isinstance(result, dict):
                continue
            for key in USER_TEXT_KEYS:
                if not isinstance(result.get(key), str):
                    continue
                descriptions += 1
                flagged += bool(scan_result(result)["reasons"].count("amount_in_user_text"))
    return {"descriptions": descriptions, "flagged": flagged}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default="")
    args = parser.parse_args()

    rows = measure()
    for row in rows:
        caught = row["caught_by_text"] or row["caught_by_allowlist"]
        how = "text" if row["caught_by_text"] else ("allowlist" if row["caught_by_allowlist"] else "-")
        print(f"{row['id']}  {row['style']:<26} {row['channel']:<11} "
              f"{'caught' if caught else 'MISS':<6} {how}")

    text = sum(r["caught_by_text"] for r in rows)
    both = sum(r["caught_by_text"] or r["caught_by_allowlist"] for r in rows)
    clean = clean_traffic()
    print()
    print(f"text rules: {text}/{len(rows)}   with allowlist: {both}/{len(rows)}")
    print(f"clean traffic: {clean['flagged']} of {clean['descriptions']} descriptions carry a six-figure amount")

    if args.json:
        payload = {"rows": rows, "text": text, "both": both, "clean": clean, "n": len(rows)}
        Path(args.json).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
