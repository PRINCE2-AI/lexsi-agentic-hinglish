"""Recompute strict-JSON metrics from saved trajectories (no GPU, no model).

The Nov 2 baseline ran before the strict metric existed. Its trajectories already
carry `metadata.conversation`, so the strict numbers can be recovered without a
34-minute re-run:

    python scripts/score_strict.py --trajectories runs/baseline/trajectories.jsonl \
        --summary runs/baseline/summary.json

Future runs compute these automatically in `score_episode`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hinglish_agent.runner import is_strict_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--summary", default=None)
    args = parser.parse_args()

    path = Path(args.trajectories)
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]

    strict_tasks, strict_rates = [], []
    for row in rows:
        conversation = (row.get("metadata") or {}).get("conversation") or []
        turns = [m for m in conversation if m.get("role") == "assistant"]
        hits = sum(1 for m in turns if is_strict_json(m.get("content", "")))
        strict_tasks.append(1.0 if turns and hits == len(turns) else 0.0)
        strict_rates.append(hits / len(turns) if turns else 0.0)

    result = {
        "n": len(rows),
        "strict": round(sum(strict_tasks) / len(rows), 4) if rows else 0.0,
        "strict_rate": round(sum(strict_rates) / len(rows), 4) if rows else 0.0,
    }
    print(json.dumps(result, indent=2))

    if args.summary:
        summary_path = Path(args.summary)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary.update(result)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"updated {summary_path}")


if __name__ == "__main__":
    main()
