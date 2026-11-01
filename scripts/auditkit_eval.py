"""Load AgentTune/Agent run logs into AuditKIT's agent evaluation layer.

Run after a baseline or tuned run:
    python scripts/auditkit_eval.py --trajectories runs/baseline/trajectories.jsonl

For the full scored report (verified outcome, tool/trace/judge diagnostics) use
AuditKIT's example notebook 14:
https://github.com/Lexsi-Labs/AuditKIT/blob/main/examples/14_agent_and_rag_evals.ipynb
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", default="runs/baseline/trajectories.jsonl")
    parser.add_argument("--model", default="hf:runs/tuned")
    args = parser.parse_args()

    path = Path(args.trajectories)
    if not path.exists():
        raise SystemExit(f"missing trajectories file: {path}")

    try:
        from auditkit.agent_eval import episodes_from_agenttune
    except ImportError as exc:
        raise SystemExit(
            "auditkit is not installed. Run: pip install \"auditkit[all]\" "
            "or pip install -e '.[transformers]' from the AuditKIT repo."
        ) from exc

    episodes = episodes_from_agenttune(str(path))
    print(f"loaded {len(episodes)} episodes from {path}")
    print(f"score them against model: {args.model}")
    if episodes:
        first = episodes[0]
        print("first episode keys:", sorted(first.keys()) if isinstance(first, dict) else type(first))
        print(json.dumps(first, indent=2, default=str)[:1500])
    print("\nNext: open AuditKIT examples/14_agent_and_rag_evals.ipynb and pass these episodes.")


if __name__ == "__main__":
    main()
