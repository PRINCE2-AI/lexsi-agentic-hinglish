"""Join sharded eval runs into one run directory.

Kaggle sessions stop at 12 hours, so a 257-task run can be executed as `--offset 0 --limit 90`,
`--offset 90 --limit 90`, ... This concatenates the shards in offset order and recomputes the
summary from the joined rows, so the published summary is never hand-assembled.

    python scripts/merge_runs.py --out runs/sft4_hard257 runs/sft4_hard257_a runs/sft4_hard257_b
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hinglish_agent.metrics import summarise  # noqa: E402


def shard_key(path: Path) -> int:
    summary = path / "summary.json"
    if not summary.exists():
        raise SystemExit(f"{path} has no summary.json")
    return int(json.loads(summary.read_text(encoding="utf-8")).get("offset") or 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("shards", nargs="+")
    args = parser.parse_args()

    shards = sorted((Path(s) for s in args.shards), key=shard_key)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    scores = []
    seconds = 0.0
    base_summary: dict | None = None
    with (out / "results.jsonl").open("w", encoding="utf-8", newline="\n") as results, (
        out / "trajectories.jsonl"
    ).open("w", encoding="utf-8", newline="\n") as trajectories:
        for shard in shards:
            summary = json.loads((shard / "summary.json").read_text(encoding="utf-8"))
            seconds += float(summary.get("seconds") or 0)
            if base_summary is None:
                base_summary = {k: v for k, v in summary.items() if k != "families"}
            rows = [
                json.loads(line)
                for line in (shard / "results.jsonl").open(encoding="utf-8")
                if line.strip()
            ]
            kept: set[str] = set()
            for row in rows:
                task_id = row.get("task_id")
                if task_id in seen:
                    continue
                if task_id:
                    seen.add(task_id)
                    kept.add(task_id)
                scores.append(row)
                results.write(json.dumps(row, ensure_ascii=False) + "\n")
            with (shard / "trajectories.jsonl").open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if not row.get("task_id") or row["task_id"] in kept:
                        trajectories.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = summarise(scores)
    for key, value in (base_summary or {}).items():
        summary.setdefault(key, value)
    summary["seconds"] = round(seconds, 1)
    summary["shards"] = [os.path.basename(str(s)) for s in shards]
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"{len(scores)} episodes from {len(shards)} shards -> {out}")
    print(json.dumps({k: summary[k] for k in ("n", "success", "tac", "arr", "calls", "strict_rate")}))


if __name__ == "__main__":
    main()
