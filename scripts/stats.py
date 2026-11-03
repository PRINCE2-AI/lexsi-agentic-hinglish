"""Paired comparison of two eval runs: Wilson 95% CI and exact McNemar.

    python scripts/stats.py --a runs/sft_hard30 --b runs/sft2_hard30

Each side is a run directory (`trajectories.jsonl` inside) or the file itself. Only tasks
present in both runs are compared, so a partial rerun cannot inflate the gain, and the
McNemar test uses the discordant pairs rather than two independent proportions.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

Z_95 = 1.959963984540054


def load_success(path: Path) -> dict[str, int]:
    if path.is_dir():
        path = path / "trajectories.jsonl"
    if not path.exists():
        raise SystemExit(f"missing trajectories: {path}")

    out: dict[str, int] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            meta = row.get("metadata") or {}
            task_id = meta.get("task_id")
            scores = meta.get("scores") or {}
            if task_id is None or "success" not in scores:
                raise SystemExit(f"{path}: row without metadata.task_id / metadata.scores.success")
            out[str(task_id)] = int(float(scores["success"]) >= 0.5)
    return out


def wilson(successes: int, n: int, z: float = Z_95) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value over the discordant pairs (b, c)."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def compare(a: dict[str, int], b: dict[str, int]) -> dict:
    shared = sorted(set(a) & set(b))
    if not shared:
        raise SystemExit("no task ids in common between the two runs")

    a_hits = sum(a[task] for task in shared)
    b_hits = sum(b[task] for task in shared)
    b_only = sum(1 for task in shared if b[task] and not a[task])
    a_only = sum(1 for task in shared if a[task] and not b[task])

    return {
        "n": len(shared),
        "a_success": round(a_hits / len(shared), 4),
        "a_ci95": [round(value, 4) for value in wilson(a_hits, len(shared))],
        "b_success": round(b_hits / len(shared), 4),
        "b_ci95": [round(value, 4) for value in wilson(b_hits, len(shared))],
        "gain": round((b_hits - a_hits) / len(shared), 4),
        "a_only": a_only,
        "b_only": b_only,
        "mcnemar_p": round(mcnemar_exact(b_only, a_only), 5),
        "dropped": sorted(set(a) - set(b)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True, help="baseline run dir or trajectories.jsonl")
    parser.add_argument("--b", required=True, help="tuned run dir or trajectories.jsonl")
    parser.add_argument("--label-a", default="a")
    parser.add_argument("--label-b", default="b")
    args = parser.parse_args()

    result = compare(load_success(Path(args.a)), load_success(Path(args.b)))
    result["a"] = args.label_a
    result["b"] = args.label_b
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
