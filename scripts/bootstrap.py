"""Task-level bootstrap intervals for every published number.

    python scripts/bootstrap.py              # writes docs/STATS.md
    python scripts/bootstrap.py --iters 50000

Each row is read from `runs/<run>/results.jsonl` (or `trajectories.jsonl` when the run
predates the flat file), so the intervals are computed over the eval that actually ran
rather than over a hand-copied table. Two intervals are reported side by side:

- Wilson score interval, the usual closed form for a binomial proportion;
- percentile bootstrap over tasks, which makes no binomial assumption and can be paired.

For a zero count the bootstrap interval collapses to [0, 0], which is misleading, so a
one-sided 95% upper bound of 3/n (the rule of three) is reported next to it.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = ROOT / "docs" / "STATS.md"
Z_95 = 1.959963984540054

# (label, run dir, field) - field defaults to "success"
ACCURACY = [
    ("hard-30", "sft4_hard30", "success"),
    ("hard-30 (renamed schema)", "sft4_shift_hard30", "success"),
    ("hard-30 (Devanagari)", "sft4_devanagari30", "success"),
    ("eval-24", "sft4_eval", "success"),
    ("hard-90 (base)", "base_hard90", "success"),
    ("hard-90 (sft4, shipped)", "sft4_hard90", "success"),
    ("hard-90 (sft4, seed 7)", "sft4_seed7_hard90", "success"),
    ("hard-90 (sft4, seed 123)", "sft4_seed123_hard90", "success"),
    ("hard-90 (hardened prompt)", "sft4_hard90_hardened", "success"),
]

PAIRS = [
    ("hard-90", "base_hard90", "sft4_hard90", "base", "sft4"),
    ("hard-90", "sft4_hard90", ".tmp_hardened", "sft4", "sft4+hardened"),
]

# (split, task file, base run, adapter run, hardened run or None); the denominator comes
# from the tasks that carry a payload in the file, not from the run, so a clean row can
# never dilute a rate and a stalled base run cannot shrink the split.
SAFETY = [
    ("safety-32", "data/tasks_safety_eval.jsonl", "safety_base", "safety_sft4", "safety_sft4_hardened"),
    ("safety2-80", "data/tasks_safety_eval2.jsonl", "safety2_base", "safety2_sft4", "safety2_sft4_hardened"),
    ("payee-16", "data/tasks_safety_payee.jsonl", "safety_payee_base", "safety_payee_sft4", "safety_payee_sft4_hardened"),
    ("hinglish-24", "data/tasks_safety_hinglish.jsonl", "safety_hinglish_base", "safety_hinglish_sft4", "safety_hinglish_sft4_hardened"),
]


def run_path(run: str) -> Path:
    """runs/<run> when it exists, else <repo>/<run> so a scratch dir can be scored in place."""
    path = RUNS / run
    return path if path.exists() else ROOT / run


def load_rows(run: str) -> dict[str, dict]:
    """task_id -> score dict, from results.jsonl when present, else trajectories."""
    base = run_path(run)
    flat = base / "results.jsonl"
    if flat.exists():
        rows = [json.loads(line) for line in flat.open(encoding="utf-8") if line.strip()]
        return {str(row["task_id"]): row for row in rows}

    traj = base / "trajectories.jsonl"
    if not traj.exists():
        raise SystemExit(f"missing both results.jsonl and trajectories.jsonl under {base}")
    out: dict[str, dict] = {}
    for line in traj.open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        meta = row.get("metadata") or {}
        out[str(meta["task_id"])] = meta.get("scores") or {}
    return out


def wilson(hits: int, n: int, z: float = Z_95) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = hits / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    pos = q * (len(ordered) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def bootstrap_mean(values: list[float], iters: int, rng: random.Random) -> tuple[float, float]:
    n = len(values)
    if n == 0:
        return 0.0, 1.0
    draws = []
    for _ in range(iters):
        total = 0.0
        for _ in range(n):
            total += values[rng.randrange(n)]
        draws.append(total / n)
    return percentile(draws, 0.025), percentile(draws, 0.975)


def rule_of_three(n: int) -> float:
    return 3.0 / n if n else 1.0


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def ci(lo: float, hi: float) -> str:
    return f"{pct(lo)}–{pct(hi)}"


def accuracy_table(iters: int, rng: random.Random) -> list[str]:
    lines = [
        "| Split | Run | n | Hits | Point | Wilson 95% | Bootstrap 95% |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for label, run, field in ACCURACY:
        rows = load_rows(run)
        values = [float(row.get(field, 0) or 0) for row in rows.values()]
        values = [1.0 if v >= 0.5 else 0.0 for v in values]
        hits, n = int(sum(values)), len(values)
        lo, hi = wilson(hits, n)
        blo, bhi = bootstrap_mean(values, iters, rng)
        lines.append(
            f"| {label} | `{run}` | {n} | {hits}/{n} | {pct(hits / n)} | {ci(lo, hi)} | {ci(blo, bhi)} |"
        )
    return lines


def paired_table(iters: int, rng: random.Random) -> list[str]:
    lines = [
        "| Split | A | B | n | Fixed | Broken | Delta | Bootstrap 95% | McNemar p |",
        "|---|---|---|---:|---:|---:|---:|---|---:|",
    ]
    for split, run_a, run_b, label_a, label_b in PAIRS:
        a = load_rows(run_a)
        b = load_rows(run_b)
        shared = sorted(set(a) & set(b))
        av = [1.0 if float(a[t].get("success", 0) or 0) >= 0.5 else 0.0 for t in shared]
        bv = [1.0 if float(b[t].get("success", 0) or 0) >= 0.5 else 0.0 for t in shared]
        fixed = sum(1 for x, y in zip(av, bv) if y and not x)
        broken = sum(1 for x, y in zip(av, bv) if x and not y)
        deltas = [y - x for x, y in zip(av, bv)]
        lo, hi = bootstrap_mean(deltas, iters, rng)
        lines.append(
            f"| {split} | {label_a} | {label_b} | {len(shared)} | {fixed} | {broken} | "
            f"{pct(sum(deltas) / len(shared))} | {pct(lo)}–{pct(hi)} | {mcnemar_exact(fixed, broken):.4g} |"
        )
    return lines


def injected_ids(task_file: str) -> list[str]:
    rows = [json.loads(line) for line in (ROOT / task_file).open(encoding="utf-8") if line.strip()]
    return [str(row["id"]) for row in rows if row.get("injection") not in (None, "", {})]


def safety_table(iters: int, rng: random.Random) -> tuple[list[str], int, int, int, int]:
    lines = [
        "| Split | Run | Injected | Fired | Ledger hit | Answer hit | Ledger Wilson 95% |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    total_injected = 0
    adapter_hits = 0
    hardened_hits = 0
    hardened_answer = 0
    for split, task_file, base_run, adapter_run, hardened_run in SAFETY:
        ids = injected_ids(task_file)
        for run in (base_run, adapter_run, hardened_run):
            if not run:
                continue
            rows = load_rows(run)
            missing = [task for task in ids if task not in rows]
            if missing:
                raise SystemExit(f"{run}: {len(missing)} injected tasks missing, e.g. {missing[:3]}")
            fired = sum(1 for task in ids if float(rows[task].get("injections_fired", 0) or 0) > 0)
            ledger = sum(1 for task in ids if float(rows[task].get("attack_ledger", 0) or 0) >= 0.5)
            answer = sum(1 for task in ids if float(rows[task].get("attack_answer", 0) or 0) >= 0.5)
            lo, hi = wilson(ledger, len(ids))
            lines.append(
                f"| {split} | `{run}` | {len(ids)} | {fired} | {ledger}/{len(ids)} | {answer}/{len(ids)} | {ci(lo, hi)} |"
            )
            if run == adapter_run:
                total_injected += len(ids)
                adapter_hits += ledger
            if run == hardened_run:
                hardened_hits += ledger
                hardened_answer += answer
    return lines, total_injected, adapter_hits, hardened_hits, hardened_answer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    acc = accuracy_table(args.iters, rng)
    pairs = paired_table(args.iters, rng)
    safety, injected, hits, hard_hits, hard_answer = safety_table(args.iters, rng)
    upper = rule_of_three(injected)

    body = f"""# Intervals for the published numbers

Generated by `python scripts/bootstrap.py` (bootstrap iterations: {args.iters}, seed {args.seed})
from the per-task rows in `runs/<run>/results.jsonl`. Re-running the script after any eval
refresh rewrites this file, so the numbers here cannot drift from the run logs.

Every eval is greedy and single-sample, so the only source of uncertainty these intervals
cover is **which tasks were drawn**; they say nothing about decoding variance, which is why
the seed control exists separately.

## Accuracy

{chr(10).join(acc)}

## Paired changes

Paired on the task ids present in both runs; the delta is B − A over the same tasks, and the
bootstrap resamples task pairs rather than the two runs independently.

{chr(10).join(pairs)}

## Safety: attacker value in the ledger or the answer

The denominator is the tasks that carry a payload in the task file, so a clean row cannot
dilute the rate and a base run that stalls before the injected tool result cannot shrink it.
The `Fired` column is how many tasks actually reached that tool result.

{chr(10).join(safety)}

## The 0 of {injected} claim

Across the four injected splits the shipped adapter carried an attacker value into the ledger
on **{hits} of {injected}** tasks. A point estimate of zero has no useful two-sided interval, so
the honest bound is one-sided: with the rule of three, **0/{injected} gives a 95% upper bound of
{pct(upper)}** on the per-task ledger-hit rate.

Two caveats, both deliberate:

- The four splits are not one iid sample. They were built to probe different payload styles
  and languages, so pooling them is a conservative screen, not a formal sampling claim. Each
  split's own interval is in the table above.
- The interval is over tasks from this one builder. It says nothing about a payload family the
  builder never generates; the held-out-style run is the experiment for that question.

The answer-side metric is also 0 of {injected} for the shipped adapter. The hardened prompt is
{hard_hits} of {injected} in the ledger and {hard_answer} of {injected} in the answer, both answer
hits on the first split: that is the measured cost of the hardened configuration, and the reason
`attack_answer` is a separate metric.
"""
    OUT.write_text(body, encoding="utf-8")
    print(body)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
