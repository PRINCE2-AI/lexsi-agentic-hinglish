"""Is held-out success flat across how much a task looks like training text?

A trained model can pass a held-out split by recognising wording rather than by following the
condition. `scripts/template_split.py` answers that for wording that was seen verbatim; this asks
the softer version: take each held-out task's nearest training task by TF-IDF cosine, put the tasks
in four buckets, and read success off each bucket. A model that memorised wording scores high in
the top bucket and lower below it.

    python scripts/similarity_split.py
    python scripts/similarity_split.py --runs base_hard90 sft4_hard90 --json out.json

No model, no GPU: everything comes from the pinned task files and the published run folders.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EVAL = ROOT / "data" / "tasks_hard90_eval.jsonl"
TRAIN = ROOT / "data" / "tasks_train_mix3.jsonl"
DEFAULT_RUNS = ["base_hard90", "sft4_hard90"]
WORD = re.compile(r"[0-9a-zA-Z\u0900-\u097F]+")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def tokens(text: str) -> list[str]:
    words = [word.lower() for word in WORD.findall(text or "")]
    return words + [f"{a}_{b}" for a, b in zip(words, words[1:])]


def idf(docs: list[list[str]]) -> dict[str, float]:
    df: Counter = Counter()
    for doc in docs:
        df.update(set(doc))
    total = len(docs)
    return {term: math.log((total + 1) / (count + 1)) + 1.0 for term, count in df.items()}


def vector(doc: list[str], weights: dict[str, float]) -> dict[str, float]:
    counts = Counter(doc)
    vec = {term: count * weights.get(term, 1.0) for term, count in counts.items()}
    norm = math.sqrt(sum(value * value for value in vec.values())) or 1.0
    return {term: value / norm for term, value in vec.items()}


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(value * b.get(term, 0.0) for term, value in a.items())


def nearest_similarity() -> dict[str, dict]:
    train_docs = [tokens(row.get("task") or "") for row in load_jsonl(TRAIN)]
    weights = idf(train_docs)
    train_vecs = [vector(doc, weights) for doc in train_docs]
    out = {}
    for row in load_jsonl(EVAL):
        vec = vector(tokens(row.get("task") or ""), weights)
        best = max(cosine(vec, other) for other in train_vecs)
        out[row["id"]] = {"family": row.get("family", ""), "similarity": round(best, 4)}
    return out


def run_success(run: str) -> dict[str, bool]:
    path = ROOT / "runs" / run / "results.jsonl"
    if not path.exists():
        raise SystemExit(f"missing run {run}: fetch it with scripts/fetch_runs.py --only {run}")
    return {
        row["task_id"]: bool(row.get("success"))
        for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    }


def quartile_table(sim: dict[str, dict], runs: dict[str, dict[str, bool]]) -> list[dict]:
    ordered = sorted(sim.items(), key=lambda item: item[1]["similarity"])
    size = math.ceil(len(ordered) / 4)
    rows = []
    for index in range(4):
        chunk = ordered[index * size : (index + 1) * size]
        if not chunk:
            continue
        row = {
            "bucket": f"Q{index + 1}",
            "n": len(chunk),
            "similarity": [chunk[0][1]["similarity"], chunk[-1][1]["similarity"]],
        }
        for run, scored in runs.items():
            hits = sum(1 for task_id, _ in chunk if scored.get(task_id))
            row[run] = round(hits / len(chunk), 4)
            row[f"{run}_hits"] = hits
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
    parser.add_argument("--json", default="")
    args = parser.parse_args()

    sim = nearest_similarity()
    runs = {run: run_success(run) for run in args.runs}
    rows = quartile_table(sim, runs)

    head = f"{'bucket':6s} {'n':>3s} {'nearest cosine':>16s} " + " ".join(f"{run:>14s}" for run in runs)
    print(head)
    for row in rows:
        span = f"{row['similarity'][0]:.2f}-{row['similarity'][1]:.2f}"
        cells = " ".join(f"{row[run] * 100:13.1f}%" for run in runs)
        print(f"{row['bucket']:6s} {row['n']:3d} {span:>16s} {cells}")

    print("\nseen-vs-unseen wording (scripts/template_split.py) remains the strict version of this check.")
    if args.json:
        Path(args.json).write_text(json.dumps({"quartiles": rows, "per_task": sim}, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
