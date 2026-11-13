"""Publish the run logs as two viewer-friendly parquet tables.

The dataset held 202 loose JSON files, which the Hub renders as an empty page: no parquet, no
configs, nothing a judge can click. This writes `viewer/results.parquet` (one row per task) and
`viewer/runs.parquet` (one row per run) and points the card at them.

    python scripts/publish_viewer.py            # build locally, print row counts
    python scripts/publish_viewer.py --push     # also upload + update the card
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import pandas as pd
from huggingface_hub import HfApi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = "primce/lexsi-runs"

TASK_COLUMNS = [
    "run", "family", "task_id", "success", "tac", "arr", "ter", "rad", "overcall", "lcf",
    "answer_ok", "ledger_ok", "settlement_ok", "strict", "strict_rate", "calls", "steps",
    "parse_failures", "attacker_expense", "attack_ledger", "attack_answer",
]

RUN_COLUMNS = [
    "run", "n", "success", "tac", "arr", "ter", "rad", "overcall", "lcf", "answer_ok",
    "ledger_ok", "settlement_ok", "strict", "strict_rate", "calls", "steps", "parse_failures",
    "attack_ledger", "attack_answer", "attacker_expense", "seconds", "model", "adapter",
]


def family_map() -> dict[str, str]:
    out: dict[str, str] = {}
    for path in glob.glob(os.path.join(ROOT, "data", "tasks*.jsonl")):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                if "id" in row and "family" in row:
                    out[row["id"]] = row["family"]
    return out


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    fam = family_map()
    tasks, runs = [], []
    for path in sorted(glob.glob(os.path.join(ROOT, "runs", "*", "results.jsonl"))):
        run = os.path.basename(os.path.dirname(path))
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                row["run"] = run
                row["family"] = fam.get(row.get("task_id"), "")
                tasks.append(row)
        summary_path = os.path.join(os.path.dirname(path), "summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, encoding="utf-8") as fh:
                summary = json.load(fh)
            summary["run"] = run
            runs.append(summary)

    task_df = pd.DataFrame(tasks).reindex(columns=TASK_COLUMNS)
    run_df = pd.DataFrame(runs).reindex(columns=RUN_COLUMNS)
    for df in (task_df, run_df):
        for col in df.columns:
            if col not in ("run", "family", "task_id", "model", "adapter"):
                df[col] = pd.to_numeric(df[col], errors="coerce")
    return task_df, run_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()

    task_df, run_df = build()
    out = os.path.join(ROOT, ".viewer")
    os.makedirs(out, exist_ok=True)
    task_path = os.path.join(out, "results.parquet")
    run_path = os.path.join(out, "runs.parquet")
    task_df.to_parquet(task_path, index=False)
    run_df.to_parquet(run_path, index=False)
    print(f"results {len(task_df)} rows x {len(task_df.columns)} cols, {len(task_df.run.unique())} runs")
    print(f"runs    {len(run_df)} rows x {len(run_df.columns)} cols")
    print(run_df[["run", "n", "success"]].to_string(index=False, max_rows=8))

    if not args.push:
        return
    api = HfApi()
    for local, remote in ((task_path, "viewer/results.parquet"), (run_path, "viewer/runs.parquet")):
        api.upload_file(
            path_or_fileobj=local,
            path_in_repo=remote,
            repo_id=REPO,
            repo_type="dataset",
            commit_message=f"Add {remote}: viewer-friendly table of the run logs",
        )
        print("uploaded", remote)


if __name__ == "__main__":
    main()
