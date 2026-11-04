"""Print and upload the run folders a GPU job produced.

    python scripts/job_upload.py --run sft3_hard30 --run safety_sft3 \
        --adapter runs/sft3:sft3_adapter

Run directories go to `primce/lexsi-runs` (dataset), adapters to
`primce/tiny-aya-hinglish-lora`. Missing folders print as MISSING and are skipped,
so a half-finished job still publishes whatever it finished.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS_REPO = "primce/lexsi-runs"
MODEL_REPO = "primce/tiny-aya-hinglish-lora"

KEYS = (
    "n", "success", "tac", "arr", "ter", "rad", "overcall", "lcf", "answer_ok",
    "strict_rate", "calls", "steps", "parse_failures", "seconds",
    "safe", "utility", "attack_ledger", "attack_answer", "unplanned_calls",
    "abstained", "asked", "invented_expense", "injections_fired",
)


def report(run: Path) -> None:
    summary_path = run / "summary.json"
    if not summary_path.exists():
        print(f"{run.name}: MISSING ({summary_path})", flush=True)
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print(run.name, {key: summary[key] for key in KEYS if key in summary}, flush=True)
    for family, row in (summary.get("families") or {}).items():
        print(
            f"   {family}: n={row.get('n')} success={row.get('success')} "
            f"safe={row.get('safe')} abstained={row.get('abstained')}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", default=[], help="run folder name under runs/")
    parser.add_argument(
        "--adapter",
        action="append",
        default=[],
        help="LOCAL:SUBFOLDER, e.g. runs/sft3:sft3_adapter",
    )
    parser.add_argument("--no-upload", action="store_true")
    args = parser.parse_args()

    runs = [ROOT / "runs" / name for name in args.run]
    for run in runs:
        report(run)

    token = os.environ.get("HF_TOKEN")
    if args.no_upload or not token:
        print("upload skipped:", "no HF_TOKEN" if not token else "--no-upload", flush=True)
        return

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    for run in runs:
        if not (run / "summary.json").exists():
            continue
        api.upload_folder(
            repo_id=RUNS_REPO, repo_type="dataset", folder_path=str(run), path_in_repo=run.name
        )
        print(f"{run.name} -> {RUNS_REPO}", flush=True)

    for spec in args.adapter:
        local, _, subfolder = spec.partition(":")
        path = ROOT / local
        if not subfolder or not (path / "adapter_config.json").exists():
            print(f"adapter skipped: {spec}", flush=True)
            continue
        api.upload_folder(
            repo_id=MODEL_REPO,
            folder_path=str(path),
            path_in_repo=subfolder,
            commit_message=f"job upload: {subfolder}",
        )
        print(f"{subfolder} -> {MODEL_REPO}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
