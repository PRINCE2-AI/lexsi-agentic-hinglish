"""Copy the run logs behind every published number out of the Hub into `runs/`.

`runs/` is gitignored, so a fresh clone can run `verify_all.py` (pins, oracle, tests) but
cannot recompute a rate, because the per-task rows live in `primce/lexsi-runs`. This is the
step that closes that gap:

    python scripts/fetch_runs.py                    # all 68 runs, ~20 MB
    python scripts/fetch_runs.py --only sft4_hard90 safety_sft4
    python scripts/bootstrap.py                      # rewrites docs/STATS.md from them

The dataset is public, so no token is needed. `--only` exists for a laptop on a small
connection; without it everything is fetched, which is what the reproduce notebook does.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import list_repo_files, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = "primce/lexsi-runs"


def folders(repo: str, only: list[str] | None) -> list[str]:
    files = list_repo_files(repo, repo_type="dataset")
    top = sorted({f.split("/")[0] for f in files if "/" in f})
    if not only:
        return top
    missing = [name for name in only if name not in top]
    if missing:
        raise SystemExit(f"not in {repo}: {', '.join(missing)}")
    return sorted(only)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--only", nargs="*", default=None, help="run folder names")
    args = parser.parse_args()

    wanted = folders(args.repo, args.only)
    dest = ROOT / "runs"
    dest.mkdir(exist_ok=True)
    print(f"fetching {len(wanted)} run folders from {args.repo} into {dest}")

    path = snapshot_download(
        repo_id=args.repo,
        repo_type="dataset",
        local_dir=str(dest),
        allow_patterns=[f"{name}/**" for name in wanted],
    )
    got = [name for name in wanted if (Path(path) / name).is_dir()]
    size = sum(f.stat().st_size for name in got for f in (Path(path) / name).rglob("*") if f.is_file())
    print(f"{len(got)} folders, {size / 1e6:.1f} MB")
    absent = [name for name in wanted if name not in got]
    if absent:
        sys.exit(f"missing after download: {', '.join(absent)}")
    print("next: python scripts/bootstrap.py   # rewrites docs/STATS.md from these rows")


if __name__ == "__main__":
    main()