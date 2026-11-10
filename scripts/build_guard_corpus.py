"""Freeze the clean-traffic corpus the guard is measured against.

`scripts/guard_eval.py` has to answer one question with saved data: on runs where no injection
was delivered, does the guard ever fire? `runs/` is not tracked (it is re-downloadable from
`primce/lexsi-runs`), so the tool results that answer it are copied here, small and compressed,
with the run name kept so the per-run table still reads.

    python scripts/build_guard_corpus.py

Writes data/guard_traffic.jsonl.gz. Rebuild it only if the source runs change; the hash it
prints is the one quoted in docs/GUARD.md.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.guard_eval import BENIGN_RUNS, DIRTY_RUNS  # noqa: E402

OUT = ROOT / "data" / "guard_traffic.jsonl.gz"


def main() -> None:
    rows = []
    missing = []
    for run in BENIGN_RUNS + DIRTY_RUNS:
        path = ROOT / "runs" / run / "trajectories.jsonl"
        if not path.exists():
            missing.append(run)
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            episode = json.loads(line)
            steps = [
                {"name": step.get("name"), "result": step.get("result")}
                for step in episode.get("steps") or []
                if isinstance(step.get("result"), dict)
            ]
            if steps:
                rows.append({"run": run, "clean": run in BENIGN_RUNS, "steps": steps})
    if missing:
        raise SystemExit(f"missing runs: {', '.join(missing)}")

    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")
    with gzip.GzipFile(OUT, "wb", mtime=0) as handle:
        handle.write(payload)

    episodes = len(rows)
    results = sum(len(row["steps"]) for row in rows)
    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
    print(f"wrote {OUT.relative_to(ROOT)}: {episodes} episodes, {results} tool results")
    print(f"sha256 {digest}")


if __name__ == "__main__":
    main()
