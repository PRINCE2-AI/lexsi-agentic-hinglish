"""Render the safety-split comparison table from saved runs.

    python scripts/safety_table.py
    python scripts/safety_table.py --run "Base=runs/safety_base" --run "SFT=/tmp/safety_sft2"

Reads `summary.json` from each run directory and prints markdown for the article and
docs/SAFETY.md. Missing runs print as "-" so a half-finished night still renders.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_RUNS = (
    ("Base Tiny Aya", "runs/safety_base"),
    ("SFT-mix (sft2)", "runs/safety_sft2"),
    ("SFT-mix + hardened prompt", "runs/safety_sft2_hardened"),
)

COLUMNS = (
    ("safe", "safe", "pct"),
    ("utility", "utility", "pct"),
    ("attack_ledger", "attack_ledger", "pct"),
    ("attack_answer", "attack_answer", "pct"),
    ("unplanned_calls", "unplanned_calls", "num"),
    ("abstained", "abstained", "pct"),
)


def load(path: Path) -> dict | None:
    summary = path / "summary.json"
    if not summary.exists():
        return None
    return json.loads(summary.read_text(encoding="utf-8"))


def cell(summary: dict | None, key: str, kind: str) -> str:
    if summary is None or key not in summary:
        return "-"
    value = float(summary[key])
    return f"{value:.2%}" if kind == "pct" else f"{value:.2f}"


def family_rows(summaries: list[tuple[str, dict | None]]) -> list[str]:
    names: list[str] = []
    for _, summary in summaries:
        for name in (summary or {}).get("families", {}):
            if name not in names:
                names.append(name)
    if not names:
        return []
    header = "| Family | " + " | ".join(label for label, _ in summaries) + " |"
    lines = [header, "|---|" + "---:|" * len(summaries)]
    for name in names:
        cells = []
        for _, summary in summaries:
            family = (summary or {}).get("families", {}).get(name)
            cells.append(cell(family, "safe", "pct"))
        lines.append(f"| `{name}` | " + " | ".join(cells) + " |")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help='LABEL=path, repeatable; defaults to the three safety runs',
    )
    args = parser.parse_args()

    pairs = []
    for item in args.run:
        label, _, path = item.partition("=")
        pairs.append((label, path))
    if not pairs:
        pairs = list(DEFAULT_RUNS)

    summaries = [(label, load(ROOT / path)) for label, path in pairs]

    header = "| System | " + " | ".join(label for _, label, _ in COLUMNS) + " |"
    lines = [header, "|---|" + "---:|" * len(COLUMNS)]
    for label, summary in summaries:
        values = [cell(summary, key, kind) for key, _, kind in COLUMNS]
        lines.append(f"| {label} | " + " | ".join(values) + " |")
    print("\n".join(lines))

    families = family_rows(summaries)
    if families:
        print()
        print("### safe rate by family")
        print()
        print("\n".join(families))

    missing = [label for label, summary in summaries if summary is None]
    if missing:
        print()
        print("missing runs:", ", ".join(missing))


if __name__ == "__main__":
    main()
