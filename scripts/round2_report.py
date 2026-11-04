"""Round-2 tables, built only from uploaded `summary.json` files.

    python scripts/round2_report.py           # pull primce/lexsi-runs, then print
    python scripts/round2_report.py --local   # use whatever is already in runs/

The output is markdown, meant to be pasted into docs/RESULTS.md unchanged: no
number in the docs is typed by hand. A run that has not landed yet prints as
`-` instead of breaking the table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stats import compare, load_success, wilson

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
REPO = "primce/lexsi-runs"

MAIN = [
    ("base_hard30", "Base - hard-30"),
    ("base_hard90", "Base - hard-90"),
    ("sft2_hard30", "SFT-mix s42 - hard-30"),
    ("sft2_hard90", "SFT-mix s42 - hard-90"),
    ("sft3_hard30", "SFT-mix2 s42 - hard-30"),
    ("sft3_eval", "SFT-mix2 s42 - eval-24"),
    ("sft3_hard90", "SFT-mix2 s42 - hard-90"),
    ("sft3_seed123_hard30", "SFT-mix2 s123 - hard-30"),
    ("sft3_seed123_hard90", "SFT-mix2 s123 - hard-90"),
    ("sft3_seed7_hard30", "SFT-mix2 s7 - hard-30"),
    ("sft3_seed7_hard90", "SFT-mix2 s7 - hard-90"),
    ("grpo3_hard30", "SFT-mix2 + GRPO - hard-30"),
    ("grpo3_hard90", "SFT-mix2 + GRPO - hard-90"),
]

SAFETY = [
    ("safety_base", "Base"),
    ("safety_sft2", "SFT-mix"),
    ("safety_sft3", "SFT-mix2"),
    ("safety_sft3_hardened", "SFT-mix2 + hardened prompt"),
    ("grpo3_safety", "SFT-mix2 + GRPO"),
]

PAIRS = [
    ("base_hard90", "sft3_hard90", "base -> SFT-mix2 on hard-90"),
    ("sft2_hard90", "sft3_hard90", "SFT-mix -> SFT-mix2 on hard-90"),
    ("sft3_hard30", "grpo3_hard30", "GRPO on hard-30"),
    ("sft3_hard90", "grpo3_hard90", "GRPO on hard-90"),
]

SEED_RUNS = {
    "hard-30": ["sft3_hard30", "sft3_seed123_hard30", "sft3_seed7_hard30"],
    "hard-90": ["sft3_hard90", "sft3_seed123_hard90", "sft3_seed7_hard90"],
}

AUDITKIT = [
    ("auditkit_base_hard30", "Base - hard-30"),
    ("auditkit_sft2_hard30", "SFT-mix - hard-30"),
    ("auditkit_sft2_hard90", "SFT-mix - hard-90"),
    ("auditkit_sft3_hard30", "SFT-mix2 - hard-30"),
    ("auditkit_sft3_hard90", "SFT-mix2 - hard-90"),
]


def download() -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        REPO,
        repo_type="dataset",
        local_dir=str(RUNS),
        allow_patterns=["*/summary.json", "*/trajectories.jsonl"],
    )


def summary(name: str) -> dict | None:
    path = RUNS / name / "summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_rate(hits, n) -> str:
    if not n:
        return "-"
    low, high = wilson(int(hits), int(n))
    return f"{100 * hits / n:.2f}% ({hits}/{n}) [{100 * low:.0f}–{100 * high:.0f}%]"


def num(value, digits=2):
    return "-" if value is None else f"{value:.{digits}f}"


def main_table() -> None:
    print("| Model | Split | success (95% CI) | answer_ok | calls | strict_rate | ter |")
    print("|---|---|---:|---:|---:|---:|---:|")
    missing = []
    for name, label in MAIN:
        row = summary(name)
        if row is None:
            missing.append(name)
            print(f"| {label} | `{name}` | - | - | - | - | - |")
            continue
        split = name.split("_")[-1]
        success = row.get("success", 0.0)
        print(
            f"| {label} | {split} | {fmt_rate(round(success * row['n']), row['n'])} "
            f"| {num(row.get('answer_ok'))} | {num(row.get('calls'))} "
            f"| {num(row.get('strict_rate'), 3)} | {num(row.get('ter'), 3)} |"
        )
    if missing:
        print("\nnot uploaded yet: " + ", ".join(f"`{name}`" for name in missing))


def safety_table() -> None:
    print("\n| Model | n | safe | attack_ledger | utility | abstained | asked | invented_expense |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, label in SAFETY:
        row = summary(name)
        if row is None:
            print(f"| {label} | - | - | - | - | - | - | - |")
            continue
        print(
            f"| {label} | {row.get('n')} | {num(row.get('safe'), 3)} "
            f"| {row.get('attack_ledger', '-')} | {num(row.get('utility'), 3)} "
            f"| {row.get('abstained', '-')} | {row.get('asked', '-')} "
            f"| {row.get('invented_expense', '-')} |"
        )


def families(name: str) -> None:
    row = summary(name)
    if not row or not row.get("families"):
        return
    print(f"\n`{name}` by family:")
    print("| family | n | success | safe | abstained |")
    print("|---|---:|---:|---:|---:|")
    for family, values in row["families"].items():
        print(
            f"| {family} | {values.get('n')} | {num(values.get('success'), 3)} "
            f"| {num(values.get('safe'), 3)} | {values.get('abstained', '-')} |"
        )


def paired() -> None:
    print("\n| Comparison | n | a | b | gain | a-only | b-only | McNemar p |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name_a, name_b, label in PAIRS:
        if not (RUNS / name_a).exists() or not (RUNS / name_b).exists():
            print(f"| {label} | - | - | - | - | - | - | - |")
            continue
        row = compare(load_success(RUNS / name_a), load_success(RUNS / name_b))
        print(
            f"| {label} | {row['n']} | {num(row['a_success'], 3)} | {num(row['b_success'], 3)} "
            f"| {row['gain']:+.3f} | {row['a_only']} | {row['b_only']} | {row['mcnemar_p']} |"
        )


def seed_spread() -> None:
    print("\n| Split | seeds | success per seed | spread | mean |")
    print("|---|---|---|---:|---:|")
    for split, names in SEED_RUNS.items():
        rows = [summary(name) for name in names]
        present = [row["success"] for row in rows if row]
        if len(present) < 2:
            print(f"| {split} | {len(present)}/{len(names)} landed | - | - | - |")
            continue
        values = ", ".join(f"{100 * value:.2f}%" for value in present)
        print(
            f"| {split} | {len(present)} | {values} "
            f"| {100 * (max(present) - min(present)):.2f} pp "
            f"| {100 * sum(present) / len(present):.2f}% |"
        )


def auditkit_table() -> None:
    print("\n| Run | n | AuditKIT success | agreement with project scorer "
          "| ledger tool_f1 | ledger validity | full tool_f1_name | loop rate |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, label in AUDITKIT:
        path = RUNS / name / "summary.json"
        if not path.exists():
            print(f"| {label} | - | - | - | - | - | - | - |")
            continue
        row = json.loads(path.read_text(encoding="utf-8"))
        ledger = row.get("ledger_route", {})
        full = row.get("full_route", {})
        print(
            f"| {label} | {row.get('n')} "
            f"| {fmt_rate(round(row.get('success', 0) * row.get('n', 0)), row.get('n'))} "
            f"| {num(row.get('agreement_with_own_scorer'), 4)} "
            f"| {num(ledger.get('tool_call_f1'), 4)} | {num(ledger.get('tool_call_validity'), 4)} "
            f"| {num(full.get('tool_call_f1_name'), 4)} "
            f"| {num(ledger.get('agent_loop_detection'), 4)} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true", help="do not touch the network")
    parser.add_argument("--family", default="sft3_hard90", help="run whose family table prints")
    args = parser.parse_args()

    if not args.local:
        download()
    main_table()
    safety_table()
    paired()
    seed_spread()
    auditkit_table()
    families(args.family)


if __name__ == "__main__":
    main()
