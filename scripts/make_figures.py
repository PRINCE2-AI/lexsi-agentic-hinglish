"""Draw the three figures the write-up needs, from the run logs.

    python scripts/make_figures.py        # writes docs/figures/*.png

The numbers here are recomputed from `runs/*/results.jsonl` and `runs/*/summary.json`; the safety
panel uses the corrected ledger counts printed in `docs/SAFETY.md` (the legacy sentinel undercounts
a shared 999999 expense, see the correction in that file), with the base row on the 136 rows where
it was measured.
"""

from __future__ import annotations

import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "figures")

INK = "#1b1b1f"
MUTED = "#8a8a93"
BAD = "#c0392b"
GOOD = "#1e7a46"
ACCENT = "#2f6fb5"


def family_map() -> dict[str, str]:
    out: dict[str, str] = {}
    for path in glob.glob(os.path.join(ROOT, "data", "tasks*.jsonl")):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                if "id" in row and "family" in row:
                    out[row["id"]] = row["family"]
    return out


def rows(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def style(ax, title: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=13, color=INK, pad=10)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d8d8dd")
    ax.grid(axis="y", color="#ececf0", linewidth=0.8)
    ax.set_axisbelow(True)


def fig_before_after() -> str:
    fam = family_map()
    base = rows(os.path.join(ROOT, "runs", "base_hard90", "results.jsonl"))
    sft4 = rows(os.path.join(ROOT, "runs", "sft4_hard90", "results.jsonl"))

    def by_family(rs):
        acc: dict[str, list[bool]] = {}
        for r in rs:
            key = fam.get(r["task_id"], r["task_id"].split("-")[0])
            acc.setdefault(key, []).append(bool(r["success"]))
        return {k: sum(v) / len(v) for k, v in acc.items()}

    b, s = by_family(base), by_family(sft4)
    order = sorted(b, key=lambda k: s[k])
    idx = range(len(order))
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=150)
    ax.bar([i - 0.2 for i in idx], [b[k] * 100 for k in order], width=0.38, color=MUTED, label="base Tiny Aya (prompt only)")
    ax.bar([i + 0.2 for i in idx], [s[k] * 100 for k in order], width=0.38, color=GOOD, label="shipped adapter (sft4)")
    for i, k in enumerate(order):
        ax.text(i + 0.2, s[k] * 100 + 2, f"{round(s[k] * 15)}/15", ha="center", fontsize=8, color=GOOD)
        ax.text(i - 0.2, b[k] * 100 + 2, f"{round(b[k] * 15)}/15", ha="center", fontsize=8, color=MUTED)
    ax.set_xticks(list(idx), order, fontsize=9)
    ax.set_ylim(0, 118)
    style(ax, "Hard-90 holdout, final ledger state, no LLM judge", "task success (%)")
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0.0, 1.0))
    path = os.path.join(OUT, "hard90_by_family.png")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_safety() -> str:
    labels = ["base Tiny Aya\n(136 rows)", "shipped adapter\n(208 rows)", "adapter + guard\n(208 rows)"]
    harmed = [85, 13, 0]
    totals = [136, 208, 208]
    colors = [MUTED, ACCENT, GOOD]
    fig, ax = plt.subplots(figsize=(7.4, 4.0), dpi=150)
    bars = ax.bar(labels, [h / t * 100 for h, t in zip(harmed, totals)], color=colors, width=0.55)
    for bar, h, t in zip(bars, harmed, totals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2, f"{h} of {t}",
                ha="center", fontsize=10, color=INK)
    ax.set_ylim(0, 78)
    style(ax, "Injected episodes that move the attacker's money into the ledger", "share of injected rows (%)")
    path = os.path.join(OUT, "safety_pool.png")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def summary(run: str) -> dict:
    with open(os.path.join(ROOT, "runs", run, "summary.json"), encoding="utf-8") as fh:
        return json.load(fh)


def fig_rounds() -> str:
    series = [
        ("base", "base_hard30", "base_hard90"),
        ("sft\n(round 0)", "sft_hard30", None),
        ("gpso\n(round 0)", "grpo_hard30", None),
        ("sft2\n(round 1)", "sft2_hard30", "sft2_hard90"),
        ("sft3\n(round 2)", "sft3_hard30", "sft3_hard90"),
        ("grpo3\n(round 2)", "grpo3_hard30", "grpo3_hard90"),
        ("sft4\n(round 3, ships)", "sft4_hard30", "sft4_hard90"),
        ("sft5\n(32 ask rows)", "sft5_hard30", "sft5_hard90"),
    ]
    labels, h30, h90 = [], [], []
    for name, a, b in series:
        try:
            labels.append(name.replace("gpso", "GRPO"))
            h30.append(summary(a)["success"] * 100)
            h90.append(summary(b)["success"] * 100 if b and os.path.exists(os.path.join(ROOT, "runs", b, "summary.json")) else float("nan"))
        except FileNotFoundError:
            continue
    idx = range(len(labels))
    fig, ax = plt.subplots(figsize=(9.4, 4.2), dpi=150)
    ax.plot(idx, h30, marker="o", color=ACCENT, linewidth=2, label="hard-30 (n=30)")
    ax.plot(idx, h90, marker="s", color=GOOD, linewidth=2, label="hard-90 (n=90)")
    ax.set_xticks(list(idx), labels, fontsize=8)
    ax.set_ylim(0, 108)
    style(ax, "Every training round, same frozen splits", "task success (%)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    path = os.path.join(OUT, "rounds.png")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    for fn in (fig_before_after, fig_safety, fig_rounds):
        print("wrote", os.path.relpath(fn(), ROOT))


if __name__ == "__main__":
    main()
