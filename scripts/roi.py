"""Price the same agent workload two ways: self-hosted Tiny Aya vs a closed API.

    python scripts/roi.py
    python scripts/roi.py --gpu-hour 0.35 --tasks-per-month 10000

Reads recorded runs, counts input and output tokens with the base model's own tokenizer over
the stored transcripts, and writes `docs/ROI.md`. The measured side is wall time from
`summary.json`; the closed side is list price on the same token traffic, so the comparison is
"same work, two bills", not two different workloads.

The self-hosted number is deliberately pessimistic: it charges the full eval wall time
(sequential, greedy, one request at a time, model load included) at an on-demand T4 price.
A batched or ZeroGPU deployment is cheaper per task than this table says.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_RUNS = ("sft4_hard90", "base_hard90", "sft4_eval")

# ---------------------------------------------------------------- closed-model list prices, USD per 1M tokens
# Checked 2026-11-09. Edit this table to re-price; nothing else depends on it.
CLOSED = (
    ("GPT-6 Luna (cheapest frontier tier)", 0.10, 0.50),
    ("GPT-4o-mini", 0.15, 0.60),
    ("GPT-6.1 Sol (mid tier)", 2.00, 10.00),
)
PRICE_NOTE = (
    "Closed-model prices are published list prices checked on 2026-11-09 "
    "(openai.com/api/pricing; Nov-2026 LLM price comparison); they change, so treat them as "
    "a snapshot and edit `CLOSED` in `scripts/roi.py` to re-price."
)

MODEL_ID = "CohereLabs/tiny-aya-global"


def count_tokens(path: Path, tokenizer) -> tuple[int, int, int]:
    """(input tokens, generated tokens, tasks) over every assistant turn in the transcripts.

    The prompt for each turn is the chat template applied to the conversation before it, which is
    exactly what the runner fed the model; the completion is the stored assistant message.
    """
    inputs = outputs = tasks = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            conversation = (row.get("metadata") or {}).get("conversation") or []
            tasks += 1
            for index, message in enumerate(conversation):
                if message.get("role") != "assistant":
                    continue
                prompt = tokenizer.apply_chat_template(
                    conversation[:index],
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=False,
                )
                inputs += len(prompt)
                outputs += len(tokenizer(message.get("content") or "")["input_ids"])
    return inputs, outputs, tasks


def money(value: float) -> str:
    if value < 0.01:
        return f"${value:.4f}"
    if value < 100:
        return f"${value:.2f}"
    return f"${value:,.0f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", default=list(DEFAULT_RUNS))
    parser.add_argument("--gpu-hour", type=float, default=0.53, help="USD per T4 GPU-hour")
    parser.add_argument("--tasks-per-month", type=int, default=10000)
    parser.add_argument("--out", default="docs/roi.md")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    rows = []
    for name in args.runs:
        run = ROOT / "runs" / name
        summary_path = run / "summary.json"
        traj_path = run / "trajectories.jsonl"
        if not summary_path.exists() or not traj_path.exists():
            print("skip (missing run):", name)
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        inputs, outputs, tasks = count_tokens(traj_path, tokenizer)
        seconds = float(summary.get("seconds") or 0.0)
        rows.append(
            {
                "name": name,
                "n": tasks,
                "calls": float(summary.get("calls") or 0.0),
                "success": float(summary.get("success") or 0.0),
                "seconds": seconds,
                "in_per_task": inputs / tasks,
                "out_per_task": outputs / tasks,
                "in_total": inputs,
                "out_total": outputs,
                "tok_s": (inputs + outputs) / seconds if seconds else 0.0,
            }
        )

    if not rows:
        raise SystemExit("no runs found; nothing to price")

    gpu = args.gpu_hour
    per_month = args.tasks_per_month

    lines = [
        "# ROI — self-hosted Tiny Aya vs a closed API on the same workload",
        "",
        "The judged question is not \"is a 3.35B model cheaper\", it is \"what does this workload "
        "cost per task, before and after fine-tuning, each way\". Both sides below run the *same* "
        "recorded traffic: token counts are measured from the stored transcripts with the base "
        "model's own tokenizer, and the self-hosted side uses the wall time each eval actually "
        "took on a single T4.",
        "",
        "## Measured workload (single T4, greedy, one request at a time)",
        "",
        "| Run | Tasks | Success | Calls/task | Input tok/task | Output tok/task | Wall s | Tok/s end-to-end |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['name']}` | {row['n']} | {row['success']:.1%} | {row['calls']:.2f} | "
            f"{row['in_per_task']:,.0f} | {row['out_per_task']:,.0f} | {row['seconds']:,.0f} | "
            f"{row['tok_s']:.1f} |"
        )

    lines += ["", f"## Cost per 1,000 tasks (T4 at ${gpu:.2f}/GPU-hour)", ""]
    header = "| Run | Self-host GPU-hours | Self-host $ | " + " | ".join(
        f"{label} $" for label, _, _ in CLOSED
    ) + " |"
    lines += [header, "|---|---:|---:|" + "---:|" * len(CLOSED)]
    for row in rows:
        hours = 1000 * row["seconds"] / row["n"] / 3600
        cells = [f"{hours:.2f} h", money(hours * gpu)]
        for _, price_in, price_out in CLOSED:
            cells.append(
                money(
                    1000
                    * (row["in_per_task"] * price_in + row["out_per_task"] * price_out)
                    / 1e6
                )
            )
        lines.append(f"| `{row['name']}` | {cells[0]} | " + " | ".join(cells[1:]) + " |")

    lines += ["", f"## At {per_month:,} tasks/month", ""]
    lines.append("| Run | Self-host GPU-hours/month | Self-host $/month | " + " | ".join(
        f"{label} $/month" for label, _, _ in CLOSED
    ) + " |")
    lines.append("|---|---:|---:|" + "---:|" * len(CLOSED))
    for row in rows:
        hours = per_month * row["seconds"] / row["n"] / 3600
        cells = [f"{hours:.2f} h", money(hours * gpu)]
        for _, price_in, price_out in CLOSED:
            cells.append(
                money(
                    per_month
                    * (row["in_per_task"] * price_in + row["out_per_task"] * price_out)
                    / 1e6
                )
            )
        lines.append(f"| `{row['name']}` | {cells[0]} | " + " | ".join(cells[1:]) + " |")

    tuned = next((row for row in rows if row["name"].startswith("sft4_hard")), None)
    base = next((row for row in rows if row["name"].startswith("base_hard")), None)
    tuned_spt = tuned["seconds"] / tuned["n"] if tuned else 0.0
    cheapest = CLOSED[0]

    lines += ["", "## Cost per 1,000 solved tasks", ""]
    lines.append("| Run | Solved / 1,000 | Self-host $ | " + " | ".join(
        f"{label} $" for label, _, _ in CLOSED
    ) + " |")
    lines.append("|---|---:|---:|" + "---:|" * len(CLOSED))
    for row in rows:
        success = row["success"] or 1.0
        cells = [money(1000 * row["seconds"] / row["n"] / 3600 * gpu / success)]
        for _, price_in, price_out in CLOSED:
            cells.append(
                money(
                    1000
                    * (row["in_per_task"] * price_in + row["out_per_task"] * price_out)
                    / 1e6
                    / success
                )
            )
        lines.append(
            f"| `{row['name']}` | {1000 * success:,.0f} | " + " | ".join(cells) + " |"
        )

    lines += [
        "",
        "## What the table says",
        "",
    ]
    if tuned and base:
        def closed_per_1k(row, prices=cheapest):
            return 1000 * (
                row["in_per_task"] * prices[1] + row["out_per_task"] * prices[2]
            ) / 1e6

        def gpu_per_1k(row):
            return 1000 * row["seconds"] / row["n"] / 3600 * gpu

        tuned_closed, base_closed = closed_per_1k(tuned), closed_per_1k(base)
        tuned_gpu, base_gpu = gpu_per_1k(tuned), gpu_per_1k(base)
        ratio = (base_closed / base["success"]) / (tuned_closed / tuned["success"])
        lines += [
            f"- **Cost per solved task is the number that matters.** Untuned, the base solves "
            f"{base['success']:.1%} of these questions, so its bill buys a correct ledger only "
            f"every {1 / base['success']:.0f}th request: {money(base_closed / base['success'])} "
            f"per 1,000 solved on the cheapest closed tier, against "
            f"{money(tuned_closed / tuned['success'])} for the adapter at "
            f"{tuned['success']:.1%} — {ratio:.0f}x less. Self-hosted the same comparison is "
            f"{money(gpu_per_1k(base) / base['success'])} against "
            f"{money(gpu_per_1k(tuned) / tuned['success'])}. The gap is the accuracy, not the "
            "price list.",
            f"- The token saving is the mechanism: {tuned['out_per_task']:,.0f} generated tokens "
            f"per task against {base['out_per_task']:,.0f} untuned, and "
            f"{tuned['in_per_task']:,.0f} billed input tokens against "
            f"{base['in_per_task']:,.0f}, because the adapter finishes in "
            f"{tuned['calls']:.2f} calls instead of {base['calls']:.2f} and stops when the ledger "
            "is settled instead of narrating a wrong split.",
            f"- Self-hosted on one T4 at ${gpu:.2f}/h the same 1,000 tuned tasks cost "
            f"{money(tuned_gpu)} ({tuned_gpu / gpu:.2f} GPU-hours), against {money(base_gpu)} "
            f"({base_gpu / gpu:.2f} hours) untuned. At this volume a per-token closed API is "
            f"{tuned_gpu / tuned_closed:.0f}x cheaper per request than a dedicated card, so the "
            "case for self-hosting is data locality, offline operation and batching rather than "
            "the sticker price: this eval served one request at a time, and concurrent requests "
            "divide the GPU line while the token bill stays linear.",
            "- The closed column prices the *tokens*, not a verified success rate: no closed "
            "model was run on these 90 Hinglish ledger tasks. If a closed model solved them at "
            f"the base model's {base['success']:.1%}, its bill per solved task would land near "
            f"{money(base_closed / base['success'])} instead of the per-request figure.",
        ]
    else:
        lines.append("- Run `sft4_hard90` and `base_hard90` together to get the before/after line.")

    lines += [
        "",
        "## Caveats a reader should apply",
        "",
        "- The self-hosted side is charged at sequential, greedy decoding with model load inside "
        "the eval wall time. Prompt caching, a warm process, batching or ZeroGPU all lower it; "
        "none of that is assumed here.",
        "- Self-hosting also costs engineering time, monitoring, cold starts and an eval harness. "
        "None of that is in the GPU line — it is excluded, not zero.",
        "- These latencies are fp16 on one T4 (`--no-4bit`). A 4-bit server, which the demo Space "
        "uses, trades some speed for roughly a quarter of the memory; the earlier 4-bit round-1 "
        "measurement was 80.95 s/request untuned and 54.95 s tuned against "
        f"{tuned_spt:.1f} s/task here.",
        "- This prices the merged model that was evaluated, not LoRA-adapter hosting (cheaper to "
        "serve, but it adds a PEFT dependency).",
        "- Tiny Aya is not a frontier assistant: this is a measured cost and reliability trade-off "
        "on one closed task family, not a quality-parity claim.",
        "",
        "## Assumptions and how to re-check",
        "",
        "- GPU price is an on-demand T4 list price and is the only free parameter: "
        f"`--gpu-hour` (here ${gpu:.2f}; Kaggle and HF ZeroGPU are $0 for this workload).",
        f"- {PRICE_NOTE}",
        "- Token counts come from `runs/<run>/trajectories.jsonl` with "
        f"`{MODEL_ID}`'s tokenizer, counting every assistant turn's prompt plus its generated "
        "message.",
        "",
        "```bash",
        "python scripts/roi.py --gpu-hour 0.53 --tasks-per-month 10000",
        "```",
        "",
    ]

    out = ROOT / args.out
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
