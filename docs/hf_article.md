# Teaching Tiny Aya to split Hinglish expenses without losing the ledger

*Lexsi AI Safety & Alignment Hackathon 2026 — Track 3, Agentic fine-tuning.*

## TL;DR

- Base Tiny Aya Global on the 20-task starter set: **25%** success, **70%** loop collapse,
  **9.25** tool calls per task against an oracle that needs 3.5.
- SFT on 100 easy oracle trajectories: **63.33%** on the held-out hard-30, **100%** on eval-24.
- Adding GRPO on top (30 steps, graded reward): **63.33%** — no change. The reward was
  saturated and the run was too short to move an answer-shape failure.
- SFT on the reweighted mix (184 rows): **96.67%** on hard-30, **100%** on eval-24, **3.57**
  calls per task — exactly the oracle's call count.
- Adapter, dataset, run logs and Space are linked at the end.

## Problem statement

Track 3, primary: **Reliable tool calling** — train an agent to choose the right tool, pass
well-formed arguments, and call tools in the right order. Secondary: **Efficient
trajectories** — cut redundant, looping or failed calls without lowering success.

Every clause is scored mechanically, with no LLM judge: tool choice (`tac`), argument
validity (`ter`/`arr`), ordering (final ledger correctness), redundancy (`rad`), looping
(`lcf`), and calls/steps per task.

## Why this task

Cohere Labs chat templates do not render `tools`, so the model has to read a tool schema
from the system prompt and emit bare JSON calls. It also has to carry a JSON ledger across
turns, do currency conversion, split expenses, and answer in Hinglish. A 3.35B multilingual
model can do parts of this; making all of it reliable is the interesting problem.

Success is a deterministic ledger state, not text similarity, so every number below can be
reproduced by re-running the dataset.

## Data

| Split | Rows | Purpose |
|---|---:|---|
| easy train | 100 | single/multi expense, balance, settle, convert |
| easy eval | 24 | in-distribution held-out |
| hard train | 40 | six stress families |
| hard eval | 30 | held-out instances of the same six families (sha256-pinned) |
| mix | 184 | easy 100 + hard 84 (hard rows repeated per family) |

The six hard families:

| Family | What it stresses | Hard-eval rows |
|---|---|---:|
| `settle5` | five people over three days | 6 |
| `two_query` | two questions in one Hinglish turn | 6 |
| `convert_sum` | two conversions then a sum | 3 |
| `empty_balance` | empty ledger, must not invent an expense | 3 |
| `trap_combo` | payer also listed among participants | 6 |
| `paraphrase` | Devanagari-mixed wording | 6 |

The eval file is hash-pinned (`37f52b47…`), and the mix builder refuses to run if a
hard-train row appears in the hard-eval set, so repetition cannot leak eval rows.

## Method

1. Five stateless tools: `add_expense`, `convert_currency`, `get_balance`, `settle_up`,
   `finish`; state travels in the ledger object, so rollouts cannot leak into each other.
2. SFT on oracle trajectories: LoRA r=16 on all linear layers, fp16, lr 2e-4, labels masked
   to assistant turns.
3. GRPO with AgentTune `create_agentic_trainer`, graded reward (format + required tools +
   argument names/values + answer number) plus a no-duplicate-calls term.
4. Evaluation on Kaggle T4 x2, greedy decoding, `--no-4bit`, max 8 steps per task.

## Results

| Model | hard-30 success | eval-24 success | answer_ok (hard) | calls (hard) |
|---|---:|---:|---:|---:|
| SFT-easy (100 rows, 3 epochs) | 63.33% (19/30) | 100% (24/24) | 66.67% | 3.47 |
| SFT-easy + GRPO (30 steps) | 63.33% (19/30) | 100% (24/24) | 66.67% | 3.47 |
| **SFT-mix (184 rows, 2 epochs)** | **96.67% (29/30)** | **100% (24/24)** | **96.67%** | **3.57** |
| oracle | 100% (30/30) | 100% (24/24) | 100% | 3.57 |

Wilson 95% intervals: 63.33% → 45.5–78.1%; 96.67% → 83.3–99.4%; eval-24 100% → 86.2–100%.
The paired exact McNemar on the uploaded trajectories is **10 tasks fixed, 0 broken,
p = 0.00195** — no hard task regressed.

Per family:

| Family | n | SFT-easy + GRPO | SFT-mix |
|---|---:|---:|---:|
| two_query | 6 | 0/6 | 6/6 |
| convert_sum | 3 | 0/3 | 2/3 |
| empty_balance | 3 | 2/3 | 3/3 |
| trap_combo | 6 | 5/6 | 6/6 |
| settle5 | 6 | 6/6 | 6/6 |
| paraphrase | 6 | 6/6 | 6/6 |

Zero tool errors, zero redundant calls, zero unnecessary calls (`overcall` 0.00) and zero
loop collapse in the frozen run. It matches the oracle on calls per task, which is why the
secondary statement ("efficient trajectories") is reported on the same row as success.
The single remaining miss is `h-e-027`, a two-currency sum.

## What did not work

GRPO on top of SFT-easy moved nothing: 30 steps is about a third of an epoch of the mix,
and with the graded reward already near ceiling both generations in a group scored alike,
so the advantage was ~0. The 4-hour run reproduced 63.33% exactly.

The failures it needed to fix were not tool-choice failures. In `two_query` the model
called the right tools (4 calls) but reported only one of the two results; in `convert_sum`
it converted once and invented the second value; in `empty_balance` it occasionally added
an expense that was never mentioned. `answer_ok` was 66.67% while `tac` was 96.67% — the
model knew what to do and dropped it when writing the answer.

Reweighting the same 40 hard rows into an SFT mix fixed that in 92 steps (+33.3 points).
The bottleneck was coverage of the answer shape, not the RL stage.

## Limitations

- hard-30 has 30 tasks; counts are reported next to percentages, and the CIs above are wide.
- hard-30 rows are held-out *instances* of the same six templates, so this measures
  generalization across amounts/names/languages, not across task formats.
- eval-24 is in-distribution with easy train, which is why it stays at 100%.
- Fixed offline FX rates, fixed six-person name universe, no free-form tools.
- One assistant turn in the frozen run was not bare JSON (`strict_rate` 0.983); task-level
  `strict` is in the uploaded summary.

## Reproduce

```bash
python data/build_hard.py
python data/build_mix.py
python scripts/train_sft.py --tasks data/tasks_train_mix.jsonl \
    --epochs 2 --out runs/sft2 --merge-out runs/sft2_merged
python scripts/run_baseline.py --model runs/sft2_merged \
    --tasks data/tasks_hard_eval.jsonl --no-4bit --out runs/sft2_hard30
python scripts/run_baseline.py --model runs/sft2_merged \
    --tasks data/tasks_eval.jsonl --no-4bit --out runs/sft2_eval
python scripts/stats.py --a runs/sft_hard30 --b runs/sft2_hard30
```

## Artifacts

- GitHub: https://github.com/PRINCE2-AI/lexsi-agentic-hinglish
- Adapter: https://huggingface.co/primce/tiny-aya-hinglish-lora (`sft2_adapter`)
- Run logs: https://huggingface.co/datasets/primce/lexsi-runs
- Space: [tuned demo link]

## Credits

Built on AgentTune and AuditKIT by Lexsi Labs, with Cohere Labs Tiny Aya and Hugging Face
hosting.
