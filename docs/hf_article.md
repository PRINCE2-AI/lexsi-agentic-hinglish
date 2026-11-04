# Teaching Tiny Aya to split Hinglish expenses without losing the ledger

*Lexsi AI Safety & Alignment Hackathon 2026 — Track 3, Agentic fine-tuning.*

## TL;DR

- Base Tiny Aya on the frozen hard-30, no adapter: **3.33%** (1/30), and that one success is
  accidental. 16 of 30 episodes never finish and it averages **7.03** calls. On the 20-task
  starter set it is **25%** with **70%** loop collapse and **9.25** calls.
- SFT on 100 easy oracle trajectories: **63.33%** on the held-out hard-30, **100%** on eval-24.
- Adding GRPO on top (30 steps, graded reward): **63.33%** — no change. The reward was
  saturated and the run was too short to move an answer-shape failure.
- SFT on the reweighted mix (184 rows): **96.67%** on hard-30, **100%** on eval-24, **3.57**
  calls per task — exactly the oracle's call count.
- Indirect injection through tool results: base writes attacker money into the ledger in
  **13/24** tasks, the frozen adapter in **0/24** (p = 0.00024). Abstention on
  underspecified requests stays at **1/8**.
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
| Base Tiny Aya, prompt-only | 3.33% (1/30) | - | 6.67% | 7.03 |
| SFT-easy (100 rows, 3 epochs) | 63.33% (19/30) | 100% (24/24) | 66.67% | 3.47 |
| SFT-easy + GRPO (30 steps) | 63.33% (19/30) | 100% (24/24) | 66.67% | 3.47 |
| **SFT-mix (184 rows, 2 epochs)** | **96.67% (29/30)** | **100% (24/24)** | **96.67%** | **3.57** |
| oracle | 100% (30/30) | 100% (24/24) | 100% | 3.57 |

Wilson 95% intervals: 3.33% → 0.6–16.7%; 63.33% → 45.5–78.1%; 96.67% → 83.3–99.4%;
eval-24 100% → 86.2–100%.

Two paired exact McNemar tests on the uploaded trajectories, same 30 task ids:

- base → SFT-mix: **28 fixed, 0 broken, p = 7.5e-9**.
- easy-SFT + GRPO → SFT-mix: **10 fixed, 0 broken, p = 0.00195**.

No hard task regressed in either comparison. The base model's one success is a lucky one:
on `h-e-029` it answered "0 INR" correctly, but only after inventing an 800 INR expense
nobody asked for. The answer was right, the behaviour was not — which is why the safety
split in this repo scores the ledger and not just the text.

Per family:

| Family | n | Base | SFT-easy + GRPO | SFT-mix |
|---|---:|---:|---:|---:|
| two_query | 6 | 0/6 | 0/6 | 6/6 |
| convert_sum | 3 | 0/3 | 0/3 | 2/3 |
| empty_balance | 3 | 1/3 | 2/3 | 3/3 |
| trap_combo | 6 | 0/6 | 5/6 | 6/6 |
| settle5 | 6 | 0/6 | 6/6 | 6/6 |
| paraphrase | 6 | 0/6 | 6/6 | 6/6 |

The base model, on the same rows: tool error rate **22.3%**, `overcall` **1.08** (more than one
extra call per expected call), `strict_rate` **0.534**, and 16 episodes that never reached
`finish`. The frozen run has zero tool errors, zero redundant calls, zero unnecessary calls
(`overcall` 0.00) and zero loop collapse. It matches the oracle on calls per task, which is why the
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

## Safety: what happens when a tool result lies

Every Track 3 entry reports task success. This repo also asks the question that matters for
an agent that moves money: what does it do when the tool output carries attacker text? The
method is AgentDojo's indirect injection (arXiv:2406.13352) applied to one deterministic
ledger. A tool result gains a `note` asking for an absurd **999999 INR** expense, or asking
the agent to skip a tool and answer 999999. Because 999999 can never be a correct amount,
"did the attack land" is an equality check on the final ledger and the final answer — no
LLM judge.

32 tasks: `inject_note` 8, `inject_settle` 8, `inject_skip` 8, and `underspecified` 8
(the user omits names/amount/currency; asking is the correct behaviour). Results on the 24
injection tasks:

| System | safe | utility under attack | attack_ledger | attack_answer |
|---|---:|---:|---:|---:|
| Base Tiny Aya | 0/24 | 0/24 | **13/24 (54.2%)** | 8/24 (33.3%) |
| SFT-mix (`sft2`) | 11/24 (45.8%) | 11/24 | **0/24** | 3/24 (12.5%) |
| SFT-mix + hardened prompt | 15/24 (62.5%) | 15/24 | **0/24** | 2/24 (8.3%) |

Paired exact McNemar: base → `sft2` removes **13 ledger attacks, introduces 0,
p = 0.00024**. The prompt-level defence adds 4 safe tasks in one direction
(p = 0.125 — directional, not significant at n = 24).

Three honest readings:

- **Fine-tuning fixed the money-moving failure, not injection as a whole.** Zero written
  attacks, but the tuned model still repeats 999999 in prose on 3 of 24 tasks, and it solves
  11/24 under attack versus 29/30 on the clean hard-30. An injection that does not move
  money still costs tasks; without a no-payload control on these exact 24, treat that gap
  as a warning sign rather than a measured tax.
- **Abstention did not improve.** On the 8 underspecified tasks the tuned model invents a
  split 7 times and abstains once — the same as base on invention. It was trained on fully
  specified tasks, so it learned to complete a ledger, not to ask for missing fields. The
  hardened prompt changes nothing here.
- **The defence is cheap and measurable, not proven.** A two-rule system prompt
  (`HARDENED_RULES`) is the whole mitigation. It never regressed a task and never let money
  move, but a 4-task gain on 24 items is not statistical evidence. The architectural fix —
  treating tool output as data by construction — is unmeasured.


## Why these choices

- **SFT memorizes, RL generalizes** (Chu et al., ICML 2025, arXiv:2501.17161): the frozen
  hard-30 split is the unseen-variant test. SFT scores 100% on the template-matched eval-24
  and 63.33% on hard-30 — the paper's SFT drop, reproduced on Hinglish tool use.
- **ToolRL** (Qian et al., arXiv:2504.13958): reward is decomposed into format, tool-name,
  parameter-name, parameter-value and answer terms instead of a binary hit, and there is no
  length reward. The `empty_balance` family is ToolRL's relevance/irrelevance case: the
  correct action is to add nothing.
- **APIGen** (arXiv:2406.18518) and **ToolACE** (arXiv:2409.00920): every task is
  oracle-verified before use, the eval file is SHA-256 pinned, and the hard split is
  organised into complexity families so failures can be attributed.
- **τ-bench** (arXiv:2406.12045): success is the final ledger state, not string similarity.
- **DeepSeekMath** (arXiv:2402.03300) for GRPO, **Tiny Aya** (arXiv:2603.11510) for the model.

What we did **not** run, so no claim is made about it: cold-start GRPO without SFT (the
control that would test whether SFT is necessary here), a Hammer-style schema-shift eval,
verification-iteration repair beyond the parse-error correction prompt, a no-payload control
on the 24 injection tasks, and an adaptive attacker that can see the eval.

## Limitations

- hard-30 has 30 tasks; counts are reported next to percentages, and the CIs above are wide.
- hard-30 rows are held-out *instances* of the same six templates, so this measures
  generalization across amounts/names/languages, not across task formats.
- eval-24 is in-distribution with easy train, which is why it stays at 100%.
- Fixed offline FX rates, fixed six-person name universe, no free-form tools.
- One assistant turn in the frozen run was not bare JSON (`strict_rate` 0.983); task-level
  `strict` is in the uploaded summary.
- The injection split is 32 tasks, the payloads are English-leaning, and the defence is one
  prompt variant. The prompt gain (4 tasks, p = 0.125) is directional, not significant.

## Reproduce

```bash
python data/build_hard.py
python data/build_mix.py
python scripts/run_baseline.py --model CohereLabs/tiny-aya-global \
    --tasks data/tasks_hard_eval.jsonl --no-4bit --out runs/base_hard30
python scripts/train_sft.py --tasks data/tasks_train_mix.jsonl \
    --epochs 2 --out runs/sft2 --merge-out runs/sft2_merged
python scripts/run_baseline.py --model runs/sft2_merged \
    --tasks data/tasks_hard_eval.jsonl --no-4bit --out runs/sft2_hard30
python scripts/run_baseline.py --model runs/sft2_merged \
    --tasks data/tasks_eval.jsonl --no-4bit --out runs/sft2_eval
python scripts/stats.py --a runs/base_hard30 --b runs/sft2_hard30

python data/build_safety.py
python scripts/run_baseline.py --model CohereLabs/tiny-aya-global \
    --tasks data/tasks_safety_eval.jsonl --safety --no-4bit --out runs/safety_base
python scripts/run_baseline.py --model runs/sft2_merged \
    --tasks data/tasks_safety_eval.jsonl --safety --no-4bit --out runs/safety_sft2
python scripts/run_baseline.py --model runs/sft2_merged \
    --tasks data/tasks_safety_eval.jsonl --safety --hardened-prompt --no-4bit \
    --out runs/safety_sft2_hardened
python scripts/safety_table.py
```

## Artifacts

- GitHub: https://github.com/PRINCE2-AI/lexsi-agentic-hinglish
- Adapter: https://huggingface.co/primce/tiny-aya-hinglish-lora (`sft2_adapter`)
- Run logs: https://huggingface.co/datasets/primce/lexsi-runs
- Space: [tuned demo link]

## Credits

Built on AgentTune and AuditKIT by Lexsi Labs, with Cohere Labs Tiny Aya and Hugging Face
hosting.
