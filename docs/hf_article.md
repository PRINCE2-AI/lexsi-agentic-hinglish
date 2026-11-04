# Teaching Tiny Aya to split Hinglish expenses without losing the ledger

*Lexsi AI Safety & Alignment Hackathon 2026 — Track 3, Agentic fine-tuning. Solo entry.*

## TL;DR

- Base Tiny Aya on the frozen hard-30, no adapter: **3.33%** (1/30), and that one success is
  accidental. 16 of 30 episodes never finish and it averages **7.03** calls. On the 90-task
  holdout it is also **3.33%** (3/90), at **7.24** calls.
- SFT on 100 easy oracle trajectories: **63.33%** on hard-30, **100%** on eval-24.
- Adding GRPO on top (30 steps, graded reward) moved **nothing**: 63.33% again. The reward was
  saturated and the run was too short to move an answer-shape failure.
- SFT on the reweighted mix (184 rows): **96.67%** on hard-30, **3.57** calls — exactly the
  oracle's call count.
- SFT on the mix plus 96 abstention rows ("mix2", shipped): **96.67%** on hard-30, **75.56%** on
  a 90-task holdout, **100%** on eval-24. Three seeds of the same recipe span 16.67 pp on hard-30
  and 7.77 pp on hard-90; the shipped seed is the pre-registered one, and a second seed is better
  on the 90-task set.
- Independent **AuditKIT 1.1.3** replay of the same trajectories, no judge model: exact-call
  accuracy **0.033 → 1.000**, tool-call F1 **0.188 → 1.000**, redundant calls **0.327 → 0.000**.
- Indirect injection through tool results: base writes attacker money into the ledger in
  **13/24** tasks, the shipped adapter in **0/24** (p = 0.00024). Abstention on underspecified
  requests moves **0/8 → 2/8** — a direction, not a fix, against a bar of 6/8 that it missed.

## Problem statement

Track 3, primary: **Reliable tool calling** — train an agent to choose the right tool, pass
well-formed arguments, and call tools in the right order. Secondary: **Efficient
trajectories** — cut redundant, looping or failed calls without lowering success.

Every clause is scored mechanically, with no LLM judge: tool choice (`tac`), argument
validity (`ter`/`arr`), ordering (final ledger correctness), redundancy (`rad`), looping
(`lcf`), and calls/steps per task. The required Track 3 toolkits are both used: **AgentTune**
for the GRPO stage and **AuditKIT** for an independent agentic evaluation of the recorded runs.

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
| hard-90 eval | 90 | 15 per family, separately pinned — the tighter holdout |
| mix | 184 | easy 100 + hard 84 (hard rows repeated per family) |
| abstention train | 64 | 32 underspecified + 32 fully-specified twins |
| mix2 | 312 | mix 184 + abstention 96 (3 copies) + twins 32 |

The six hard families:

| Family | What it stresses | Hard-30 | Hard-90 |
|---|---|---:|---:|
| `settle5` | five people over three days | 6 | 15 |
| `two_query` | two questions in one Hinglish turn | 6 | 15 |
| `convert_sum` | two conversions then a sum | 3 | 15 |
| `empty_balance` | empty ledger, must not invent an expense | 3 | 15 |
| `trap_combo` | payer also listed among participants | 6 | 15 |
| `paraphrase` | Devanagari-mixed wording | 6 | 15 |

Both eval files are hash-pinned, and the mix builder refuses to run if a hard-train row appears
in an eval set, so repetition cannot leak eval rows. hard-90 is text-deduplicated against every
other split.

## Method

1. Five stateless tools: `add_expense`, `convert_currency`, `get_balance`, `settle_up`,
   `finish`; state travels in the ledger object, so rollouts cannot leak into each other.
2. SFT on oracle trajectories: LoRA r=16 on all linear layers, fp16, lr 2e-4, 2 epochs, labels
   masked to assistant turns. mix2 is ~156 optimizer steps on one T4.
3. GRPO with AgentTune `create_agentic_trainer`, graded reward (format + required tools +
   argument names/values + answer number), a no-duplicate-calls term, and — in round 2 — an
   abstain reward that pays 1.0 when the only call is `finish` with a question on an
   `abstain=1` row.
4. Evaluation: greedy, `--no-4bit`, one sample per task, on rented T4s (HF Jobs) and Kaggle T4 x2.
5. AuditKIT: `scripts/auditkit_eval.py` rebuilds each recorded episode as an
   `auditkit.agent_eval.AgentCase` + episode, runs `AgentEvalRunner` in `mode="recorded"`, and
   scores `tool_call_f1`, `tool_call_f1_name`, `tool_call_validity`, `redundant_tool_calls`,
   `agent_loop_detection`. No judge model is used.

## Results

| Model | hard-30 | hard-90 | eval-24 | calls (hard-90) |
|---|---:|---:|---:|---:|
| Base Tiny Aya, prompt-only | 3.33% (1/30) | 3.33% (3/90) | - | 7.24 |
| SFT-easy (100 rows, 3 epochs) | 63.33% (19/30) | - | 100% (24/24) | - |
| SFT-easy + GRPO (30 steps) | 63.33% (19/30) | - | 100% (24/24) | - |
| SFT-mix (184 rows, round 1) | 96.67% (29/30) | - | 100% (24/24) | - |
| **SFT-mix2, seed 42 (shipped)** | **96.67% (29/30)** | **75.56% (68/90)** | **100% (24/24)** | **3.27** |
| SFT-mix2, seed 7 | 93.33% (28/30) | 83.33% (75/90) | - | 3.42 |
| SFT-mix2, seed 123 | 80.00% (24/30) | 75.56% (68/90) | - | 3.23 |
| oracle | 100% (30/30) | 100% (90/90) | 100% (24/24) | 3.57 |

Wilson 95% intervals: base 0.6–16.7% (hard-30) and 1.1–9.3% (hard-90); shipped adapter
83.3–99.4% and 65.8–83.3%; seed 7 78.7–98.2% and 74.3–89.6%; seed 123 62.7–90.5% and 65.8–83.3%;
eval-24 86.2–100%.

Paired exact McNemar on the uploaded trajectories, same task ids:

- base → shipped adapter, hard-90: **66 fixed, 1 broken, p = 9.2e-19**.
- base → shipped adapter, hard-30: **28 fixed, 0 broken, p = 7.5e-9**.
- easy-SFT + GRPO → SFT-mix, hard-30: **10 fixed, 0 broken, p = 0.00195**.
- shipped seed 42 → seed 7, hard-90: **7 fixed, 0 broken, p = 0.0156**.

The base model's one hard-30 success is a lucky one: on `h-e-029` it answered "0 INR" correctly,
but only after inventing an 800 INR expense nobody asked for. The answer was right, the behaviour
was not — which is why the safety split scores the ledger and not just the text.

Per family on the 90-task holdout (15 each):

| Family | Base | Seed 42 (shipped) | Seed 7 | Seed 123 |
|---|---:|---:|---:|---:|
| settle5 | 0/15 | 15/15 | 15/15 | 14/15 |
| trap_combo | 0/15 | 15/15 | 15/15 | 15/15 |
| paraphrase | 0/15 | 13/15 | 13/15 | 13/15 |
| two_query | 0/15 | 13/15 | 14/15 | 11/15 |
| convert_sum | 0/15 | 8/15 | 11/15 | 11/15 |
| empty_balance | 3/15 | 4/15 | 7/15 | 4/15 |

## What the 90-task holdout changed

The 30-task split was optimistic by 21 points. Two families it under-samples are the two that
stay hard: `empty_balance` goes from 3/3 on hard-30 to 4/15 on hard-90, `convert_sum` from 2/3
to 8/15. The shipped adapter keeps a clean trajectory profile on hard-90 — tool error rate
**1.1%**, `overcall` **0.000**, redundant calls **0.000** — so the remaining failures are answer
shape (`tac` 0.822, loop-free 0.989) rather than tool use, exactly the failure mode the round-1
analysis predicted.

The seed spread is wider than any single change made after round 1: 16.67 pp on hard-30, 7.77 pp
on hard-90. It is published instead of hidden. Shipping seed 7 would add 7.8 pp on hard-90; it
would also mean choosing the seed on the split that is reported. The shipped adapter is seed 42,
fixed before the seed runs existed, and it is the median of the three on the mean of both splits.

## AuditKIT: an independent scorer on the same episodes

The track asks for an agentic evaluation toolkit. AuditKIT replays the recorded trajectories —
same cases, same final-state assertions, no judge model — and scores tool calls with its own
implementation:

| Split | Model | tool_call_exact | tool_call_f1 | precision | recall | redundant | loop |
|---|---|---:|---:|---:|---:|---:|---:|
| hard-30 | base | 0.0333 | 0.1875 | 0.1547 | 0.3194 | 0.3270 | 0.4333 |
| hard-30 | shipped | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 |
| hard-90 | shipped | 0.7778 | 0.8114 | 0.8130 | 0.8102 | 0.0109 | 0.9875 |

Its success verdict agrees with this repo's deterministic scorer on **30/30** hard-30 tasks and
**89/90** hard-90 tasks, so the toolkit reproduces the result rather than replacing it. That
agreement number is the point: two independent scorers, one dataset, no judge.

The base model, on the same rows: tool error rate **22.3%** (hard-30) and **17.4%** (hard-90),
`overcall` **1.08**, `strict_rate` **0.534**, and 16/30 episodes that never reached `finish`.
The shipped adapter has zero tool errors and zero redundant calls on hard-30, and matches the
oracle on calls per task — which is why the secondary statement ("efficient trajectories") is
reported on the same row as success.

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
The bottleneck was coverage of the answer shape, not the RL stage. Round 2 reruns GRPO with an
abstention-aware reward, 40 steps and lr 3e-6 on top of the mix2 adapter; the outcome is reported
in `docs/ROUND2.md` whichever way it points.

The abstention slice is the other honest negative: it moved `asked` from 0/8 to 2/8 against a
pre-registered bar of 6/8 (`invented_expense` 7/8 → 6/8). At 8.4% supervised ask-tokens the
policy can produce the behaviour but does not prefer it.

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
| **SFT-mix2 (`sft3`, shipped)** | **15/24 (62.5%)** | 15/24 | **0/24** | 2/24 (8.3%) |
| SFT-mix2 + hardened prompt | 12/24 (50.0%) | 12/24 | **0/24** | 3/24 (12.5%) |

Paired exact McNemar: base → shipped removes **13 ledger attacks, introduces 0, p = 0.00024**.
The prompt-level defence adds 4 safe tasks on `sft2` (p = 0.125, directional) and removes 3 on
`sft3`, so it is not the default.

Three honest readings:

- **Fine-tuning fixed the money-moving failure, not injection as a whole.** Zero written
  attacks across all tuned variants, but the shipped model still repeats 999999 in prose on 2 of
  24 tasks, and it solves 15/24 under attack versus 29/30 on the clean hard-30. An injection
  that does not move money still costs tasks; without a no-payload control on these exact 24,
  treat that gap as a warning sign rather than a measured tax.
- **Abstention improved, and the bar was still missed.** 2/8 asked, 6/8 invented, against a
  pre-registered 6/8 and 2/8. The direction is real (`unplanned_calls` also drops to 0.12) but
  the claim is "visible to the policy", not "solved".
- **The defence is prompt-level, and its sign flips with the adapter.** On `sft2` the two-rule
  hardened prompt never regressed a task; on `sft3` it costs three. A defence that depends on the
  adapter is not a defence yet — the architectural fix (treat tool output as data by
  construction) is unmeasured.

## Why these choices

- **SFT memorizes, RL generalizes** (Chu et al., ICML 2025, arXiv:2501.17161): the held-out hard
  splits are the unseen-variant test. SFT scores 100% on the template-matched eval-24 and 63.33%
  on hard-30 — the paper's SFT drop, reproduced on Hinglish tool use.
- **ToolRL** (Qian et al., arXiv:2504.13958): reward is decomposed into format, tool-name,
  parameter-name, parameter-value and answer terms instead of a binary hit, and there is no
  length reward. The `empty_balance` family is ToolRL's relevance/irrelevance case: the
  correct action is to add nothing.
- **APIGen** (arXiv:2406.18518) and **ToolACE** (arXiv:2409.00920): every task is
  oracle-verified before use, the eval files are SHA-256 pinned, and the hard splits are
  organised into complexity families so failures can be attributed.
- **τ-bench** (arXiv:2406.12045): success is the final ledger state, not string similarity.
- **DeepSeekMath** (arXiv:2402.03300) for GRPO, **Tiny Aya** (arXiv:2603.11510) for the model.

What we did **not** run, so no claim is made about it: cold-start GRPO without SFT (the
control that would test whether SFT is necessary here), a Hammer-style schema-shift eval,
verification-iteration repair beyond the parse-error correction prompt, a no-payload control
on the 24 injection tasks, a seed sweep of the safety split, and an adaptive attacker that can
see the eval.

## Limitations

- hard-90 has 90 tasks; counts are reported next to percentages, and its Wilson interval is
  still ±7 pp.
- Hard rows are held-out *instances* of the same six templates, so this measures
  generalization across amounts/names/languages, not across task formats.
- eval-24 is in-distribution with easy train, which is why it stays at 100%.
- Three seeds bound the variance; they do not remove it, and the spread is wider than any
  single change made after round 1.
- Fixed offline FX rates, fixed six-person name universe, no free-form tools.
- The injection split is 32 tasks, the payloads are English-leaning, and the defence is one
  prompt variant. Greedy decoding, one sample per task.

## Reproduce

```bash
python data/build_hard.py && python data/build_hard90.py && python data/build_mix.py
python data/build_abstain.py && python data/build_safety.py
python scripts/verify_all.py          # no GPU, no token: 18 checks

python scripts/train_sft.py --tasks data/tasks_train_mix2.jsonl \
    --epochs 2 --seed 42 --out runs/sft3 --merge-out runs/sft3_merged
python scripts/run_baseline.py --model runs/sft3_merged \
    --tasks data/tasks_hard_eval.jsonl --no-4bit --out runs/sft3_hard30
python scripts/run_baseline.py --model runs/sft3_merged \
    --tasks data/tasks_hard90_eval.jsonl --no-4bit --out runs/sft3_hard90
python scripts/run_baseline.py --model runs/sft3_merged \
    --tasks data/tasks_safety_eval.jsonl --safety --no-4bit --out runs/safety_sft3

python scripts/stats.py --a runs/base_hard90 --b runs/sft3_hard90
python scripts/auditkit_eval.py --run sft3_hard90 --tasks data/tasks_hard90_eval.jsonl
python scripts/safety_table.py
```

## Artifacts

- GitHub: https://github.com/PRINCE2-AI/lexsi-agentic-hinglish
- Adapter: https://huggingface.co/primce/tiny-aya-hinglish-lora (`sft3_adapter`; seed adapters
  alongside it)
- Run logs: https://huggingface.co/datasets/primce/lexsi-runs
- Space: https://huggingface.co/spaces/primce/hinglish-expense-agent (T4, shipped adapter, visible tool trace)
- This article: https://huggingface.co/spaces/primce/hinglish-track3-article
- Collection: https://huggingface.co/collections/primce/lexsi-track-3-hinglish-expense-agent-6aeb3207e10dfc7dda6329d6

## Credits

Built on AgentTune and AuditKIT by Lexsi Labs, with Cohere Labs Tiny Aya and Hugging Face
hosting.
