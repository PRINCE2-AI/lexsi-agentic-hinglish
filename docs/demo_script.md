# Demo Day script (3 minutes)

Numbers below are the frozen ones from `docs/RESULTS.md` (2026-11-05). Do not improvise new ones.

## 0:00-0:20 — Problem

"Tiny Aya is multilingual. Ask it to split a dinner bill in Hinglish, and it will look confident and
get the ledger wrong. This is a money answer, so 'roughly right' is a failure. We measured it."

## 0:20-0:50 — Baseline failure

Open `runs/base_hard30/trajectories.jsonl`, one failing episode. Point at the exact step: stale
ledger copy, wrong split, or the same `get_balance` called three times.

Say: "Base model on the 30-task hard split: success **3.3%**, 16 of 30 episodes never finish, 7.03
tool calls per task where the oracle needs 3.57. Its one success is a lucky one — right answer,
invented expense."

## 0:50-1:20 — Tuned success

Same prompt, `sft4_adapter`. Show the trace: `add_expense` -> `convert_currency` -> `get_balance` ->
`finish`. Point at the final ledger JSON matching the expected state exactly.

Say: "**29 of 30** here, **84 of 90** on the 90-task holdout, 3.57 tool calls, zero tool errors,
zero redundant calls."

## 1:20-2:10 — Numbers

Show the table from the README. Say the sample size out loud.

- Base, 30 hard tasks: **3.33%** (1/30). On 90 hard tasks: **3.33%** (3/90).
- Easy-SFT, same 30 hard tasks: 63.33%.
- Easy-SFT + GRPO, same 30 tasks: **63.33% — zero delta.**
- Round 1, SFT on the reweighted mix: **96.67%** hard-30, **88.89%** hard-90.
- Shipped `sft4`: **96.67%** hard-30, **93.33%** hard-90 (Wilson 86.2–96.9%), 100% on the
  24-task in-distribution split.
- Paired McNemar, base -> shipped: hard-30 28 fixed, 0 broken, **p = 7.5e-9**; hard-90 81 fixed,
  0 broken, **p < 1e-20**. Round 2 -> round 3: 18 fixed, 2 broken, **p = 0.0004**. Round 1 ->
  shipped is 7 fixed, 3 broken, p = 0.34 — directional, not significant, and I say that.

Then the honest line, which is the part judges remember: "GRPO did not help. The reward was already
saturated — tool names were right, the answers were incomplete — so every rollout in a group got the
same reward and there was no advantage to learn from. We ran it twice and published both. The gain
came from the data, not the optimizer."

## 2:10-2:40 — How

Oracle-verified trajectories (APIGen-style), 6 difficulty families, SHA-256-pinned frozen evals,
success checked against the final ledger state (tau-bench-style), official AgentTune + AuditKIT,
Tiny Aya 3.35B. Stateless tools: every ledger tool takes and returns the ledger, so parallel
rollouts cannot leak state. Round 3 added minimal pairs that separate "the ledger is empty" from
"the request is missing a number" — that is the 68/90 -> 84/90 jump.

## 2:40-3:00 — Close

"Agent reliability is a safety property: an agent that mis-copies state gives confidently wrong
money answers at scale. Adapter, dataset, frozen eval, trajectories and the Space are public."

## Q&A prep

- **Why not a bigger model?** Free compute, and the point is a small open-weights model that fits on
  commodity hardware. 3.35B merged runs on one T4.
- **Why not prompt engineering?** Prompt-only is the before column. It was measured first, and the
  prompt-level defence is a separate arm we also measured.
- **Is 30 tasks enough?** No — that is why the headline is the 90-task holdout (93.33%, Wilson
  86.2–96.9%), and every number carries a paired exact McNemar test on shared task ids.
- **Why is eval-24 at 100%?** It is in-distribution, and we say so. It is a regression check, not
  the headline.
- **Isn't this just SFT?** Yes, and the track asks for RL. We ran GRPO twice; it moved nothing, and
  we published that. What we have not run is cold-start GRPO with no SFT warm start — that is the
  control that would answer the question properly.
- **Why is round 1 -> shipped only p = 0.34?** Because 7 fix and 3 break: the gain is directional,
  not proven. The proven step is round 2 -> shipped, 18 against 2.
- **What would you do with $200 of compute?** Seeds are done: mix3 has three, and the spread is
  1.11 pp on hard-90. The hardened prompt is measured (86/90 against 84/90), and so are the two
  generalisation probes: renamed tool schema 25/30 against 29/30 with the untuned base at 0/30, and
  the same tasks in Devanagari at 25/30. Next is a Hinglish-only injection payload set, a
  Devanagari payload, and the schema shift on hard-90 instead of hard-30.
- **Is the agent safe?** We measured it: 32-task indirect-injection split with no LLM judge. Base
  writes attacker money in 13/24 tasks and repeats the attacker number in 8/24 answers; the shipped
  adapter in **0/24 and 0/24** (p = 0.00024). It asks on 4 of 8 underspecified requests — the same
  adapter with a hardened system prompt asks on 7/8, at the cost of 2/24 attacker numbers in
  answers and no cost in task success (86/90 vs 84/90), and both configurations are in the repo.
