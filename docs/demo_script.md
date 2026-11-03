# Demo Day script (3 minutes)

Numbers below are the frozen ones from `docs/RESULTS.md` (2026-11-03). Do not improvise new ones.

## 0:00-0:20 — Problem

"Tiny Aya is multilingual. Ask it to split a dinner bill in Hinglish, and it will look confident and
get the ledger wrong. This is a money answer, so 'roughly right' is a failure. We measured it."

## 0:20-0:50 — Baseline failure

Open `runs/base_hard30/trajectories.jsonl`, one failing episode. Point at the exact step: stale
ledger copy, wrong split, or the same `get_balance` called three times.

Say: "Base model on the 30-task hard split: success **X%**, 70% of episodes never finish, 9.25 tool
calls per task where the oracle needs 3.5."

## 0:50-1:20 — Tuned success

Same prompt, `sft2_adapter`. Show the trace: `add_expense` -> `convert_currency` -> `get_balance` ->
`finish`. Point at the final ledger JSON matching the expected state exactly.

Say: "**29 of 30**, 3.57 tool calls, zero tool errors."

## 1:20-2:10 — Numbers

Show the table from the README. Say the sample size out loud.

- Base, 20 easy tasks: 25% success.
- Easy-SFT, 30 hard tasks: 63.33%.
- Easy-SFT + GRPO, same 30 tasks: **63.33% — zero delta.**
- SFT on the reweighted mix: **96.67%**, and 100% on the 24-task in-distribution split.
- Paired McNemar on the 30 shared tasks: 10 fixed, 0 broken, **p = 0.00195**.

Then the honest line, which is the part judges remember: "GRPO did not help. The reward was already
saturated — tool names were right, the answers were incomplete — so every rollout in a group got the
same reward and there was no advantage to learn from. The gain came from the data, not the optimizer."

## 2:10-2:40 — How

Oracle-verified trajectories (APIGen-style), 6 difficulty families, SHA-256-pinned frozen eval,
success checked against the final ledger state (tau-bench-style), official AgentTune + AuditKIT,
Tiny Aya 3.35B. Stateless tools: every ledger tool takes and returns the ledger, so parallel
rollouts cannot leak state.

## 2:40-3:00 — Close

"Agent reliability is a safety property: an agent that mis-copies state gives confidently wrong
money answers at scale. Adapter, dataset, frozen eval, trajectories and the Space are public."

## Q&A prep

- **Why not a bigger model?** Free compute, and the point is a small open-weights model that fits on
  commodity hardware. 3.35B merged runs on one T4.
- **Why not prompt engineering?** Prompt-only is the before column. It was measured first.
- **Is 30 tasks enough?** No — that is why every number carries a Wilson interval (~83-99% for
  29/30) and a paired test. Expanding to 90 tasks is the first thing more compute buys.
- **Why is eval-24 at 100%?** It is in-distribution, and we say so. It is a regression check, not
  the headline. The headline is hard-30.
- **Isn't this just SFT?** Yes, and the track asks for RL. We ran GRPO; it moved nothing, and we
  published that. What we have not run is cold-start GRPO with no SFT warm start — that is the
  control that would answer the question properly.
- **What would you do with $200 of compute?** 90 hard tasks, three seeds, and one schema-shift eval
  (rename tools and parameters) to see whether the adapter learned the task or the format.
