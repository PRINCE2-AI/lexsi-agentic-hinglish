# Demo Day script (3 minutes)

## 0:00–0:20 — Problem

"Tiny Aya is multilingual, but it cannot reliably carry a ledger across tool calls in Hinglish. We measured it."

## 0:20–0:50 — Baseline failure

Show one baseline trace: wrong participant split or stale ledger copy. Point at the exact failed step and the tool error.

## 0:50–1:30 — Tuned success

Same prompt, tuned adapter. Show the tool trace: add_expense → convert → get_balance → finish, with the ledger intact. Point at the final ledger matching the expected state.

## 1:30–2:10 — Numbers

Show the before/after table. Say the sample size out loud: "20 tasks, success X to Y". Mention tool error rate and loop collapse; those are the reliability metrics, not just accuracy.

## 2:10–2:40 — How

AgentTune GRPO with stateless tools, AuditKIT for evaluation, Tiny Aya 3.35B 4-bit. One sentence on why stateless tools matter for parallel rollouts.

## 2:40–3:00 — Close

"Agent reliability is a safety property: an agent that mis-copies state gives confidently wrong money answers. The adapter, dataset, logs and Space are public."

## Q&A prep

- Why not a bigger model? Free compute and the point is a small open-weights model.
- Why not prompt engineering? We measured the prompt-only baseline first; it is the before column.
- Is 20 tasks enough? It is a starter set; report counts and add tasks if asked.
- What would you do with $200 credits? Expand to 100 tasks and run three seeds.
