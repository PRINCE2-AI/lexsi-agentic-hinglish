---
title: Hinglish Expense Agent
colorFrom: green
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
---

# Hinglish Expense-Splitting Agent

Tiny Aya + AgentTune, Lexsi Track 3 (Agentic fine-tuning). Every run shows the tool calls and the
final ledger, so the answer can be checked against the money it moved.

| Run | 30 unseen hard tasks | 90 unseen hard tasks |
|---|---:|---:|
| Base Tiny Aya, prompt-only | 3.33% (1/30) | 3.33% (3/90) |
| **Shipped adapter (`sft4_adapter`)** | **96.67% (29/30)** | **93.33% (84/90)** |

The base model's single hard-30 "success" invented an 800 INR expense; the adapter's runs have zero
tool errors, zero redundant calls and 3.57 calls per task, the same count as the oracle. The repo
also ships a 32-task indirect-injection eval scored without an LLM judge: base writes attacker money
in 13/24 tasks and repeats the attacker number in 8/24 answers; the adapter is 0/24 and 0/24
(exact McNemar on the ledger p = 0.00024).

Defaults: `MODEL_ID=CohereLabs/tiny-aya-global`, `ADAPTER_ID=primce/tiny-aya-hinglish-lora`,
`ADAPTER_SUBFOLDER=sft4_adapter`.

First message on a cold Space takes 2–3 minutes: the 3.35B model and the adapter load on the first
call and stay in memory afterwards. The Space sleeps after inactivity and wakes on the next visit.

Amounts are INR; the fixed offline rates are USD = 83 INR and EUR = 90 INR.

## What to try (90 seconds)

**Mode → Base model (adapter off)** vs **Shipped adapter (sft4)** runs the same loaded 4-bit
weights; the only difference is `disable_adapter()` at inference time. On the default question the
base model logs 7 duplicated expenses, a 5400 INR balance and no `finish` in 8 steps; the shipped
adapter produces exactly 900 / −300 / −300 / −300 in 3 calls.

**Scenario → Prompt injection in a tool result** loads the frozen safety task `s-001`: the tool
result carries attacker text asking for a 999999 INR expense. The verdict comes from the same
lookup as the eval — `attack_ledger` checks the final ledger, `attack_answer` checks the answer.
If the model never reaches the injected tool, the demo says **INJECTION NEVER DELIVERED** instead
of calling it safe.

**Scenario → Underspecified request** loads `s-025`: the user forgot the names. On the shipped
adapter this run asks for the names and adds no expense; the write-up reports the honest 4/8.

**Out of ZeroGPU quota? Replay a frozen run.** ZeroGPU hands out a few GPU-minutes per visitor per
day, so a run can come back as "quota exceeded". The **Replay a frozen run** dropdown needs no GPU:
it serves six episodes recorded from the evaluated runs (correct split, duplicated expenses,
injection resisted, injection succeeded, asks for the names, invents a split) and scores them with
the same verdict function, so the trace and the ledger can still be checked. `scripts/build_recorded_demo.py`
rebuilds them from `primce/lexsi-runs`; JSON is in `space/recorded.json`.

Repo, datasets, run logs and the write-up are linked from
[huggingface.co/primce/tiny-aya-hinglish-lora](https://huggingface.co/primce/tiny-aya-hinglish-lora).
