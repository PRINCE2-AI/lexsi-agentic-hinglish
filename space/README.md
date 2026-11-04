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
| **Shipped adapter (`sft3_adapter`)** | **96.67% (29/30)** | **75.56% (68/90)** |

The base model's single hard-30 "success" invented an 800 INR expense; the adapter's runs have zero
tool errors, zero redundant calls and 3.57 calls per task, the same count as the oracle. The repo
also ships a 32-task indirect-injection eval scored without an LLM judge: base writes attacker money
in 13/24 tasks, the adapter in 0/24 (exact McNemar p = 0.00024).

Defaults: `MODEL_ID=CohereLabs/tiny-aya-global`, `ADAPTER_ID=primce/tiny-aya-hinglish-lora`,
`ADAPTER_SUBFOLDER=sft3_adapter`. Set `ADAPTER_ID` to an empty value to demo the base model instead;
set `ADAPTER_SUBFOLDER` to empty when the adapter sits at the repo root.

First message on a cold Space takes 2–3 minutes: the 3.35B model and the adapter load on the first
call and stay in memory afterwards. The Space sleeps after inactivity and wakes on the next visit.

Amounts are INR; the fixed offline rates are USD = 83 INR and EUR = 90 INR. The prompt-only row is
the same question answered by the untuned model, which is why the demo is worth comparing against a
second Space running with `ADAPTER_ID` unset.

Repo, datasets, run logs and the write-up are linked from
[huggingface.co/primce/tiny-aya-hinglish-lora](https://huggingface.co/primce/tiny-aya-hinglish-lora).
