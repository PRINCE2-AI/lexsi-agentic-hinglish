---
title: Hinglish Expense Agent
colorFrom: green
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
---

# Hinglish Expense-Splitting Agent

Tiny Aya + AgentTune, Lexsi Track 3. Tool trace and final ledger are visible for every run.

Set `MODEL_ID` (default `CohereLabs/tiny-aya-global`) and optionally `ADAPTER_ID` to your LoRA adapter.
The frozen submission adapter is `primce/tiny-aya-hinglish-lora` at `sft2_adapter`, so the Space
variables are:

```
ADAPTER_ID=primce/tiny-aya-hinglish-lora
ADAPTER_SUBFOLDER=sft2_adapter
```

`ADAPTER_SUBFOLDER` defaults to `sft2_adapter`; set it to empty when the adapter sits at the repo root.
