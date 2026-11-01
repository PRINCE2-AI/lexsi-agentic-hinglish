# Hinglish Expense-Splitting Agent — Lexsi Track 3

A solo-friendly Track 3 (Agentic Fine-tuning) entry for the **Lexsi AI Safety & Alignment Hackathon**, Nov 1–22, 2026.

Tiny Aya learns to call tools reliably in Hinglish: it records shared expenses, carries the ledger across turns, converts USD/EUR, and settles balances. Success is checked against the **final ledger state**, not an LLM judge, so the before/after numbers are reproducible.

## Why this project

- Track 3 asks for an agent that is **post-trained for reliable tool use**, evaluated with AuditKIT.
- The Cohere models do not render `tools` in their chat template, so AgentTune falls back to tool schemas in the system prompt. Baseline Tiny Aya is therefore weak at this task — which gives a real, measurable before/after gap.
- The outcome is deterministic. Judges can re-run the dataset and get the same ledger.
- Hinglish + Aya's multilingual pretraining is a differentiated angle in a crowded safety hackathon.

## What is already here

- Deterministic ledger and five stateless tools (`add_expense`, `convert_currency`, `get_balance`, `settle_up`, `finish`).
- **20 Hinglish tasks**, all verified against the oracle (`20/20` success, 0 tool errors).
- Programmatic metrics: `success`, `tac`, `arr`, `ter`, `rad`, `lcf`, plus answer match.
- Standalone Tiny Aya baseline runner (transformers, 4-bit) that writes AgentTune-compatible `trajectories.jsonl`.
- GRPO training script using `create_agentic_trainer` with custom rewards.
- AuditKIT handoff via `episodes_from_agenttune`.
- Gradio demo with a visible tool trace and final ledger.

## Layout

```
hinglish_agent/
  ledger.py      deterministic money math, splits, settlement
  tools.py       stateless tool functions + JSON schemas
  prompts.py     Tiny Aya fallback tool prompt
  parser.py      robust JSON tool-call parser
  runner.py      agent loop, oracle, metrics
  metrics.py     summary + markdown table
data/
  build_tasks.py generates and verifies tasks.jsonl
  tasks.jsonl    20 Hinglish tasks (generated, do not hand-edit)
scripts/
  run_oracle_eval.py  verify the dataset, no GPU
  run_baseline.py     Tiny Aya baseline / tuned evaluation
  train_grpo.py       GRPO fine-tuning
  auditkit_eval.py    load run logs into AuditKIT
space/
  app.py              Gradio demo
docs/
  hf_article.md       submission write-up skeleton
  roi.md              ROI comparison template
  demo_script.md      3-minute Demo Day script
tests/
  test_scaffold.py    no-dependency sanity tests
```

## Quickstart

One-click Colab (T4 GPU): [notebooks/colab_starter.ipynb](notebooks/colab_starter.ipynb) — open it from GitHub's "Open in Colab" badge.

Pure-Python checks (Windows/macOS/Linux, no GPU, no model download):

```bash
python data/build_tasks.py
python scripts/run_oracle_eval.py
python tests/test_scaffold.py
```

Expected: `oracle check: 20/20 success, 0 tool errors`.

### Tiny Aya access

1. Accept the licence on [CohereLabs/tiny-aya-global](https://huggingface.co/CohereLabs/tiny-aya-global) (and `tiny-aya-base` if you use it).
2. Create a **Write** token in HF Settings → Access Tokens.
3. On Colab/Kaggle: `pip install -U huggingface_hub` then `hf auth login`.

The model is gated; without this, loading fails with a 401/403.

### GPU environment (Colab or Kaggle)

Use Linux + CUDA. The pinned AgentTune stack (Python 3.12, torch 2.11, transformers >= 5.15) does not belong in a local Windows venv.

```bash
git clone <your-repo-url>
cd lexsi-agentic-hinglish
pip install -r requirements.txt
```

Then:

```bash
# 1) baseline
python scripts/run_baseline.py --model CohereLabs/tiny-aya-global --limit 20

# 2) training: smoke test first, then a real run
python scripts/train_grpo.py --check
python scripts/train_grpo.py --max-steps 10
python scripts/train_grpo.py --max-steps 120

# 3) tuned evaluation
python scripts/run_baseline.py --model CohereLabs/tiny-aya-global \
    --adapter runs/grpo_tiny_aya --out runs/tuned

# 4) AuditKIT handoff
python scripts/auditkit_eval.py --trajectories runs/baseline/trajectories.jsonl
```

`--limit 20` is the whole starter set. On a free T4, baseline inference is roughly 5–10 minutes for 20 tasks; 120 GRPO steps is a multi-hour job, so run it on Kaggle's 2×T4 (30 h/week) and keep `--max-completion-length 256`.

## Metrics

Programmatic, computed in `hinglish_agent/runner.py`:

| Code | Meaning |
|---|---|
| `success` | final ledger matches expected balances (and settlement, when asked) |
| `tac` | tool accuracy: all required tools were called |
| `arr` | argument reliability: `1 - ter` |
| `ter` | tool error rate: failed calls / total calls |
| `rad` | redundant actions: repeated identical calls |
| `lcf` | loop collapse: episode hit max steps without `finish` |

Fill this in after your runs:

| Metric | Baseline | Tuned |
|---|---:|---:|
| Task success | TBD | TBD |
| tac | TBD | TBD |
| arr | TBD | TBD |
| ter | TBD | TBD |
| rad | TBD | TBD |
| lcf | TBD | TBD |

The oracle table should be perfect; use it as a sanity check, not as a result.

## Training notes

- Tools are **stateless**: every ledger tool takes and returns the ledger object. Parallel GRPO rollouts cannot leak state.
- The GRPO reward is format + expected tool + answer number. The real quality signal is the post-hoc ledger check, so do not tune rewards to chase `success` on the eval set.
- Baseline must be measured **before** training on the same 20 tasks. If you change tasks after seeing baseline results, re-run both.
- `max_steps=10` only proves the pipeline works. It is not a result.
- If GRPO OOMs or crawls on a T4, reduce `--max-completion-length`, keep `--generations 2`, and use `--batch-size 1 --grad-accum 4`. Do not switch to 8B/32B models on free hardware.

## Demo deployment

The Space needs a GPU. Two options:

1. Push the whole repo to a Space and set `app_file: space/app.py` in the Space README.
2. Push `space/` plus `hinglish_agent/` to the Space (keep the same relative layout) and use `space/app.py` as the app file.

Set `MODEL_ID` and, for the tuned demo, `ADAPTER_ID=<you>/tiny-aya-hinglish-lora`. Deploy two Spaces (base and tuned) or show the before/after table in the model card.

## Submission checklist (Track 3)

- [ ] Public GitHub repo with this code and the generated `data/tasks.jsonl`
- [ ] Public HF Space demo with tool trace
- [ ] HF artifacts: LoRA adapter, tokenizer, `trajectories.jsonl` logs, dataset
- [ ] HF Article: metrics chosen and why, dataset/task/model rationale, before/after table
- [ ] At least two metrics, before and after
- [ ] Optional ROI table: self-hosted Tiny Aya vs closed-source API at a realistic volume
- [ ] Only Cohere Labs models used for the trained model
- [ ] Submission form link from Discord filled before Nov 14 end of day

## 13-day plan

| Day | Work |
|---|---|
| Nov 1 | Accept licence, install, run oracle + spine notebooks, commit repo |
| Nov 2 | Baseline on 20 tasks, inspect failures |
| Nov 3–4 | Expand to 80–100 tasks in `data/build_tasks.py`, re-verify |
| Nov 5–7 | GRPO smoke test then real run; tuned eval |
| Nov 8–9 | Error analysis, reward/task fixes, second run |
| Nov 10 | Freeze results; optional extra analysis |
| Nov 11–12 | Space, HF Article, artifacts, README table |
| Nov 13 | Submit early |
| Nov 16–21 | If shortlisted: polish + demo script |
| Nov 22 | Demo Day |

## Gotchas (from the library READMEs)

- `ToolRegistry.get(...)` imports every builtin tool and can crash without `langchain_community`. This repo uses plain functions instead.
- Cohere chat templates do not render `tools`; AgentTune's fallback prompt is the intended path, not a bug to patch.
- `agenttune train` CLI has no `--tools` flag; agentic training is Python API only.
- `RougeMetric`/`BLEU` and `DistilledJudge` silently return `0.0` when optional deps are missing. This repo avoids them.
- AuditKIT reads `trajectories.jsonl` directly; do not convert formats.
- `max_len` for Tiny Aya must be unset or `>= 512` (its system preamble is ~366 tokens).
- On Colab, `pip uninstall -y torchao` before importing SafeTune. Not needed here, but applies if you add Track 2 tools.

## Credits

Built on [AgentTune](https://github.com/Lexsi-Labs/AgentTune) and [AuditKIT](https://github.com/Lexsi-Labs/AuditKIT) by Lexsi Labs, with Cohere Labs Tiny Aya and Hugging Face hosting.
