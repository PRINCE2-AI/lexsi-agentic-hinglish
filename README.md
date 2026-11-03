# Hinglish Expense-Splitting Agent — Lexsi Track 3

A solo-friendly Track 3 (Agentic Fine-tuning) entry for the **Lexsi AI Safety & Alignment Hackathon**, Nov 1–22, 2026.

Tiny Aya learns to call tools reliably in Hinglish: it records shared expenses, carries the ledger across turns, converts USD/EUR, and settles balances. Success is checked against the **final ledger state**, not an LLM judge, so the before/after numbers are reproducible.

## Problem statement (Track 3)

The live Track 3 statements are on [hackathon.lexsi.ai/tracks](https://hackathon.lexsi.ai/tracks). This project targets one as primary and measures a second:

> **Reliable tool calling** — *Train an agent with reinforcement learning on real tool-calling rollouts to choose the right tool, pass well-formed arguments and call tools in the right order.*

> **Efficient trajectories** — *Cut redundant, looping or failed tool calls and the number of steps per task, without lowering the success rate.*

The third and final statement this project touches is *Data and database agents*: the agent answers finance questions by operating on a ledger and is rewarded on verified results, not on text similarity.

How each clause is measured, all without an LLM judge:

| Statement clause | Metric in this repo |
|---|---|
| choose the right tool | `tac` — every required tool was called |
| well-formed arguments | `ter`, `arr` — failed calls and argument reliability |
| right order | `ledger_ok` — order shows up as a correct final ledger |
| redundant calls | `rad` — repeated identical calls |
| looping | `lcf` — episode hit max steps without `finish` |
| steps per task | `calls`, `steps` — averaged in `summary.json` |
| without lowering success | `success` reported alongside, never traded away |

## Why this project

- The Cohere models do not render `tools` in their chat template, so AgentTune falls back to tool schemas in the system prompt. Baseline Tiny Aya is therefore weak at this task — which gives a real, measurable before/after gap.
- The outcome is deterministic. Judges can re-run the dataset and get the same ledger.
- Hinglish + Aya's multilingual pretraining is a differentiated angle in a crowded safety hackathon.

## What is already here

- Deterministic ledger and five stateless tools (`add_expense`, `convert_currency`, `get_balance`, `settle_up`, `finish`).
- **226 generated Hinglish tasks**, oracle-verified: 100 train + 24 held-out eval, 40 train /
  30 held-out hard tasks across six stress families (multi-day settlement, two queries in one
  turn, conversion sums, empty ledger, payer-in-participants trap, paraphrase), and **32 safety
  tasks** (indirect prompt injection through tool results, plus underspecified requests). The
  20-task starter set is still there for quick checks.
- Programmatic metrics: `success`, `tac`, `arr`, `ter`, `rad`, `overcall`, `lcf`, plus answer match,
  trajectory efficiency (`calls`, `steps`) and strict-JSON rate.
- **Safety split**: a tool result can carry attacker text; `safe`, `attack_ledger`,
  `attack_answer`, `unplanned_calls` and `abstained` are measured without an LLM judge. See
  [docs/SAFETY.md](docs/SAFETY.md).
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
  safety.py      injection / underspecification scoring
data/
  task_builder.py shared scenario -> expected-fields conversion
  build_tasks.py  generates and verifies tasks.jsonl
  tasks.jsonl     20 Hinglish starter tasks (generated, do not hand-edit)
  build_generated.py  100 train + 24 held-out eval tasks, oracle-verified
  tasks_train.jsonl / tasks_eval.jsonl
  build_hard.py  40 train + 30 held-out hard tasks, SHA-256 pinned
  build_mix.py   easy 100 + hard 84 reweighted SFT mix
  build_safety.py 32 injection / underspecified tasks, SHA-256 pinned
scripts/
  run_oracle_eval.py  verify the dataset, no GPU
  run_baseline.py     Tiny Aya baseline / tuned evaluation (--safety, --hardened-prompt)
  train_sft.py        oracle-trajectory SFT warm-start
  train_grpo.py       GRPO fine-tuning
  stats.py            Wilson 95% CI + exact McNemar on two eval runs
  safety_table.py     render the injection comparison table from saved runs
  score_strict.py     recover strict-JSON metrics from saved trajectories
  auditkit_eval.py    load run logs into AuditKIT
space/
  app.py              Gradio demo
docs/
  hf_article.md       submission write-up skeleton
  roi.md              ROI comparison with measured token counts
  SAFETY.md           injection eval: threat model, metrics, results
  demo_script.md      3-minute Demo Day script
  PLANBOOK.md         full Nov 2–14 operating plan (data, SFT→GRPO, artifacts, risks)
tests/
  test_scaffold.py    no-dependency sanity tests
  test_hard_data.py   hard-split pin, oracle, require_answer gate
  test_safety.py      injection eval pin, payload delivery, attack scoring
  test_stats.py       Wilson CI + exact McNemar
```

## Quickstart

One-click Colab (T4 GPU): [notebooks/colab_grpo.ipynb](notebooks/colab_grpo.ipynb) — fresh clone, verifies the fp16 fix, then baseline and a 2-step GRPO smoke test. The older [colab_starter.ipynb](notebooks/colab_starter.ipynb) still works but trains in float32.

Pure-Python checks (Windows/macOS/Linux, no GPU, no model download):

```bash
python data/build_tasks.py
python data/build_generated.py
python scripts/run_oracle_eval.py
python tests/test_scaffold.py
```

Expected: `oracle check: 20/20 success, 0 tool errors`, then `train: 100/100` and `eval: 24/24`, no train/eval text overlap.

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
# 1) baseline on the held-out split (left column, ~45 min)
python scripts/run_baseline.py --model CohereLabs/tiny-aya-global \
    --tasks data/tasks_eval.jsonl --out runs/baseline_eval

# 1b) SFT warm-start on oracle trajectories (installs the JSON + ledger-copy format)
python scripts/train_sft.py --check
python scripts/train_sft.py --tasks data/tasks_train.jsonl --epochs 3 \
    --out runs/sft --merge-out runs/sft_merged

# 2) GRPO from the SFT model: smoke test first, then a real run
python scripts/train_grpo.py --check
python scripts/train_grpo.py --model runs/sft_merged --max-steps 2
python scripts/train_grpo.py --model runs/sft_merged --max-steps 60 --efficiency

# 3) tuned evaluation on held-out tasks
python scripts/run_baseline.py --model CohereLabs/tiny-aya-global \
    --adapter runs/sft --tasks data/tasks_eval.jsonl --no-4bit --out runs/sft_eval

# 4) paired significance: Wilson CI + exact McNemar on shared task ids
python scripts/stats.py --a runs/sft_hard30 --b runs/sft2_hard30

# 5) safety split: injection through tool results, no LLM judge
python data/build_safety.py
python scripts/run_baseline.py --model runs/sft2_merged     --tasks data/tasks_safety_eval.jsonl --safety --no-4bit --out runs/safety_sft2
python scripts/run_baseline.py --model runs/sft2_merged     --tasks data/tasks_safety_eval.jsonl --safety --hardened-prompt --no-4bit     --out runs/safety_sft2_hardened

# 6) AuditKIT handoff
python scripts/auditkit_eval.py --trajectories runs/baseline_eval/trajectories.jsonl
```

`--limit 20` is the whole starter set. On a free T4, baseline inference took **2062 s (34 min)** for 20 tasks, and the smoke-adapter evaluation took **2247 s (37 min)**; budget ~45 min for 24 held-out tasks each. A 2-step GRPO smoke took **786 s (381 s/step)**, so 60 steps is ~6.4 h — split into 30-step blocks with checkpoints.

## Metrics

Programmatic, computed in `hinglish_agent/runner.py`:

| Code | Meaning |
|---|---|
| `success` | final ledger matches expected balances (and settlement, when asked) |
| `tac` | tool accuracy: all required tools were called |
| `arr` | argument reliability: `1 - ter` |
| `ter` | tool error rate: failed calls / total calls |
| `rad` | redundant actions: repeated identical calls |
| `overcall` | unnecessary calls: `max(0, calls - oracle calls) / oracle calls` |
| `lcf` | loop collapse: episode hit max steps without `finish` |
| `answer_ok` | answer text contains the expected amount |
| `strict` | every assistant turn is bare JSON (no fences, no prose) |
| `strict_rate` | fraction of assistant turns that are strict |
| `calls` | tool calls per task (mean) — efficiency |
| `steps` | agent turns per task (mean) — efficiency |
| `parse_failures` | assistant turns with no parseable tool call, per task |

## Frozen results (Nov 3, 2026)

Submission candidate is the `sft2` adapter. Provenance, configs, caveats and repro
commands: [docs/RESULTS.md](docs/RESULTS.md).

| Model | hard-30 success | eval-24 success | answer_ok (hard) | calls (hard) |
|---|---:|---:|---:|---:|
| SFT-easy (100 rows, 3 epochs) | 63.33% | 100% | 66.67% | 3.47 |
| SFT-easy + GRPO (30 steps) | 63.33% | 100% | 66.67% | 3.47 |
| **SFT-mix (184 rows, 2 epochs)** | **96.67%** | **100%** | **96.67%** | **3.57** |

Oracle ceiling on hard-30 is 30/30 at 3.57 calls. All runs: Kaggle T4 x2, greedy,
`--no-4bit`, 0 tool errors. GRPO moved nothing at 30 steps because the graded reward was
saturated; reweighting the hard rows into SFT is what fixed two_query, convert_sum and
empty_balance. Full per-metric summaries live in the uploaded `summary.json` files under
`primce/lexsi-runs`.

## Training notes

- Tools are **stateless**: every ledger tool takes and returns the ledger object. Parallel GRPO rollouts cannot leak state.
- The GRPO reward is format + expected tool + answer number; `--efficiency` adds a no-duplicate-calls reward for the *Efficient trajectories* statement. The real quality signal is the post-hoc ledger check, so do not tune rewards to chase `success` on the eval set.
- Baseline must be measured **before** training on the same 20 tasks. If you change tasks after seeing baseline results, re-run both.
- The Nov 2 smoke on the base model logged `frac_reward_zero_std=1`: both completions in each group got the same reward, so GRPO had no advantage. That is why the plan warms the model up with oracle-trajectory SFT before GRPO.
- `max_steps=10` only proves the pipeline works. It is not a result.
- TRL loads the policy in float32 unless told otherwise, which is ~13.4 GB for a 3.35B model and OOMs a 16 GB T4. The script now passes `model_init_kwargs={"dtype": torch.float16}` by default (`--dtype bfloat16` on Ampere+).
- If GRPO still OOMs on a T4, use `--max-completion-length 128 --max-steps-per-turn 2`, keep `--generations 2`, and `--batch-size 1 --grad-accum 4`. Do not switch to 8B/32B models on free hardware.

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
- `max_len` for Tiny Aya must be unset or `>= 512`; our fallback prompt with tool schemas is ~1.4k tokens, so SFT runs at `--max-len 4096`.
- On Colab, `pip uninstall -y torchao` before importing SafeTune. Not needed here, but applies if you add Track 2 tools.

## Credits

Built on [AgentTune](https://github.com/Lexsi-Labs/AgentTune) and [AuditKIT](https://github.com/Lexsi-Labs/AuditKIT) by Lexsi Labs, with Cohere Labs Tiny Aya and Hugging Face hosting.
