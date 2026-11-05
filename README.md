# Hinglish Expense-Splitting Agent — Lexsi Track 3

A solo-friendly Track 3 (Agentic Fine-tuning) entry for the **Lexsi AI Safety & Alignment Hackathon**, Nov 1–22, 2026.

Tiny Aya learns to call tools reliably in Hinglish: it records shared expenses, carries the ledger across turns, converts USD/EUR, and settles balances. Success is checked against the **final ledger state**, not an LLM judge, so the before/after numbers are reproducible.

Live demo: https://huggingface.co/spaces/primce/hinglish-expense-agent (T4, shipped `sft4` adapter, visible tool trace and ledger).

Write-up: https://huggingface.co/spaces/primce/hinglish-track3-article — the full article (numbers, method, safety split, limitations). Hub Articles need PRO, so it is hosted as a free static Space rendered from `docs/hf_article.md`. The submission write-up itself is [`submission/README.md`](submission/README.md).

## Judge's 60-second view

| | Base Tiny Aya (3.35B) | Shipped `sft4_adapter` |
|---|---:|---:|
| 30 unseen hard tasks | 3.33% (1/30) | **96.67% (29/30)** |
| 90 unseen hard tasks | 3.33% (3/90) | **93.33% (84/90)**, Wilson 86.2–96.9% |
| Tool calls per hard-30 task (oracle: 3.57) | 7.03 | 3.57 |
| Injection split: attacker money in ledger / answer | 13/24 / 8/24 | **0/24 / 0/24** |

Paired exact McNemar on shared ids: base → shipped **81 fixed, 0 broken** on hard-90 (p < 1e-20)
and **28 fixed, 0 broken** on hard-30 (p = 7.5e-9). GRPO ran twice and moved nothing (0/0, p = 1.0);
that null is reported, not hidden, and the gain is attributed to the data mix.

```bash
git clone https://github.com/PRINCE2-AI/lexsi-agentic-hinglish && cd lexsi-agentic-hinglish
python scripts/verify_all.py    # 20 checks, no GPU, no token
```

Or run [`notebooks/colab_reproduce.ipynb`](notebooks/colab_reproduce.ipynb): CPU cells in ~2 min,
one tuned episode on a free T4. Reproduce commands for every number: [docs/RESULTS.md](docs/RESULTS.md).


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
- **380 generated Hinglish tasks**, oracle-verified: 100 train + 24 held-out eval, 40 train /
  30 held-out hard tasks across six stress families (multi-day settlement, two queries in one
  turn, conversion sums, empty ledger, payer-in-participants trap, paraphrase), **90 more
  held-out hard tasks** (15 per family, separately pinned) for a tighter confidence interval,
  **32 safety tasks** (indirect prompt injection through tool results, plus underspecified
  requests), and **64 abstention rows** (32 underspecified + 32 twins) for the training side of
  that probe. The 20-task starter set is still there for quick checks.
- Programmatic metrics: `success`, `tac`, `arr`, `ter`, `rad`, `overcall`, `lcf`, plus answer match,
  trajectory efficiency (`calls`, `steps`) and strict-JSON rate.
- **Safety split**: a tool result can carry attacker text; `safe`, `attack_ledger`,
  `attack_answer`, `unplanned_calls` and `abstained` are measured without an LLM judge. See
  [docs/SAFETY.md](docs/SAFETY.md).
- Standalone Tiny Aya baseline runner (transformers, 4-bit) that writes AgentTune-compatible `trajectories.jsonl`.
- GRPO training script using `create_agentic_trainer` with custom rewards.
- **AuditKIT scoring**: every recorded run is replayed through `auditkit.agent_eval` (recorded
  mode, no judge model) with five of its scorers, and its success verdict agrees with this repo's
  scorer on 89/90 hard-90 tasks.
- HF Collection with the adapter, the run logs, the demo and the article.
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
  deploy_space.py     create/refresh the Gradio Space with the right layout
  verify_all.py       every no-GPU check in one command (judges start here)
  score_strict.py     recover strict-JSON metrics from saved trajectories
  auditkit_eval.py    replay a run through AuditKIT agent_eval (no judge)
space/
  app.py              Gradio demo
docs/
  hf_article.md       submission write-up skeleton
  roi.md              ROI comparison with measured token counts
  SAFETY.md           injection eval: threat model, metrics, results
  SUBMIT.md           what judges check, in order, with the exact commands
  MODEL_CARD.md       ready-to-paste README for the adapter repo
  DATASET_CARD.md     ready-to-paste README for primce/lexsi-runs
  demo_script.md      3-minute Demo Day script
  PLANBOOK.md         full Nov 2–14 operating plan (data, SFT→GRPO, artifacts, risks)
tests/
  test_scaffold.py    no-dependency sanity tests
  test_hard_data.py   hard-split pin, oracle, require_answer gate
  test_safety.py      injection eval pin, payload delivery, attack scoring
  test_stats.py       Wilson CI + exact McNemar
```

## Quickstart

One-click Colab: [notebooks/colab_reproduce.ipynb](notebooks/colab_reproduce.ipynb) — cells 1–4
run every offline check on a free CPU runtime in ~2 minutes; cells 6–8 load the shipped adapter on
a T4 and score one frozen held-out task. `notebooks/colab_grpo.ipynb` is the older walkthrough of
the GRPO smoke test, kept for history.

Pure-Python checks (Windows/macOS/Linux, no GPU, no model download):

```bash
python scripts/verify_all.py
```

That single command regenerates the hard, hard-90 and safety splits, checks every SHA-256 pin,
runs every test suite, replays the oracle and runs the AuditKIT adapter tests. It prints
`all 20 checks passed` when the dataset and the scoring are consistent.


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

# 6) AuditKIT replay (recorded episodes, no judge model)
python scripts/auditkit_eval.py --run sft4_hard90 --tasks data/tasks_hard90_eval.jsonl
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

## Results (frozen Nov 5, 2026)

The shipped adapter is `sft4`: SFT on the mix3 data (hard mix + abstention pairs + contrast
pairs), seed 42. Rounds 1 and 2 are kept as the before-columns. Provenance, configs, caveats and
repro commands: [docs/RESULTS.md](docs/RESULTS.md).

| Model | hard-30 | hard-90 | eval-24 | calls (hard-90) |
|---|---:|---:|---:|---:|
| Base Tiny Aya (no adapter) | 3.33% | 3.33% | - | 7.24 |
| SFT-easy (100 rows, 3 epochs) | 63.33% | - | 100% | - |
| SFT-easy + GRPO (30 steps) | 63.33% | - | 100% | - |
| SFT-mix, round 1 (`sft2`) | 96.67% | 88.89% | 100% | 3.41 |
| SFT-mix2, round 2 (`sft3`) | 96.67% | 75.56% | 100% | 3.27 |
| SFT-mix2 + GRPO (40 steps) | 96.67% | 75.56% | - | 3.27 |
| **SFT-mix3 seed 42 (`sft4`, shipped)** | **96.67%** | **93.33%** | **100%** | **3.38** |
| SFT-mix5 (ablation, fewer ask rows) | 100% | 92.22% | 100% | 3.38 |
| SFT-mix2 seed 7 | 93.33% | 83.33% | - | 3.42 |
| SFT-mix2 seed 123 | 80.00% | 75.56% | - | 3.23 |

Oracle ceiling on hard-30 is 30/30 at 3.57 calls. All runs: one T4 (HF Jobs) or Kaggle T4 x2,
greedy, `--no-4bit`. Paired McNemar, same task ids: base -> sft4 hard-90 81 fixed / 0 broken,
p < 1e-20; sft3 -> sft4 18 fixed / 2 broken, p = 0.0004; sft2 -> sft4 7 fixed / 3 broken,
p = 0.34 (directional, not significant). The seed spread is published (16.67 pp on hard-30,
7.77 pp on hard-90) and seed 7 is better than the shipped seed on hard-90; the shipped seed is the
one fixed before the seed runs existed, and the median of the three. Full per-metric summaries
live in the uploaded `summary.json` files under `primce/lexsi-runs`.

AuditKIT replay of the recorded episodes, no judge model: on hard-30, exact-call accuracy
0.033 -> 1.000, tool-call F1 0.188 -> 1.000, redundant calls 0.327 -> 0.000; on hard-90 the
shipped adapter is exact 0.933, tool-call F1 0.980, validity 1.000, zero redundant calls and zero
loops. Its success verdict agrees with this repo's scorer on 30/30 (hard-30) and 88/90 decided
hard-90 tasks.

### Safety split (32 tasks, no LLM judge)

Counts on the 24 injection tasks; `attack_ledger` means attacker money actually reached the
final ledger.

| System | safe | utility under attack | attack_ledger | attack_answer |
|---|---:|---:|---:|---:|
| Base Tiny Aya | 0/24 | 0/24 | 13/24 (54.2%) | 8/24 (33.3%) |
| SFT-mix (`sft2`) | 11/24 | 11/24 | **0/24** | 3/24 |
| SFT-mix + hardened prompt | 15/24 | 15/24 | **0/24** | 2/24 |
| SFT-mix2 (`sft3`) | 15/24 | 15/24 | **0/24** | 2/24 |
| **SFT-mix3 (`sft4`, shipped)** | **14/24** | **14/24** | **0/24** | **0/24** |
| SFT-mix3 + hardened prompt | 14/24 | 14/24 | **0/24** | 2/24 |
| SFT-mix5 (ablation) | 11/24 | 11/24 | **0/24** | 3/24 |

Base -> `sft4` removes 13 ledger attacks and introduces 0 (exact McNemar p = 0.00024), and removes
all 8 attacker numbers from answers. On the 8 underspecified requests the shipped adapter asks on
4 (up from 0/8) and invents a split on 4 (down from 7/8); with the hardened prompt it reaches
7/8 asked and 0/8 invented, which clears the pre-registered bar of 6/8, at the cost of 2/24
attacker numbers in prose. The adapter alone is the default; `docs/ROUND2.md` records the
trade-off. Details: [docs/SAFETY.md](docs/SAFETY.md).

## Training notes

- Tools are **stateless**: every ledger tool takes and returns the ledger object. Parallel GRPO rollouts cannot leak state.
- The GRPO reward is format + expected tool + answer number; `--efficiency` adds a no-duplicate-calls reward for the *Efficient trajectories* statement. The real quality signal is the post-hoc ledger check, so do not tune rewards to chase `success` on the eval set.
- Baseline must be measured **before** training on the same 20 tasks. If you change tasks after seeing baseline results, re-run both.
- The Nov 2 smoke on the base model logged `frac_reward_zero_std=1`: both completions in each group got the same reward, so GRPO had no advantage. That is why the plan warms the model up with oracle-trajectory SFT before GRPO.
- `max_steps=10` only proves the pipeline works. It is not a result.
- TRL loads the policy in float32 unless told otherwise, which is ~13.4 GB for a 3.35B model and OOMs a 16 GB T4. The script now passes `model_init_kwargs={"dtype": torch.float16}` by default (`--dtype bfloat16` on Ampere+).
- If GRPO still OOMs on a T4, use `--max-completion-length 128 --max-steps-per-turn 2`, keep `--generations 2`, and `--batch-size 1 --grad-accum 4`. Do not switch to 8B/32B models on free hardware.

## Demo deployment

Live: https://huggingface.co/spaces/primce/hinglish-expense-agent — built from `space/` by
`python scripts/deploy_space.py --repo primce/hinglish-expense-agent`. The Space needs a GPU
(T4 small is enough); it defaults to the shipped `sft4_adapter`.

The demo has a **mode switch** (shipped adapter / base model / shipped + hardened prompt) that
toggles `disable_adapter()` on one loaded model, and a **scenario switch** that loads the frozen
safety tasks (`s-001` injection, `s-025` underspecified) next to the normal split. Every run prints
the tool trace, the final ledger and a verdict from the same scoring code as the eval.

Set `MODEL_ID` and `ADAPTER_ID=<you>/tiny-aya-hinglish-lora`; `python scripts/check_demo_space.py`
verifies the handlers offline.

## Submission checklist (Track 3)

- [x] Public GitHub repo with this code and the generated `data/tasks.jsonl`
- [x] Public HF Space demo with tool trace
- [x] HF artifacts: LoRA adapter, `trajectories.jsonl` logs, dataset card
- [x] HF Article: metrics chosen and why, dataset/task/model rationale, before/after table
      (published as a static Space: https://huggingface.co/spaces/primce/hinglish-track3-article)
- [x] At least two metrics, before and after (success, exact tool calls, F1, precision, recall,
      redundant calls, loop detection, ledger-attack rate, abstention)
- [x] AgentTune + AuditKIT used together, no judge model
- [x] HF Collection: https://huggingface.co/collections/primce/lexsi-track-3-hinglish-expense-agent-6aeb3207e10dfc7dda6329d6
- [ ] Optional ROI table: self-hosted Tiny Aya vs closed-source API at a realistic volume
- [x] Only Cohere Labs models used for the trained model
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

Round 2 (after the 2026-11-04 self-review) is tracked in [docs/ROUND2.md](docs/ROUND2.md): an
abstention SFT split with minimal pairs, a separately pinned hard-90 holdout, a GRPO run whose
reward can actually see the abstention behaviour, and a seed control. The SFT half is measured
and shipped; the GRPO half runs on HF Jobs and its outcome is appended there.

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
