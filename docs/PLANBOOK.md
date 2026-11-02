# Plan Book — Lexsi Track 3, solo run

**Owner:** Prince Maurya (@PRINCE2-AI)
**Deadline:** submit **Nov 14, 2026, end of day**. Demo Day Nov 22 if shortlisted.
**Track:** 3 — Agentic Fine-tuning. Model: `CohereLabs/tiny-aya-global` (3.35B, gated).
**Status on Nov 2:** scaffold complete, oracle 20/20, **baseline measured at 25% success (5/20)**, GRPO smoke passed on Kaggle but is compute-bound (381 s/step) and reward-dead on the base model.

**Measured baseline (Kaggle T4, 4-bit, greedy, 20 tasks, 2062 s):**

| Metric | Baseline |
|---|---:|
| success | 25.0% (5/20) |
| tac | 45.0% |
| arr / ter | 88.16% / 11.84% |
| rad | 29.84% |
| lcf | 70.0% |
| answer_ok | 10.0% |
| calls / steps | 9.25 / 9.60 |

Oracle reference: 100%, 3.5 calls, 3.5 steps. The baseline solves some tasks only by accident (e.g. `t13`: 22 calls, 4 errors, never called `finish`, but the ledger ended correct) — efficiency metrics exist precisely to expose that. `strict` was added after this run; recover it with `scripts/score_strict.py`.

**Pilot caveat:** this run used the older prompt rule that told the model to call `convert_currency` before USD/EUR arithmetic, even though the ledger tools already convert. That rule is now clarified in `prompts.py`. Re-run the baseline once with the current prompt before freezing the final table; keep these numbers only as the pilot.

> **Rule of the run:** every number in the write-up comes from `runs/*/summary.json`. No estimates, no vibes.

---

## 0. One-page summary

| Item | State |
|---|---|
| Repo | https://github.com/PRINCE2-AI/lexsi-agentic-hinglish @ `aa710b5` |
| Local checks | `oracle check: 20/20 success, 0 tool errors`, scaffold tests pass |
| Dataset | 20 hand-written Hinglish tasks, oracle-verified, deterministic ledger |
| Baseline | **not run yet** (Colab quota died; Kaggle is the machine now) |
| Training | GRPO script fixed for T4 (fp16 + grad checkpointing), smoke test not yet confirmed on Kaggle |
| Winning wedge | **oracle-trajectory SFT warm-start, then GRPO** + strict-JSON metric |
| Must-have artifacts | adapter, dataset, trajectories, GPU Space, HF Collection, article |
| Hard freeze | Nov 10 results frozen; Nov 13 submit; Nov 14 buffer only |

**Why SFT first:** the closest public Track 3 attempt (ToolBoundary) trained 12 examples for 30 GRPO steps and got **0/24 strict JSON before and after**. GRPO on a 3.35B model that cannot emit the format at all has almost no reward signal to climb. SFT on oracle traces installs the format and the ledger-copy behaviour; GRPO then optimizes success and efficiency. Report both stages separately so the contribution is visible.

---

## 1. How judging works (what to optimize)

From the kickoff slides:

| Criterion | Weight | What actually moves it |
|---|---:|---|
| Write-up clarity | /10 | problem → method → numbers → limitations, all in one article |
| Practical usefulness | /10 | working Space, real Hinglish task, deterministic answers, ROI |
| Code / notebook quality | /10 | one-click notebook, tests, seeded data generation, no LLM judge |
| Overall review | /100 | novelty + rigor + artifacts + honesty about failures |

Submission checklist (all mandatory):

- [ ] Public GitHub repo with code **and** generated dataset
- [ ] Public HF Space demo with visible tool trace
- [ ] Model artifact public (LoRA adapter) in an **HF Collection**
- [ ] Data artifact public (dataset + `trajectories.jsonl`) in an **HF Collection**
- [ ] Experiments run, findings, and what else was tested (ablations)
- [ ] Approach + before/after numbers
- [ ] Only Cohere Labs models for the trained model

**Judge-friendly properties we already have:** deterministic outcome check, no API keys needed to reproduce, fixed FX rates, fixed task set. Keep them.

---

## 2. Current assets (do not rebuild)

```
hinglish_agent/     ledger, tools, prompts, parser, runner, metrics
data/build_tasks.py 20 scenarios -> data/tasks.jsonl + oracle verification
scripts/            run_oracle_eval.py, run_baseline.py, train_grpo.py, auditkit_eval.py
space/app.py        Gradio demo, lazy load, 4-bit, ZeroGPU hook
docs/               hf_article.md, roi.md, demo_script.md, this file
notebooks/colab_grpo.ipynb  fresh clone + assert + baseline + smoke test
```

Known-good commands (already verified):

```bash
python data/build_tasks.py          # 20/20 success, 0 tool errors
python scripts/run_oracle_eval.py
python tests/test_scaffold.py
```

Metrics already computed per episode: `success`, `tac`, `arr`, `ter`, `rad`, `lcf`, `answer_ok`, `strict`, `strict_rate`, `ledger_ok`, `settlement_ok`, `calls`, `errors`, `steps`.

Built on Nov 2: `strict` metric + `scripts/score_strict.py`, and `scripts/train_sft.py`.

**Still missing:**

1. `data/build_generated.py` — 100 train + 24 held-out eval tasks, oracle-verified, deduped.
2. A Kaggle SFT smoke result, then the full SFT run.
3. HF Collection, model/dataset/Space cards, article numbers.
4. Ablation table + error taxonomy.

---

## 3. Architecture (what the model actually does)

One episode:

```
system prompt (tool schemas + rules)
  -> user: Hinglish task
  -> assistant: [{"tool_name": "...", "parameters": {...}}]     # strict JSON only
  -> tool executes, returns {"ok": true, "ledger": {...}, ...}
  -> user: {"tool_results": [...]}                              # ledger copied back
  -> ... repeat ...
  -> assistant: [{"tool_name": "finish", "parameters": {"answer": "..."}}]
```

Ledger is **explicit state carried by the model** in integer paise. Tools are stateless, so parallel GRPO rollouts cannot leak state. This is the difficulty: the model must copy a JSON object exactly across turns, in Hinglish, with currency conversion.

Scoring is post-hoc and deterministic:

- `ledger_ok`: final `balances_cents` match the oracle.
- `settlement_ok`: `settle_ledger` transfers match.
- `success`: ledger + settlement for expense tasks; `answer_ok` for conversion tasks.
- Efficiency: `calls`, `steps`, `rad`, `lcf`.

---

## 4. Compute plan and limits

| Resource | Limit | Use |
|---|---|---|
| Kaggle 2×T4 | 30 GPU-h/week, ~9–12 h sessions | SFT, GRPO, ablations. Use **one** T4 (`CUDA_VISIBLE_DEVICES=0`). |
| Colab free T4 | quota exhausted now, resets later | fallback only |
| HF credits | $20, expire ~Dec 1 | Space/ZeroGPU or a short endpoint, **not** a 3-day training run |
| Local Windows | no NVIDIA GPU | oracle/tests/docs only |

Kaggle habits that save the run:

- Save checkpoints every 20 steps and upload the adapter to HF after each phase. Sessions die; `/kaggle/working` does not survive forever.
- Never run two notebooks on the same account at once.
- Set `%env CUDA_VISIBLE_DEVICES=0` before importing torch.
- `%env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (the script already sets it as a default).

**Measured throughput (Nov 2, Kaggle T4, one GPU):**

| Job | Time | Implication |
|---|---:|---|
| Baseline, 20 tasks, 4-bit, ≤8 steps | 2062 s (34 min) | eval is not free; budget ~35 min per 20-task pass |
| GRPO smoke, 2 steps, 2 generations, 256 tokens, ≤3 turns | 786 s (~13 min), 381 s/step | 120 steps ≈ **12.7 h**, longer than a safe Kaggle session |
| GRPO, completions ~457 tokens mean | — | SFT should terminate in fewer turns and cut step time |

Revised limits:

- A 60-step GRPO run is the default (≈6.4 h at baseline speed; ~2.5–3.5 h if SFT cuts turns).
- If step time stays above 250 s, do **two 30-step sessions with checkpoints**, not one 60-step run.
- 120 steps only if measured step time drops below 150 s after SFT.


---

## 5. Data plan: 20 → 100 train + 24 eval

### 5.1 Why expand

20 tasks cannot support an ablation table. 100 train / 24 held-out is enough for stable percentages at this scale, and still cheap: SFT ~300 steps, GRPO rollouts ~100 prompts.

### 5.2 Split

| File | Rows | Purpose | Rule |
|---|---:|---|---|
| `data/tasks.jsonl` | 20 | public starter set, continuity | keep frozen |
| `data/tasks_train.jsonl` | 100 | training only | generated, seed 7 |
| `data/tasks_eval.jsonl` | 24 | held-out evaluation | generated, seed 99, disjoint texts |

Never evaluate on `tasks_train.jsonl`. Report three columns if time allows: starter-20, held-out-24, train-100 (train number is diagnostic only; it is not a generalization claim).

### 5.3 Families and counts

| Family | Train | Eval | Example answer type |
|---|---:|---:|---|
| Single expense → balance | 30 | 6 | balance |
| 2–4 expenses → balance | 25 | 6 | balance |
| 2–4 expenses → settle | 25 | 6 | settlement |
| Currency conversion | 10 | 3 | converted amount |
| Edge cases | 10 | 3 | balance/settle |
| **Total** | **100** | **24** | |

Edge cases to include (deterministic, oracle-safe):

- payer not in participants (Rahul pays for Priya/Amit)
- queried person not in any expense (balance 0.00)
- self-only expense (payer is sole participant)
- amounts with paise remainders (e.g. 1001 INR / 3, 47 USD / 4, 19 EUR / 6)
- 5–6 participants
- two expenses that cancel out (settle → "sab settled")
- USD→INR, EUR→INR, INR→USD
- duplicate names across expenses, different participant sets

Do **not** include ambiguous tasks. Every task needs one oracle answer. If the scenario cannot be scored mechanically, it does not belong in the dataset.

### 5.4 Generator design

New file `data/build_generated.py`:

- Reuse `empty_ledger`, `execute_tool`, `balance_inr`, `settle_ledger`, `public_balances` from `hinglish_agent`.
- Reuse the `scenario -> task` logic from `build_tasks.py`; extract it into `data/task_builder.py` so both scripts share one code path (do not copy-paste the answer formatting).
- 3 Hinglish phrasings per family, chosen by `random.Random(seed)`.
- Name pool: the six names in `hinglish_agent/ledger.py` (`Rahul, Priya, Amit, Neha, Sara, Vikram`). Adding names means editing `PEOPLE` too; do not generate a name the ledger rejects.
- Currency pool: INR, USD, EUR; amounts from a fixed list (integers + paise-friendly values).
- Deduplicate by normalised task text; reject collisions.
- Run `run_oracle` + `score_episode` on every row; abort the build on any `success != 1` or `ter != 0`.
- Write JSONL with stable IDs `g-train-001`, `g-eval-001`.

Commands:

```bash
python data/build_generated.py --split train --n 100 --seed 7
python data/build_generated.py --split eval  --n 24  --seed 99
python scripts/run_oracle_eval.py --tasks data/tasks_train.jsonl
python scripts/run_oracle_eval.py --tasks data/tasks_eval.jsonl
```

Expected: all rows `success=1`, `ter=0`. If one fails, fix the generator, not the JSONL.

### 5.5 Leakage checks

```bash
python - <<'PY'
import json
def texts(p): return [json.loads(l)["task"].strip().lower() for l in open(p, encoding="utf-8")]
tr, ev = texts("data/tasks_train.jsonl"), texts("data/tasks_eval.jsonl")
print("overlap:", len(set(tr) & set(ev)))
assert not (set(tr) & set(ev))
PY
```

Also eyeball 10 random eval tasks for Hinglish quality. Native phrasing matters for the write-up.

---

## 6. Measurement plan

### 6.1 Strict JSON vs fuzzy

`parse_calls` is deliberately forgiving (code fences, prose around JSON). That hides format failure. Add both:

- `strict`: 1.0 only if **every assistant turn** in the episode is bare JSON (starts with `[`/`{`, ends with `]`/`}`, `json.loads` succeeds, every item has `tool_name` + `parameters`), no fences/prose.
- `strict_rate`: fraction of assistant turns that are strict.
- Existing runs: `python scripts/score_strict.py --trajectories runs/baseline/trajectories.jsonl --summary runs/baseline/summary.json` recovers the number without a GPU.
- Keep `success` on fuzzy parsing so a correct-but-chatty model still gets credit for the task.

Where:

- `hinglish_agent/runner.py`: compute from `episode["messages"]` inside `score_episode`.
- `hinglish_agent/metrics.py`: add `strict` and `strict_rate` to `METRIC_KEYS` and the markdown table.
- `tests/test_scaffold.py`: one strict-true case, one fenced-JSON case, one prose case.

Why this matters: it lets the write-up say “strict JSON 0/24 → N/24, fuzzy success X% → Y%” instead of only the flattering number.

### 6.2 Evaluation protocol

- Baseline: `--limit 0 --tasks data/tasks_eval.jsonl --out runs/baseline_eval`.
- Tuned: same tasks, `--adapter <adapter> --out runs/tuned_eval`.
- Decoding: greedy (`do_sample=False`) in `run_baseline.py`; report it. Training uses sampling; evaluation does not.
- Report counts too (`n=24`), not only percentages.
- Re-run baseline once after any task-set change. Never compare a number measured on a different set.
- Do not tune rewards against eval. Use train-100 for training, eval-24 for reporting.

### 6.3 AuditKIT handoff

```bash
python scripts/auditkit_eval.py --trajectories runs/tuned_eval/trajectories.jsonl --model hf:runs/grpo_tiny_aya
```

Paste the first episode check into the write-up appendix. If AuditKIT scoring needs a judge, report it as a secondary diagnostic only; the ledger remains the primary metric.

---

## 7. Training plan

### Phase 0 — baseline (Nov 2) — **done**

```bash
python scripts/run_baseline.py --model CohereLabs/tiny-aya-global --limit 20
```

Result (Kaggle T4, 2062 s): `success=0.25`, `tac=0.45`, `ter=0.1184`, `rad=0.2984`, `lcf=0.70`, `answer_ok=0.10`, `calls=9.25`, `steps=9.60`. Five tasks pass: `t06`, `t08`, `t09`, `t12`, `t13`; only `t06` is a conversion, the rest pass on the ledger. `t13` needed 22 calls and 4 errors to get there.

GRPO smoke (2 steps, 786 s): reward 0.083, `frac_reward_zero_std=1`. Format/prose/loop failures dominate the completions. Do not start a long GRPO run from this base.

Then run the held-out set once after data generation:

```bash
python scripts/run_baseline.py --model CohereLabs/tiny-aya-global \
  --tasks data/tasks_eval.jsonl --out runs/baseline_eval
```

Record: success, tac, ter, rad, lcf, calls, steps, strict, seconds. Classify the first 10 failures into: no JSON, wrong tool, bad arguments, wrong ledger copy, wrong split, wrong number, loop.

### Phase 1 — SFT warm-start (Nov 4)

New `scripts/train_sft.py`. Target format must match inference exactly.

For each training task, generate the full oracle conversation:

1. `system` = `SYSTEM_PROMPT`, `user` = task.
2. For each oracle action: assistant = `format_call(action)`; then `user` = `{"tool_results": [result]}` using the real tool result.
3. Final assistant = `format_call({"tool_name": "finish", "parameters": {"answer": task["answer_text"]}})`.

Training details:

| Setting | Value | Reason |
|---|---|---|
| Base | `CohereLabs/tiny-aya-global` | required model family |
| Precision | fp16 | 4-bit merge is annoying; 6.7 GB fits one T4 with checkpointing |
| LoRA | r=16, alpha=32, all-linear, dropout 0.05 | same shape as GRPO stage |
| Seq length | 4096 | the system prompt alone is 4,966 chars (~1.4k tokens); 4-expense tasks reach ~7.8k chars. Skip, never truncate, an overlong target. |
| Batch | 1 × grad-accum 8 | T4-safe |
| LR | 2e-4, warmup 10 | standard LoRA SFT; only ~40 optimizer steps on 100 rows |
| Epochs | 3 | ~300 steps for 100 rows |
| Loss | assistant tokens only (`-100` on system/user) | learn format + ledger copy, not prompts |
| Save | adapter + merged fp16 model | GRPO needs a plain model path |

Implementation note: tokenise the full chat once, then for every assistant index compute the prefix length with `add_generation_prompt=True` and the end of that turn; copy labels only for the assistant span, `-100` everywhere else. Fail loudly if the template is not prefix-stable, and skip (never truncate) a row over `--max-len`. This avoids relying on `{% generation %}` markers that the Cohere template may not provide.

Commands:

```bash
python scripts/train_sft.py --check
python scripts/train_sft.py --limit 20 --epochs 1 --out runs/sft_smoke --no-merge   # ~20 min smoke
python scripts/train_sft.py --tasks data/tasks_train.jsonl --epochs 3 \
  --out runs/sft --merge-out runs/sft_merged
python scripts/run_baseline.py --model CohereLabs/tiny-aya-global \
  --adapter runs/sft --tasks data/tasks_eval.jsonl --out runs/sft_eval
```

If the merged model is saved as `runs/sft_merged`, the eval command uses `--model runs/sft_merged` instead of `--adapter` (or keep the adapter for eval; both should match — verify once).

**Gate B:** SFT success on eval-24.

- `>= 70%` → proceed to GRPO for efficiency/robustness, not for basic format.
- `40–70%` → inspect failures; fix data coverage (likely participants/ledger copy) before GRPO.
- `< 40%` → do not start a long GRPO run. Fix SFT (more epochs, more edge cases, lower LR) first.

### Phase 2 — GRPO from the SFT model (Nov 6–7)

```bash
python scripts/train_grpo.py \
  --model runs/sft_merged \
  --tasks data/tasks_train.jsonl \
  --out runs/grpo_sft \
  --max-steps 60 --efficiency
```

Keep the current safe config unless it OOMs: batch 1, grad-accum 4, generations 2, completion 256, 3 tool steps per turn, fp16, LoRA r=16. **Why 60, not 120:** the Nov 2 smoke measured 381 s/step, so 120 steps is ~12.7 h. SFT should shorten trajectories; if step time falls below 150 s, extend to 120. If it stays above 250 s, run 30 steps, save, resume for another 30.

The Nov 2 smoke also reported `frac_reward_zero_std=1` on the base model: every completion group had identical reward, so GRPO had no advantage to learn from. That is direct evidence for the SFT warm-start, and it goes in the write-up.

Checkpoint discipline:

- After 20 steps, confirm the adapter directory exists and loss/reward are logged.
- After each 30-step block, evaluate and upload the adapter to HF.
- If the session dies, resume from the last checkpoint or restart from SFT (SFT is cheap).

### Phase 3 — ablations (Nov 8–9)

Time-box each arm. The point is a table, not perfection.

| Arm | Init | GRPO | Efficiency reward | Purpose |
|---|---|---|---|---|
| A0 | base | no | — | baseline |
| A1 | SFT | no | — | does SFT alone fix format? |
| A2 | base | 20 steps | no | can GRPO work without warm-start? |
| A3 | SFT | 60 steps | no | main result |
| A4 | SFT | 60 steps | yes | efficiency statement |

If GPU time is tight, run A0, A1, A3, A4. A2 is the most interesting negative control but the most expensive at 381 s/step; 20 steps (~2.1 h) is enough to show whether format reward ever fires.

Report every arm on eval-24, same decoding. A4 ships only if it does not lower success versus A3.

### Phase 4 — freeze (Nov 10)

- Pick the best checkpoint by eval success, tie-broken by lower `ter`, then lower `calls`.
- Re-run final eval twice (greedy is deterministic, so this is a sanity check, not a seed study).
- Write the error taxonomy with 3 concrete episodes (task id + what went wrong).
- No training after Nov 10. Only docs, Space, artifacts.

---

## 8. Day-by-day schedule (Nov 2–14)

### Nov 2 (today) — Kaggle baseline + failure taxonomy — **done**

- [x] Kaggle notebook: GPU T4×2, Internet On, `CUDA_VISIBLE_DEVICES=0`.
- [x] Clone + install + login.
- [x] Baseline: `success=0.25`, 5/20, 2062 s.
- [x] GRPO 2-step smoke: 786 s, reward 0.083, `frac_reward_zero_std=1`.
- [ ] `python scripts/score_strict.py --trajectories runs/baseline/trajectories.jsonl --summary runs/baseline/summary.json` (fills the missing strict column, no GPU).
- [ ] Save 5 raw failure examples and classify them (no JSON / prose / wrong ledger / loop / wrong number).
- [ ] Fill the baseline column in a scratch table (not README yet).

Kaggle cell block:

```python
!nvidia-smi
%cd /kaggle/working
![ -d lexsi-agentic-hinglish ] && (cd lexsi-agentic-hinglish && git pull) || git clone https://github.com/PRINCE2-AI/lexsi-agentic-hinglish.git
%cd lexsi-agentic-hinglish
!pip install -q -r requirements.txt
```

Restart the kernel after install, then:

```python
from huggingface_hub import login
login()  # paste the Write token here only
```

```python
from pathlib import Path
assert "model_init_kwargs" in Path("scripts/train_grpo.py").read_text()
!git log --oneline -1
!python scripts/run_oracle_eval.py
```

**Gate A:** if baseline success is 0%, that is expected for a 3.35B model with a fallback tool prompt; SFT is mandatory. If it is already >50%, still run SFT but prioritize efficiency and robustness in the story.

### Nov 3 — data expansion + baseline re-measure

- [ ] Extract shared scenario→task builder; add `data/build_generated.py`.
- [ ] Generate 100 train + 24 eval; oracle-verify both; leakage check.
- [ ] Spot-check 10 eval tasks for natural Hinglish.
- [ ] Re-run the baseline with the clarified prompt: starter-20 and eval-24; this is the final "before" column.
- [ ] Commit + push (approval flow: report → “ha” → push).

### Nov 4 — SFT

- [ ] Write `scripts/train_sft.py`; dry-run on 8 examples for 2 steps.
- [ ] Full SFT (3 epochs). Save adapter + merged model.
- [ ] Evaluate on eval-24; save `runs/sft_eval/summary.json`.

### Nov 5 — SFT analysis + GRPO smoke

- [ ] Error taxonomy after SFT (format vs ledger vs arithmetic).
- [ ] If Gate B passes: `python scripts/train_grpo.py --model runs/sft_merged --max-steps 2` then `--max-steps 10`.
- [ ] Confirm no OOM and reward is non-zero.
- [ ] Upload SFT adapter to HF (private first is fine; make public before submission).

### Nov 6–7 — main GRPO

- [ ] 60-step run from SFT, `--efficiency`, train-100 (two 30-step blocks if step time > 250 s).
- [ ] Checkpoint every ~20 steps; upload adapter at the end.
- [ ] Evaluate eval-24; save `runs/tuned_eval/summary.json`.
- [ ] If the session dies: resume or rerun; SFT is intact.

### Nov 8 — ablations

- [ ] A2: base + GRPO 20 steps (negative control), eval.
- [ ] A3 vs A4: efficiency reward on/off, 60 steps each if time allows; otherwise A3 from Nov 6–7 plus a 30-step A4.
- [ ] Build the ablation table.

### Nov 9 — second seed / final run

- [ ] Only if a clear win: one more run with a different seed or slightly higher generations.
- [ ] Otherwise stop training and start packaging.
- [ ] Upload final adapter + tokenizer.

### Nov 10 — freeze results

- [ ] Final eval on starter-20 and eval-24; both summary files committed.
- [ ] AuditKIT run on tuned trajectories.
- [ ] Error analysis with 3 episode examples.
- [ ] Fill README + `docs/hf_article.md` tables.

### Nov 11 — HF artifacts + Space

- [ ] Model card, dataset card, Collection.
- [ ] Deploy Space; test with base and tuned adapter.
- [ ] Add links to README.

### Nov 12 — write-up

- [ ] Finish article: method, results, ablations, limitations, reproduce.
- [ ] 60–90 s screen recording of the Space (tool trace visible).
- [ ] Proofread for claims that the numbers do not support.

### Nov 13 — submit early

- [ ] Fill the submission form from Discord.
- [ ] Verify every link is public in an incognito window.
- [ ] Submit. Do not wait for Nov 14.

### Nov 14 — buffer only

- [ ] Fix broken links or a Space crash. No new experiments.

---

## 9. HF artifacts and Collection

### 9.1 Repos to create

```bash
hf auth login
hf repo create primce/tiny-aya-hinglish-lora --repo-type model
hf repo create primce/hinglish-expense-split --repo-type dataset
```

Upload:

```bash
hf upload primce/tiny-aya-hinglish-lora runs/grpo_sft --repo-type model
hf upload primce/hinglish-expense-split data/tasks_train.jsonl tasks/train.jsonl --repo-type dataset
hf upload primce/hinglish-expense-split data/tasks_eval.jsonl tasks/eval.jsonl --repo-type dataset
hf upload primce/hinglish-expense-split runs/baseline_eval/trajectories.jsonl logs/baseline_eval.jsonl --repo-type dataset
hf upload primce/hinglish-expense-split runs/tuned_eval/trajectories.jsonl logs/tuned_eval.jsonl --repo-type dataset
```

### 9.2 Model card (minimum)

- Base model + adapter type (LoRA r=16, alpha=32, all-linear).
- Training stages: SFT (epochs, LR, rows) → GRPO (steps, rewards, generations).
- Eval table copied from `summary.json`, with `n` and decoding settings.
- Intended use + limitations (fixed FX, six fixed names, ledger copy burden).
- Links: GitHub, dataset, Space, article.
- License: inherit the Cohere Labs Tiny Aya terms for the adapter; state it explicitly.

### 9.3 Dataset card

- 100 train / 24 eval / 20 starter rows; generation script + seed.
- Deterministic oracle; no human labels.
- Fixed FX rates (1 USD = 83 INR, 1 EUR = 90 INR) — record the exact constants from `ledger.py`, do not approximate in prose.
- License: pick one and be consistent (suggest CC-BY-4.0 for data, Apache-2.0 for code; verify Tiny Aya terms for the adapter).

### 9.4 Collection

HF UI: Profile → **New collection** → name “Lexsi Track 3 — Hinglish Expense Agent” → add the model, dataset, and Space. Pin the collection on the HF profile. Both the model and data artifacts must be in a Collection per the submission checklist.

### 9.5 Space

- New Space → Gradio → hardware with GPU (ZeroGPU if available).
- Files at root: `app.py` (copy of `space/app.py`), `hinglish_agent/`, `requirements.txt` (gradio, torch, transformers, peft, bitsandbytes, accelerate, sentencepiece).
- Variables: `MODEL_ID=CohereLabs/tiny-aya-global`, `ADAPTER_ID=primce/tiny-aya-hinglish-lora`, `MAX_NEW_TOKENS=256`.
- Secret: `HF_TOKEN` (read token) so the gated base model downloads.
- Space README frontmatter: `sdk: gradio`, `app_file: app.py`, `pinned: false`.
- Test: one balance task, one settlement task, one conversion task; confirm the tool trace is visible and the final ledger matches.

Optional but strong: a base-model Space next to the tuned one, or a “compare” toggle. Only do this after the tuned Space works.

---

## 10. Write-up structure (article)

Follow `docs/hf_article.md`, fill every bracket:

1. **Title + TL;DR** — 3 bullets with real numbers and `n`.
2. **Problem statement** — quote both Track 3 statements; map each clause to a metric.
3. **Why Hinglish** — multilingual gap + deterministic ledger.
4. **Method** — tools, data, baseline, SFT, GRPO, rewards.
5. **Metrics** — table; explain strict vs fuzzy and why both are reported.
6. **Results** — before/after table, ablation table, counts.
7. **Error analysis** — 3 concrete failures, what changed, what did not.
8. **Limitations** — 20/24/100 samples, fixed FX, fixed names, ledger-copy burden.
9. **Reproduce** — exact commands, compute used.
10. **Artifacts** — GitHub, adapter, dataset, trajectories, Space, Collection.
11. **Credits** — Lexsi Labs, Cohere Labs, HF.

Write-up rules:

- Never claim “state of the art”. Claim “measurable improvement on a held-out set of 24”.
- If a metric got worse, say so and explain. Judges reward honesty more than a clean story.
- Do not compare against competitors by name; compare against your own baseline and the published failure mode.
- Every table row must be traceable to a file path.

---

## 11. Demo Day script (if shortlisted)

Already in `docs/demo_script.md`. Keep the 3-minute shape:

1. 0:00–0:20 — problem: Hinglish expense splitting, ledger must survive turns.
2. 0:20–0:50 — baseline fails: show one raw output (no JSON / wrong ledger).
3. 0:50–1:40 — tuned Space: live task, visible tool trace, final answer.
4. 1:40–2:20 — numbers: success, ter, calls/steps before/after.
5. 2:20–2:50 — how: oracle SFT + GRPO, strict-JSON metric.
6. 2:50–3:00 — artifacts + repo.

Rehearse once with a timer. Have a recorded backup in case the Space is cold.

---

## 12. Risk register

| Risk | Trigger | Response |
|---|---|---|
| CUDA OOM | `torch.OutOfMemoryError` on T4 | Restart runtime; verify `model_init_kwargs` present; `--max-completion-length 128 --max-steps-per-turn 2`, generations 2, batch 1, grad-accum 4. Never switch to 8B on free hardware. |
| Stale code | traceback line numbers do not match repo | `git pull`, `git log --oneline -1`, assert `model_init_kwargs` in script; fresh clone dir. |
| GPU quota | Kaggle refuses / Colab “cannot connect” | Kaggle weekly reset; run only one notebook; move eval to CPU only if unavoidable (slow). |
| Session timeout | Kaggle stops mid-run | Checkpoint every 20 steps; upload adapter per phase; SFT is re-runnable in ~1–2 h. |
| Gated model 401 | load fails | Accept licence on the model page; `hf auth login`; Kaggle secret `HF_TOKEN`. |
| SFT does not learn format | success < 40% after 3 epochs | More epochs, lower LR, add format-only examples, check label masking, confirm EOS. |
| GRPO reward stays 0 | no improvement in 20 steps | Start from SFT, not base; verify `format_reward` fires on SFT outputs; shorten completions. |
| GRPO worse than SFT | success drops | Ship SFT-only as tuned model; report GRPO as a negative ablation. This is a valid result. |
| Time overrun | Nov 10 arrives mid-experiment | Freeze; ship the best measured checkpoint; write-up > extra run. |
| Space cold / OOM | demo fails | ZeroGPU + `MAX_NEW_TOKENS=256`; pre-warm before demo; recorded backup. |
| Data bug | oracle gate fails | Fix generator; never hand-edit `tasks.jsonl`. |
| Personal data leak | logs contain names/emails | Only synthetic names; scan trajectories before upload. |
| Credits burn | HF endpoint running idle | Delete endpoints; credits are for the Space, not training. |

---

## 13. Daily scorecard (copy into a scratch file)

| Date | GPU-h | Runs completed | Eval success (n) | Best ter | Best calls | Decision |
|---|---|---:|---|---:|---:|---|
| Nov 2 | | baseline | | | | |
| Nov 3 | | data build | — | — | — | |
| Nov 4 | | SFT | | | | |
| Nov 5 | | SFT eval + smoke | | | | |
| Nov 6 | | GRPO 60 | | | | |
| Nov 7 | | GRPO eval | | | | |
| Nov 8 | | ablations | | | | |
| Nov 9 | | final run | | | | |
| Nov 10 | | freeze | | | | |

Rules: one row per day; if a run failed, write the error class, not “failed”.

---

## 14. Command reference

Local (no GPU):

```bash
python data/build_tasks.py
python scripts/run_oracle_eval.py
python tests/test_scaffold.py
python scripts/score_strict.py --trajectories runs/baseline/trajectories.jsonl \
  --summary runs/baseline/summary.json
```

Kaggle bootstrap:

```bash
git clone https://github.com/PRINCE2-AI/lexsi-agentic-hinglish.git
cd lexsi-agentic-hinglish
pip install -q -r requirements.txt
hf auth login
```

Baseline / eval:

```bash
python scripts/run_baseline.py --model CohereLabs/tiny-aya-global --limit 20
python scripts/run_baseline.py --model CohereLabs/tiny-aya-global \
  --tasks data/tasks_eval.jsonl --out runs/baseline_eval
python scripts/run_baseline.py --model CohereLabs/tiny-aya-global \
  --adapter runs/grpo_sft --tasks data/tasks_eval.jsonl --out runs/tuned_eval
```

Training:

```bash
python scripts/train_grpo.py --check
python scripts/train_grpo.py --max-steps 2
python scripts/train_grpo.py --model runs/sft_merged --tasks data/tasks_train.jsonl \
  --out runs/grpo_sft --max-steps 60 --efficiency
```

Low-memory fallback:

```bash
python scripts/train_grpo.py --model runs/sft_merged --max-steps 60 \
  --max-completion-length 128 --max-steps-per-turn 2 \
  --batch-size 1 --grad-accum 4 --generations 2
```

AuditKIT:

```bash
python scripts/auditkit_eval.py --trajectories runs/tuned_eval/trajectories.jsonl \
  --model hf:runs/grpo_sft
```

Artifacts:

```bash
hf upload primce/tiny-aya-hinglish-lora runs/grpo_sft --repo-type model
hf upload primce/hinglish-expense-split data/tasks_eval.jsonl tasks/eval.jsonl --repo-type dataset
```

---

## 15. Non-negotiables

1. Baseline before training, same tasks, same decoding.
2. Held-out eval is never trained on.
3. Strict and fuzzy metrics both reported.
4. Every claimed number has a file path.
5. Freeze Nov 10; submit Nov 13.
6. If something fails, report the failure. A well-documented negative result beats an unsupported win.
