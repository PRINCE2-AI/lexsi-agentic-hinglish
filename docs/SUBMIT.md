# How this entry wins

A solo entry beats better-resourced teams by having **no empty rubric cell** and **one thing
nobody else did**. Model quality is already settled: 96.67% on the hard split with a paired test.
Everything left is packaging, and packaging is what most solo entries lose on.

## The five things that decide it

| # | What judges check | Status |
|---|---|---|
| 1 | Public repo, runnable, numbers reproducible | done (`8c78979`+) |
| 2 | **Live Space demo** with a visible tool trace | **missing** — `scripts/deploy_space.py` is ready |
| 3 | **HF Article** with metrics, before/after, rationale | drafted, needs final tables |
| 4 | HF artifacts: adapter + run logs + dataset | adapter and logs uploaded; model card missing |
| 5 | **One memorable differentiator** | the injection eval (32 tasks, no LLM judge) |

## Order of work (do not reorder)

### 1. Space live (30 min, blocking)

```bash
export HF_TOKEN=...          # PowerShell: $env:HF_TOKEN="..."
python scripts/deploy_space.py --repo primce/hinglish-expense-agent
```

Then Space → Settings → Hardware → **T4 small** (the free CPU tier cannot load a 3.35B model).
Wait for "Running". Open it, run the default question, confirm the tool trace and the final ledger
render. If the build fails, the log will name the missing package: add it to `space/requirements.txt`.

The Space defaults to the frozen adapter. That matters: a judge who opens it sees the tuned agent,
not the failing baseline.

### 2. Model card (5 min)

`docs/MODEL_CARD.md` is ready to paste as the README of `primce/tiny-aya-hinglish-lora`.
Fill the base-model hard-30 row from `runs/base_hard30/summary.json` when that run finishes.

Then tag the frozen adapter so the exact bytes are named:

```bash
python -c "from huggingface_hub import HfApi; HfApi().create_tag('primce/tiny-aya-hinglish-lora', tag='sft2-frozen', repo_type='model')"
```

### 3. Article (45 min)

Paste `docs/hf_article.md`. Three tables must be filled from run summaries, never estimated:

- before/after on hard-30 (base, easy-SFT, +GRPO, mix) — `runs/*/summary.json`
- per-family residual failures — `scripts/stats.py --a ... --b ...`
- injection results — `python scripts/safety_table.py`

### 4. Submission form (10 min, hard deadline)

Fill it on the day the form is confirmed open, not on Nov 13. Answers that already exist in the
repo, so paste rather than write:

- **Track**: 3, Agentic fine-tuning.
- **Model**: Cohere Labs Tiny Aya (`CohereLabs/tiny-aya-global`), LoRA, 3.35B.
- **One-line result**: 25% -> 96.67% task success on a frozen 30-task hard split (p = 0.00195),
  at 3.57 tool calls per task versus an oracle 3.5.
- **Differentiator**: the repo ships a 32-task indirect prompt-injection eval for ledger agents,
  scored without an LLM judge, plus a prompt-level defence measured on the same tasks.
- **Links**: repo, Space, model, dataset (`primce/lexsi-runs`), article.

### 5. Demo Day (only if shortlisted)

`docs/demo_script.md` is timed to 3 minutes and every number in it is frozen. Rehearse the
honest GRPO line — it is the part that makes the rest credible.

## What a judge can verify in two minutes

```bash
git clone https://github.com/PRINCE2-AI/lexsi-agentic-hinglish && cd lexsi-agentic-hinglish
python data/build_hard.py && python data/build_safety.py
python tests/test_hard_data.py && python tests/test_safety.py && python tests/test_stats.py
python scripts/run_oracle_eval.py --tasks data/tasks_hard_eval.jsonl
```

No GPU, no token, no model download. Expected: pin matches, 30/30 oracle success, all tests pass.
Put this block at the end of the article too.

## Failure modes to avoid

- Do not report a number that is not in a `summary.json`. One invented decimal costs more than a
  missing table.
- Do not deploy the Space on CPU hardware "to check it" — it will OOM and look broken in the logs.
- Do not skip the safety split because the numbers might look bad. A negative safety result with a
  tested mitigation is a stronger submission than no safety section on a safety track.
- Do not let the adapter tag drift. `sft2-frozen` must name the artifact the article measures.
