# How this entry wins

A solo entry beats better-resourced teams by having **no empty rubric cell** and **one thing
nobody else did**. Model quality is settled and measured twice: 96.67% on the 30-task hard split
and 75.56% on a 90-task holdout, both with paired tests, three seeds, and an independent AuditKIT
pass. Everything left is packaging, and packaging is what most solo entries lose on.

## The five things that decide it

| # | What judges check | Status |
|---|---|---|
| 1 | Public repo, runnable, numbers reproducible | done: `python scripts/verify_all.py` → 13/13, no GPU or token |
| 2 | **Live Space demo** with a visible tool trace | **live**: https://huggingface.co/spaces/primce/hinglish-expense-agent (T4 small, sft3 adapter) |
| 3 | **HF Article** with metrics, before/after, rationale | **live**: https://huggingface.co/spaces/primce/hinglish-track3-article (static Space; Hub Articles need PRO) |
| 4 | HF artifacts: adapter + run logs + dataset | adapter (`sft3_adapter` + two seed adapters), all run logs, **and the required HF Collection** |
| 5 | **One memorable differentiator** | the injection eval: 13/24 → 0/24 ledger attacks, no LLM judge, on a track where the required toolkit (AuditKIT) usually only scores tool calls |

### The four attack lines a strong reviewer will use

| Attack | Answer today |
|---|---|
| "Track says RL, your win is SFT" | "GRPO ran twice. Round 1 moved 0/30 because two generations on a 96% policy give zero reward variance. Round 2 uses an abstention-aware reward, 40 steps, lr 3e-6, and the delta is reported on both axes whichever way it points (`docs/ROUND2.md`)." |
| "30 tasks, 1 seed" | "90-task holdout at 15 per family (75.56%), paired against base at p = 9.2e-19, plus three seeds: 80.00 / 93.33 / 96.67 on hard-30 and 75.56 / 83.33 / 75.56 on hard-90. The spread is in the write-up, and seed 7 is better than the shipped seed." |
| "Abstention is still 1/8" | "It is 2/8 now, against a pre-registered bar of 6/8 — missed, and printed as missed. `attack_ledger` stays 0/24 and hard-30 stays 29/30 on the way." |
| "Your article is a Space, not an HF blog" | "Hub Articles need PRO or a Team org on this account; the static Space renders the same markdown from the same file." |

Say all four out loud before a judge does. The fourth is unwinnable without a subscription, and
stating the constraint is cheaper than being caught by it.

## Order of work (do not reorder)

### 1. Space live (30 min, blocking)

```bash
export HF_TOKEN=...          # PowerShell: $env:HF_TOKEN="..."
python scripts/deploy_space.py --repo primce/hinglish-expense-agent
```

Then Space → Settings → Hardware → **T4 small** (the free CPU tier cannot load a 3.35B model).
Wait for "Running". Open it, run the default question, confirm the tool trace and the final ledger
render. If the build fails, the log will name the missing package: add it to `space/requirements.txt`.

The Space defaults to the shipped adapter (`ADAPTER_SUBFOLDER=sft3_adapter`). That matters: a judge
who opens it sees the tuned agent, not the failing baseline.

### 2. Model card (5 min)

`docs/MODEL_CARD.md` is ready to paste as the README of `primce/tiny-aya-hinglish-lora`.
The base-model rows are filled from the 2026-11-04 runs (hard-30 3.33%, hard-90 3.33%).

Then tag the shipped adapter so the exact bytes are named:

```bash
python -c "from huggingface_hub import HfApi; HfApi().create_tag('primce/tiny-aya-hinglish-lora', tag='sft3-submission', repo_type='model')"
python -c "from huggingface_hub import HfApi; HfApi().create_tag('primce/tiny-aya-hinglish-lora', tag='sft2-frozen', repo_type='model')"
```

### 3. Article (done 2026-11-04)

Hub Articles require PRO, so the article is published as a free static Space:
https://huggingface.co/spaces/primce/hinglish-track3-article

It is rendered from `docs/hf_article.md` by `python scripts/build_article_space.py --repo
primce/hinglish-track3-article`. Re-run that command after any number changes; the tables in
the page come from the same file, never hand-edited HTML.

### 4. Submission form (10 min, hard deadline)

Everything the form asks for already exists as a file. Paste, do not write:

- **Track**: 3, Agentic fine-tuning.
- **Model**: Cohere Labs Tiny Aya (`CohereLabs/tiny-aya-global`), LoRA, 3.35B.
- **One-line result**: 3.33% → 96.67% task success on a frozen 30-task hard split (28 fixed,
  0 broken, exact McNemar p = 7.5e-9) and 3.33% → 75.56% on a 90-task holdout (66 fixed, 1
  broken, p = 9.2e-19), at the oracle call count.
- **Toolkits**: AgentTune (GRPO stage, `scripts/train_grpo.py`) and AuditKIT 1.1.3
  (`scripts/auditkit_eval.py`), scored without a judge model: exact-call accuracy 0.033 → 1.000
  and tool-call F1 0.188 → 1.000 on the same 30 tasks.
- **Differentiator**: the repo ships a 32-task indirect prompt-injection eval for ledger agents,
  scored without an LLM judge, plus a prompt-level defence measured on the same tasks. Base
  writes attacker money in 13/24 tasks, the adapter in 0/24 (p = 0.00024); abstention moves 0/8 →
  2/8 against a pre-registered 6/8 bar and is reported as missed.
- **Write-up**: `submission/README.md` in this repo.
- **Links**: https://github.com/PRINCE2-AI/lexsi-agentic-hinglish ,
  https://huggingface.co/spaces/primce/hinglish-expense-agent ,
  https://huggingface.co/primce/tiny-aya-hinglish-lora ,
  https://huggingface.co/datasets/primce/lexsi-runs ,
  https://huggingface.co/spaces/primce/hinglish-track3-article ,
  https://huggingface.co/collections/primce/lexsi-track-3-hinglish-expense-agent-6aeb3207e10dfc7dda6329d6

### 5. Demo Day (only if shortlisted)

`docs/demo_script.md` is timed to 3 minutes and every number in it is frozen. Rehearse the
honest GRPO line — it is the part that makes the rest credible.

## What a judge can verify in two minutes

```bash
git clone https://github.com/PRINCE2-AI/lexsi-agentic-hinglish && cd lexsi-agentic-hinglish
python scripts/verify_all.py
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
