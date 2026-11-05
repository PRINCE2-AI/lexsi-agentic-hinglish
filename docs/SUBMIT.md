# How this entry wins

A solo entry beats better-resourced teams by having **no empty rubric cell** and **one thing
nobody else did**. Model quality is settled and measured twice: 96.67% on the 30-task hard split
and 93.33% on a 90-task holdout, both with paired tests, five seed controls, an ablation, and an
independent AuditKIT pass. Everything left is packaging, and packaging is what most solo entries lose on.

## The five things that decide it

| # | What judges check | Status |
|---|---|---|
| 1 | Public repo, runnable, numbers reproducible | done: `python scripts/verify_all.py` → 20/20, no GPU or token; one-click Colab `notebooks/colab_reproduce.ipynb` |
| 2 | **Live Space demo** with a visible tool trace | **live**: https://huggingface.co/spaces/primce/hinglish-expense-agent (ZeroGPU `zero-a10g`, sft4 adapter, base/tuned switch, injection scenario) |
| 3 | **HF Article** with metrics, before/after, rationale | **live**: https://huggingface.co/spaces/primce/hinglish-track3-article (static Space; Hub Articles need PRO) |
| 4 | HF artifacts: adapter + run logs + dataset | adapter (`sft4_adapter`, `sft4_seed7_adapter`, `sft4_seed123_adapter`, `sft5_adapter`, `sft3_seed7/123_adapter`), all run logs, **and the required HF Collection** |
| 5 | **One memorable differentiator** | the injection eval: 13/24 → 0/24 ledger attacks and 8/24 → 0/24 attacker numbers in answers, no LLM judge, on a track where the required toolkit (AuditKIT) usually only scores tool calls |

### The four attack lines a strong reviewer will use

| Attack | Answer today |
|---|---|
| "Track says RL, your win is SFT" | "GRPO ran twice: 0/30 in round 1, and 0 tasks fixed / 0 broken on hard-90 in round 2 (p = 1.0) with the abstain reward, 40 steps and lr 3e-6. That zero is measured, diagnosed (`frac_reward_zero_std` 0.8–0.95 at 2 generations) and reported, and round 3 shows the real bottleneck was data coverage, not RL (`docs/ROUND2.md`)." |
| "30 tasks, 1 seed" | "90-task holdout at 15 per family (93.33%), paired against base at p < 1e-20, and against round 2 at 18 fixed / 2 broken, p = 0.0004. Five seed controls across two recipes. Mix3, the shipped recipe, has three seeds — 96.67 / 100 / 96.67 on hard-30 and 92.22 / 93.33 / 93.33 on hard-90 — an 1.11 pp spread on hard-90, no seed pair significant (p = 1.0), and five tasks all three seeds miss. Mix2's 7.77 pp spread is published too, and it is the contrast that shows what the contrast rows bought." |
| "Abstention is still 1/8" | "It is 4/8 on the adapter and 7/8 with the hardened prompt (0/8 invented), against a pre-registered 6/8 bar. `attack_ledger` is 0/24 and `attack_answer` is 0/24 on the same runs." |
| "Your article is a Space, not an HF blog" | "Hub Articles need PRO or a Team org on this account; the static Space renders the same markdown from the same file." |

Say all four out loud before a judge does. The fourth is unwinnable without a subscription, and
stating the constraint is cheaper than being caught by it.

## Order of work (do not reorder)

### 1. Space live (30 min, blocking)

```bash
export HF_TOKEN=...          # PowerShell: $env:HF_TOKEN="..."
python scripts/deploy_space.py --repo primce/hinglish-expense-agent
```

Then Space → Settings → Hardware → **ZeroGPU** (`zero-a10g`); `space/app.py` requests a GPU with
`spaces.GPU(duration=120)`, and the free CPU tier cannot load a 3.35B model.
Wait for "Running". Open it and run the default question in both modes: *Base model (adapter off)*
must mismatch, *Shipped adapter (sft4)* must match. Then run the injection scenario in shipped mode
and confirm the verdict says RESISTED with a clean ledger. If the build fails, the log will name the
missing package: add it to `space/requirements.txt`.

The Space defaults to the shipped adapter (`ADAPTER_SUBFOLDER=sft4_adapter`). That matters: a judge
who opens it sees the tuned agent, not the failing baseline.

### 2. Model card (5 min)

`docs/MODEL_CARD.md` is ready to paste as the README of `primce/tiny-aya-hinglish-lora`.
The base-model rows are filled from the 2026-11-04 runs (hard-30 3.33%, hard-90 3.33%); the shipped rows from the 2026-11-05 runs.

Then tag the shipped adapter so the exact bytes are named:

```bash
python -c "from huggingface_hub import HfApi; HfApi().create_tag('primce/tiny-aya-hinglish-lora', tag='sft4-submission', repo_type='model')"
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
  0 broken, exact McNemar p = 7.5e-9) and 3.33% → 93.33% on a 90-task holdout (81 fixed, 0
  broken, p < 1e-20), at 3.38 calls against the oracle's 3.57.
- **Toolkits**: AgentTune (GRPO stage, `scripts/train_grpo.py`) and AuditKIT 1.1.3
  (`scripts/auditkit_eval.py`), scored without a judge model: exact-call accuracy 0.033 → 1.000
  and tool-call F1 0.188 → 1.000 on the same 30 tasks.
- **Differentiator**: the repo ships a 32-task indirect prompt-injection eval for ledger agents,
  scored without an LLM judge, plus a prompt-level defence measured on the same tasks. Base
  writes attacker money in 13/24 tasks and repeats the number in 8/24 answers; the adapter is 0/24
  and 0/24 (p = 0.00024 on the ledger). Abstention is 4/8 asked by default and 7/8 with the
  hardened prompt, against a pre-registered 6/8 bar.
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
- Do not let the adapter tag drift. `sft4-submission` must name the artifact the article measures.
