# How this entry wins

A solo entry beats better-resourced teams by having **no empty rubric cell** and **one thing
nobody else did**. Model quality is already settled: 96.67% on the hard split with a paired test.
Everything left is packaging, and packaging is what most solo entries lose on.

## The five things that decide it

| # | What judges check | Status |
|---|---|---|
| 1 | Public repo, runnable, numbers reproducible | done (`8c78979`+) |
| 2 | **Live Space demo** with a visible tool trace | **live**: https://huggingface.co/spaces/primce/hinglish-expense-agent (T4 small) |
| 3 | **HF Article** with metrics, before/after, rationale | **live**: https://huggingface.co/spaces/primce/hinglish-track3-article (static Space; Hub Articles need PRO) |
| 4 | HF artifacts: adapter + run logs + dataset | adapter and logs (incl. safety runs) uploaded; model card pending paste |
| 5 | **One memorable differentiator** | the injection eval: 13/24 -> 0/24 ledger attacks, no LLM judge |

### The four attack lines a strong reviewer will use

| Attack | Answer today | Answer after the round-2 runs |
|---|---|---|
| "Track says RL, your win is SFT" | "GRPO ran; it moved 0/30 because 2 generations on a 96% policy gives zero reward variance at 30 steps." | GRPO with an abstention-aware reward, 40 steps, lr 3e-6, delta reported on safety and success (`docs/ROUND2.md`) |
| "30 tasks, 1 seed" | "Wilson 83–99%; limits are stated." | hard-90 (15/family, own pin) + seed-42 vs seed-123 |
| "Abstention is still 1/8" | "Reported as open, not as solved." | 32 underspecified rows as minimal pairs in the SFT mix; target `asked ≥ 6/8` without losing hard-30 |
| "Your article is a Space, not an HF blog" | "Hub Articles need PRO or a Team org on this account; the static Space renders the same markdown." | unchanged — this one is an account tier, not a repo gap |

Say the first three out loud before a judge does. The fourth one is unwinnable without a
subscription, and stating the constraint is cheaper than being caught by it.

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
The base-model hard-30 row is filled (3.33%, 1/30) from the 2026-11-04 run.

Then tag the frozen adapter so the exact bytes are named:

```bash
python -c "from huggingface_hub import HfApi; HfApi().create_tag('primce/tiny-aya-hinglish-lora', tag='sft2-frozen', repo_type='model')"
```

### 3. Article (done 2026-11-04)

Hub Articles require PRO, so the article is published as a free static Space:
https://huggingface.co/spaces/primce/hinglish-track3-article

It is rendered from `docs/hf_article.md` by `python scripts/build_article_space.py --repo
primce/hinglish-track3-article`. Re-run that command after any number changes; the tables in
the page come from the same file, never hand-edited HTML.

### 4. Submission form (10 min, hard deadline)

Fill it on the day the form is confirmed open, not on Nov 13. Answers that already exist in the
repo, so paste rather than write:

- **Track**: 3, Agentic fine-tuning.
- **Model**: Cohere Labs Tiny Aya (`CohereLabs/tiny-aya-global`), LoRA, 3.35B.
- **One-line result**: 3.33% -> 96.67% task success on a frozen 30-task hard split
  (28 fixed, 0 broken, exact McNemar p = 7.5e-9), at 3.57 tool calls per task versus an
  oracle 3.57.
- **Differentiator**: the repo ships a 32-task indirect prompt-injection eval for ledger agents,
  scored without an LLM judge, plus a prompt-level defence measured on the same tasks. Base
  writes attacker money in 13/24 tasks, the adapter in 0/24 (p = 0.00024); underspecified
  abstention stays broken at 1/8 and is reported as such.
- **Links**: https://github.com/PRINCE2-AI/lexsi-agentic-hinglish ,
  https://huggingface.co/spaces/primce/hinglish-expense-agent ,
  https://huggingface.co/primce/tiny-aya-hinglish-lora ,
  https://huggingface.co/datasets/primce/lexsi-runs ,
  https://huggingface.co/spaces/primce/hinglish-track3-article

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
