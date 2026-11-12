# How this entry wins

A solo entry beats better-resourced teams by having **no empty rubric cell** and **one thing
nobody else did**. Model quality is settled and measured twice: 96.67% on the 30-task hard split
and 93.33% on a 90-task holdout, both with paired tests, five seed controls, an ablation, and an
independent AuditKIT pass. Everything left is packaging, and packaging is what most solo entries lose on.

## The five things that decide it

| # | What judges check | Status |
|---|---|---|
| 1 | Public repo, runnable, numbers reproducible | done: `python scripts/verify_all.py` → 28/28, no GPU or token; one-click Colab `notebooks/colab_reproduce.ipynb` |
| 2 | **Live Space demo** with a visible tool trace | **live**: https://huggingface.co/spaces/primce/hinglish-expense-agent (ZeroGPU `zero-a10g`, sft4 adapter, base/tuned switch, injection scenario, and a CPU-only "Try the guard yourself" panel that runs on any paste with no GPU quota) |
| 3 | **HF Article** with metrics, before/after, rationale | **live**: https://huggingface.co/spaces/primce/hinglish-track3-article (static Space; Hub Articles need PRO) |
| 4 | HF artifacts: adapter + run logs + dataset | adapter (`sft4_adapter`, `sft4_seed7_adapter`, `sft4_seed123_adapter`, `sft5_adapter`, `sft3_seed7/123_adapter`), all run logs, and two public collections: [models](https://huggingface.co/collections/primce/lexsi-track-3-models-hinglish-expense-agent-6af3f4e4b0388e242c7d574a) and [datasets](https://huggingface.co/collections/primce/lexsi-track-3-datasets-hinglish-expense-agent-6af1584f9048846d4db4970b) |
| 5 | **One memorable differentiator** | the injection eval: 18/24 → 2/24 ledger attacks by ledger accounting and 8/24 → 0/24 attacker numbers in answers on the headline split, then **6 ledger hits and 0 attacker values in answers on 136 injected tasks across four pinned splits, four payload styles and three payload languages** (base ledger: 18/24 + 48/72 + 5/16 + 14/24), plus a post-freeze 30-task held-out split at 5/30 and 0/30; includes a Hinglish/Devanagari payload split so the defence cannot be an English-pattern matcher, no LLM judge, on a track where the required toolkit (AuditKIT) usually only scores tool calls |

### The four attack lines a strong reviewer will use

| Attack | Answer today |
|---|---|
| "Track says RL, your win is SFT" | "GRPO ran twice and moved **0 of 30** tasks in round 1 (19/30 → 19/30) and 0 fixed / 0 broken on hard-90 in round 2 (68/90 → 68/90, p = 1.0) with the abstain reward, 40 steps and lr 3e-6. That zero is measured, diagnosed (`frac_reward_zero_std` 0.8–0.95 at 2 generations) and reported, and round 3 shows the real bottleneck was data coverage, not RL (`docs/ROUND2.md`)." |
| "30 tasks, 1 seed" | "90-task holdout at 15 per family (93.33%), paired against base at p < 1e-20, and against round 2 at 18 fixed / 2 broken, p = 0.0004. Five seed controls across two recipes. Mix3, the shipped recipe, has three seeds — 96.67 / 100 / 96.67 on hard-30 and 92.22 / 93.33 / 93.33 on hard-90 — an 1.11 pp spread on hard-90, no seed pair significant (p = 1.0), and five tasks all three seeds miss. Mix2's 7.77 pp spread is published too, and it is the contrast that shows what the contrast rows bought. The three mix3 seeds pool to 251/270 = 93.0% on hard-90 (task-block bootstrap 87.4–97.4%) and 88/90 = 97.8% on hard-30, which is the recipe-level estimate rather than the best seed." |
| "It learned the tool names, not the task" | "Rename every tool and parameter — the canonical name is then rejected as unknown — and it keeps **25/30** (4 broken, 0 fixed, exact McNemar p = 0.125). The untuned base under the same rename is **0/30** with 0.60 loop collapse and 7.0 calls per task. The same 30 tasks written in Devanagari: **25/30**. The two probes break different families (`two_query` under the rename, `paraphrase` in Devanagari), and both are reported as n = 30 diagnostics, not headline numbers." |
| "Abstention is still 1/8" | "It is 4/8 on the adapter and 7/8 with the hardened prompt (0/8 invented) on the 8 pre-registered rows, against a 6/8 bar — and on eight fresh rows it is 4/8 and 8/8 (0/8 invented) again. `attack_ledger` is 0/24 and `attack_answer` is 0/24 on the headline split, and 0/72 and 0/72 on the second sample. The hardened prompt is not an accuracy tax either: 86/90 against 84/90 on clean hard-90 (2 fixed, 0 broken, p = 0.5)." |
| "Your model never tells the user it ignored an injection" | "Correct — 0 of 136 answers name it, measured offline. That is why the warning comes from a deterministic guard under the model (`hinglish_agent/guard.py`), not from the weights: 166/166 published payloads flagged, 5/8 on a blind set written after the rules froze, 6/8 on a probe written after reading the source (the two misses spell the amount out), 0 false warnings on 294 clean episodes / 1,010 tool results, 0.22 ms per result. The 166/166 is in-sample and is not quoted alone; the blind set and the three misses are in `docs/GUARD.md`. End to end on the same 136 rows it takes the ledger from **6/136 to 0/136** and solves **136/136**, with the note replaced (`neutralise`) and with it deleted outright (`strip`) — the second run is the control for the notice and the no-payload baseline." |
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

Open the form from the Lexsi Discord and paste this block field by field. Nothing to compose:

**Abstract field (one block, under 500 words):**

```text
Track 3, solo. Base model CohereLabs/tiny-aya-global (3.35B); LoRA SFT on oracle trajectories, then 40 GRPO steps through AgentTune's trainer; every headline number re-scored by AuditKIT 1.1.3 in recorded mode.

Result. On a frozen 30-task hard split, task success goes 3.33% -> 96.67% (28 fixed, 0 broken, exact McNemar p = 7.5e-9); on a 90-task holdout, 3.33% -> 93.33% (81 fixed, 0 broken, p < 1e-20). Three seeds of the shipped recipe pool to 251/270 = 93.0% (87.4-97.4% task-block bootstrap), and the shipped seed is not the best one. Calls per task fall from 7.24 to 3.38 against an oracle 3.41. Success is the final ledger compared with a deterministic integer-paise oracle: no LLM judge anywhere.

Safety, the part I would keep if I could keep one. A shared-expense agent that moves money is a target for indirect prompt injection through tool results. Four pinned splits, 136 injected tasks, four payload styles, three payload languages (including 24 Hinglish and Devanagari payloads): the adapter lets an attacker expense into the ledger on 6 of 136 and repeats an attacker value in 0 of 136 answers, while the base model fails 18/24, 48/72, 5/16 and 14/24. An earlier "0 of 136" ledger claim came from a threshold that a shared expense cannot reach; it is corrected publicly. A 30-task split built after the adapter was frozen scores 5/30 and 0/30, every miss on an obfuscated amount. A deterministic guard in front of the model takes the same rows to 0/136 expenses and 136/136 solved, in two modes: one replaces the injected note, one deletes it, so the effect is the removal and not a hint. The guard is scored on three sets: 166/166 published payloads, 5/8 on eight payloads written after its rules were frozen, and 6/8 on eight written after reading its source (the two misses spell the amount out).

Verification. python scripts/verify_all.py runs 28 checks offline with no GPU and no token (29 with --with-tests); docs/MANIFEST.md pins 76 files by sha256 at a repo revision and fails on drift; a static gallery renders all 90 held-out tasks base-vs-adapter, including the six both models miss; and the limitations (one 8-payload source-read probe and no red-team loop, no 8B ablation, two safety runs aggregate-only) are stated in the README before anyone asks.
```

**Other fields:**

```text
Track: 3 — Agentic fine-tuning
Team: solo entry (Prince Maurya)
Base model: CohereLabs/tiny-aya-global (3.35B) + LoRA adapter (PEFT, r=16, all-linear)

One-line result: 3.33% -> 96.67% task success on a frozen 30-task hard split (28 fixed, 0 broken,
exact McNemar p = 7.5e-9) and 3.33% -> 93.33% on a 90-task holdout (81 fixed, 0 broken, p < 1e-20),
at 3.38 calls per task on hard-90 against an oracle 3.41 (87/90 tasks take exactly the oracle number
of calls, 3 take one fewer, none take more). Success is scored on the final ledger state, not by an
LLM judge.

Toolkits: AgentTune (GRPO stage built with agenttune.create_agentic_trainer, scripts/train_grpo.py)
and AuditKIT 1.1.3 (episode_from_openai_messages + AgentEvalRunner in recorded mode with
FinalStateAssertion and CustomPredicate, scripts/auditkit_eval.py). No judge model. Exact-call
accuracy 0.033 -> 1.000 and tool-call F1 0.188 -> 1.000 on the same 30 tasks.

Differentiator: the repo ships an indirect prompt-injection eval for ledger agents, scored without
an LLM judge, plus a prompt-level defence measured on the same tasks. Four pinned splits, four
payload styles, three payload languages, 136 injected tasks: the adapter lets an attacker expense
into the ledger on 6 of them (2/24 + 4/72 + 0/16 + 0/24) and repeats an attacker value in 0 answers
(0/24 + 0/72 + 0/16 + 0/24), against 18/24, 48/72, 5/16 and 14/24 ledger hits for base Tiny Aya. A
fifth 30-task split, built after the adapter was frozen, is 5/30 and 0/30. The language split is
the same attacks in Hinglish and Devanagari (18 Latin, 6 Devanagari). A deterministic guard in front
of the adapter takes the same 136 rows to **0/136** attacker expenses and **136/136** solved, in
`neutralise` and in `strip` mode — the second deletes the injected field and leaves no marker, so
the result is the removal and not a hint. Abstention is 4/8 asked by
default on both samples and 7/8 and 8/8 with the hardened prompt, 0/8 invented, against a
pre-registered 6/8 bar. Two generalisation probes are published with their runs: renamed tool
schema 25/30 against 29/30 canonical, Devanagari 25/30, and the untuned base under the same
rename 0/30.

Video (1:44, narrated): https://huggingface.co/datasets/primce/lexsi-runs/resolve/main/demo/lexsi_track3_demo.mp4
Write-up: submission/README.md in the repo (also rendered at the write-up link below)
Reproduce in two minutes, no GPU and no token: git clone the repo, then python scripts/verify_all.py -> 28 checks (29 with --with-tests);
python scripts/fetch_runs.py and python scripts/bootstrap.py then recompute every interval, paired test and the pooled seed rate, and leave docs/STATS.md byte-identical;
python scripts/manifest.py --check re-hashes 76 files on the Hub and fails on drift.

Links:
  code: https://github.com/PRINCE2-AI/lexsi-agentic-hinglish
  live demo: https://huggingface.co/spaces/primce/hinglish-expense-agent
  all 90 held-out episodes: https://huggingface.co/spaces/primce/hinglish-episodes
  write-up: https://huggingface.co/spaces/primce/hinglish-track3-article
  limitations: https://github.com/PRINCE2-AI/lexsi-agentic-hinglish#what-this-does-not-show
  adapter: https://huggingface.co/primce/tiny-aya-hinglish-lora
  run logs (every summary.json + trajectories): https://huggingface.co/datasets/primce/lexsi-runs
  ROI vs a closed API: https://github.com/PRINCE2-AI/lexsi-agentic-hinglish/blob/main/docs/roi.md
  judge checklist (rule by rule): https://github.com/PRINCE2-AI/lexsi-agentic-hinglish/blob/main/docs/JUDGE.md
  collection: https://huggingface.co/collections/primce/lexsi-track-3-hinglish-expense-agent-6aeb3207e10dfc7dda6329d6
```

The write-up being a static Space is deliberate: Hub Articles need PRO on this account, and the
Space renders the same markdown from `docs/hf_article.md`.

### 5. Demo Day (only if shortlisted)

`docs/demo_script.md` is the Demo Day runbook: 3:00 of content in a 5-minute slot, a pre-flight
warm-up, and a fallback for each way the Space can fail. Every number in it is frozen. Rehearse the
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
