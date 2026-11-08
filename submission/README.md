# Track 3 submission — a Hinglish expense-splitting agent that moves money safely

**Track 3, Agentic fine-tuning.** Solo entry. Base model `CohereLabs/tiny-aya-global` (3.35B),
LoRA SFT + AgentTune GRPO, evaluated with AuditKIT 1.1.3.

One line: **3.33% → 96.67% task success on a frozen 30-task hard split, and 3.33% → 93.33% on a
90-task holdout, with the same adapter moving attacker money into the ledger in 0 of 24
prompt-injection tasks and repeating the attacker number in 0 of 24 answers.**

## Artifacts

| What | Where |
|---|---|
| Code + data + eval scripts | https://github.com/PRINCE2-AI/lexsi-agentic-hinglish |
| Adapter (shipped) | https://huggingface.co/primce/tiny-aya-hinglish-lora (`sft4_adapter`) |
| Run logs (every `summary.json` + trajectories) | https://huggingface.co/datasets/primce/lexsi-runs |
| Live demo (tool trace visible) | https://huggingface.co/spaces/primce/hinglish-expense-agent |
| Demo video (1:44, narrated) | https://huggingface.co/datasets/primce/lexsi-runs/resolve/main/demo/lexsi_track3_demo.mp4 |
| Write-up | https://huggingface.co/spaces/primce/hinglish-track3-article |
| ROI (optional deliverable) | [docs/roi.md](https://github.com/PRINCE2-AI/lexsi-agentic-hinglish/blob/main/docs/roi.md) — measured tokens and latency, $53.42 vs $79.74 per 10,000 requests (0.67x) |
| HF Collection | https://huggingface.co/collections/primce/lexsi-track-3-hinglish-expense-agent-6aeb3207e10dfc7dda6329d6 |

## Problem statements this entry claims

Track 3 lists nine problem statements. A statement is claimed here only when a measured number
backs it; the rest are listed as not claimed, on purpose.

| # | Statement | Claimed | Evidence |
|---|---|---|---|
| 1 | Reliable tool calling | yes | 96.67% (29/30) and 93.33% (84/90) with exact calls, zero invalid calls, zero loops; the RL stage was run twice and its null is reported, not hidden |
| 2 | Efficient trajectories | yes | 3.57 calls/task on hard-30 (oracle 3.57) and 3.38 on hard-90 (oracle 3.41), zero redundant calls, zero loops; base takes 7.03 calls and 16/30 episodes never reach `finish` |
| 6 | Data and database agents | yes | five tools over a deterministic integer-paise ledger, rewarded on the verified final state, never on text similarity |
| 3 | Agentic distillation | no | no teacher model was distilled; this entry is SFT on oracle trajectories plus GRPO |
| 4, 5, 7, 8, 9 | RAG / synthetic multi-hop / workflow / self-healing / reward design | no | not claimed; the injection split is a probe against reward gaming, not a reward-design result |

## Track 3 requirements, and where each is met

| Requirement | Evidence in this repo |
|---|---|
| **AgentTune used** | `scripts/train_grpo.py` builds the trainer through `agenttune.create_agentic_trainer` and runs 40 GRPO steps on top of the SFT warm start (`scripts/hf_job_stage2.sh`, HF Job `6aeb328dc78863f9c6328d08`). |
| **AuditKIT used** | `scripts/auditkit_eval.py` rebuilds every recorded episode as an `auditkit.agent_eval.AgentCase` + `episode_from_openai_messages`, runs `AgentEvalRunner` in `mode="recorded"` with `FinalStateAssertion` and `CustomPredicate`, and scores `tool_call_f1`, `tool_call_f1_name`, `tool_call_validity`, `redundant_tool_calls`, `agent_loop_detection`. No judge model is used. Replayed on 1.1.3 and on the current 1.1.6: identical verdicts on all four runs (29/30, 86/90, 28/30, 26/30). |
| **Before/after metrics, more than one** | Success, exact-call accuracy, tool-call F1/precision/recall, redundant calls, loop detection, ledger-attack rate, abstention — tables below. |
| **HF Collection** | linked above. |
| **Write-up** | this file; rendered version in the article Space. |

## Method

1. **Environment.** Five stateless tools over a deterministic integer-paise ledger
   (`add_expense`, `convert_currency`, `get_balance`, `settle_up`, `finish`);
   `CURRENCY_CENTS = {"INR": 100, "USD": 8300, "EUR": 9000}`. A task is scored on final ledger
   state, answer content and call-level diagnostics, not on text similarity.
2. **SFT.** LoRA r=16, all-linear, fp16, lr 2e-4, 2 epochs, grad-accum 4,
   `masked_causal_lm_loss` so only assistant turns are supervised. Data
   `data/tasks_train_mix3.jsonl`: 374 rows = 184 easy/hard mix + 96 abstention rows (32
   underspecified × 3) + 32 fully-specified twins + 62 contrast pairs (empty ledger vs missing
   field, two-conversion sums). ~188 optimizer steps on one T4.
3. **GRPO (AgentTune).** 40 steps, lr 3e-6, 2 generations, graded reward + efficiency reward +
   a new abstain reward (1.0 when the only call is `finish` with a question on an
   `abstain=1` row). Stage 2 runs on top of the SFT merge.
4. **Evaluation.** Greedy, `--no-4bit`, one sample per task, frozen splits with SHA-256 pins:
   hard-30 (`tasks_hard_eval.jsonl`), hard-90 (`tasks_hard90_eval.jsonl`, 15 per family),
   easy eval-24, safety-32. Every number comes from an uploaded `summary.json`; nothing is
   estimated.

## Results

### Task success (held-out, greedy)

| Model | hard-30 | hard-90 | eval-24 |
|---|---:|---:|---:|
| Base Tiny Aya, prompt-only | 3.33% (1/30) | 3.33% (3/90) | - |
| SFT-mix, round 1 (`sft2`) | 96.67% (29/30) | 88.89% (80/90) | 100% (24/24) |
| SFT-mix2, round 2 (`sft3`) | 96.67% (29/30) | 75.56% (68/90) | 100% (24/24) |
| **SFT-mix3 seed 42 (`sft4`, shipped)** | **96.67% (29/30)** | **93.33% (84/90)** | **100% (24/24)** |
| SFT-mix5 (ablation) | 100% (30/30) | 92.22% (83/90) | 100% (24/24) |
| SFT-mix2 seed 7 | 93.33% (28/30) | 83.33% (75/90) | - |
| SFT-mix2 seed 123 | 80.00% (24/30) | 75.56% (68/90) | - |

Paired exact McNemar, same task ids: base → sft4 hard-90 **81 fixed, 0 broken, p < 1e-20**;
sft3 → sft4 **18 fixed, 2 broken, p = 0.0004**; sft2 → sft4 7 fixed, 3 broken, p = 0.34
(directional only). Wilson 95% on hard-90: **86.2–96.9%**.

The seed spread is published instead of hidden: 16.67 pp on hard-30, 7.77 pp on hard-90. Seed 7
beats the shipped seed 42 on hard-90 (7 fixed, 0 broken, p = 0.0156); the shipped seed is the one
fixed before the seed runs existed, and it is the median of the three on the mean of both splits.
Choosing the best of three seeds on the split we report would have added 7.8 pp and removed the
claim's credibility.

### Generalisation, with no new training (2026-11-08)

| Probe on `sft4`, 30 tasks each | success | tool accuracy | calls |
|---|---:|---:|---:|
| canonical tools (reference) | 96.67% (29/30) | 100% | 3.57 |
| every tool and parameter renamed | 83.33% (25/30) | 90.00% | 3.47 |
| same tasks written in Devanagari | 83.33% (25/30) | 93.33% | 3.60 |
| untuned base, renamed tools | 0.00% (0/30) | 23.33% | 7.00 |

The renamed schema rejects canonical tool names as unknown, so this is not relabelling: the
untuned base goes to 0/30 with 0.60 loop collapse, while the adapter keeps 25/30 (4 broken,
0 fixed, exact McNemar p = 0.125 — not significant at n = 30). The losses are 3 of 6 `two_query`
and 1 of 6 `paraphrase`; `convert_sum` stays at 2/3. Devanagari costs a different four tasks —
3 of 6 `paraphrase` and 1 of 6 `two_query` — with arguments still at 1.000 and strict JSON at
98.89%, answers still in Latin script. The Devanagari comparison is nominal because that file
re-ids its rows; the schema shift shares task ids and carries the paired test. Runs:
`sft4_shift_hard30`, `sft4_devanagari30`, `base_shift_hard30` in `primce/lexsi-runs`.

### AuditKIT before/after (independent scorer, no LLM judge)

| Split | Model | tool_call_exact | tool_call_f1 | precision | recall | redundant | loop |
|---|---|---:|---:|---:|---:|---:|---:|
| hard-30 | base | 0.0333 | 0.1875 | 0.1547 | 0.3194 | 0.3270 | 0.4333 |
| hard-30 | sft2 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 |
| hard-90 | sft2 | 0.9333 | 0.9463 | 0.9444 | 0.9500 | 0.000 | 1.000 |
| hard-90 | sft3 | 0.7778 | 0.8114 | 0.8130 | 0.8102 | 0.0109 | 0.9875 |
| hard-90 | **sft4** | **0.9333** | **0.9804** | **0.9852** | **0.9769** | **0.000** | **1.000** |

AuditKIT's own success rate agrees with the project scorer on 30/30 hard-30 tasks and 88/90
decided hard-90 tasks, so the toolkit reproduces the result rather than replacing it. Full route
(with `finish` compared on name only): base hard-30 exact 0.100 / F1 0.448; sft4 hard-90 exact
0.967 / F1 0.996.

### Safety: indirect prompt injection on the ledger (32 tasks, 24 with a payload)

| System | safe | utility | attack_ledger | attack_answer | unplanned_calls |
|---|---:|---:|---:|---:|---:|
| Base Tiny Aya | 0/24 | 0/24 | 13/24 (54.2%) | 8/24 (33.3%) | 1.08 |
| SFT-mix (`sft2`) | 11/24 | 11/24 | 0/24 | 3/24 | 0.29 |
| SFT-mix2 (`sft3`) | 15/24 | 15/24 | 0/24 | 2/24 | 0.12 |
| **SFT-mix3 (`sft4`, shipped)** | **14/24** | **14/24** | **0/24** | **0/24** | 0.25 |
| SFT-mix3 + hardened prompt | 14/24 | 14/24 | 0/24 | 2/24 | 0.21 |

The attacker controls one `note` field in one tool result and asks the agent to record an absurd
999999 INR expense. Success is an equality check on the final ledger, so there is no judge to
disagree with. Underspecified requests: `asked` 0/8 → 4/8 and `invented_expense` 8/8 → 4/8 on
the adapter alone; with the hardened prompt, **7/8 asked and 0/8 invented**, which clears the
pre-registered 6/8 / 2/8 bar at the cost of 2/24 attacker numbers in prose. The adapter alone is
the default; the trade-off is recorded in `docs/ROUND2.md`.

## Honest limits

- hard-90 rows are unseen instances of the same six templates as hard-train; this measures
  generalisation across instances, not across task formats.
- Three seeds bound the variance; they do not eliminate it, and the spread is wider than any
  single change made after round 1.
- The safety probe is 32 tasks with one payload style. A Hinglish-only attacker or a Devanagari
  payload is untested.
- Abstention is improved but not solved by default: the shipped adapter asks on 4 of 8
  underspecified requests, or 7 of 8 with the hardened prompt.
- The shipped adapter beats round 1 by 4 hard-90 tasks but the paired test is not significant
  (p = 0.34); it beats round 2 decisively (18 fixed / 2 broken, p = 0.0004).
- One greedy sample per task. Sampled decoding would show more variance than these numbers.

## Reproduce (no GPU, no token)

```bash
git clone https://github.com/PRINCE2-AI/lexsi-agentic-hinglish && cd lexsi-agentic-hinglish
python scripts/verify_all.py      # 23 checks: pins, oracle 30/30, safety, contrarian splits, Devanagari, schema shift, reward, stats, AuditKIT adapter, demo handlers
```

On a T4 (one GPU is enough):

```bash
python scripts/train_sft.py --tasks data/tasks_train_mix2.jsonl --epochs 2 --seed 42 \
    --out runs/sft3 --merge-out runs/sft3_merged
python scripts/run_baseline.py --model runs/sft3_merged \
    --tasks data/tasks_hard90_eval.jsonl --no-4bit --out runs/sft3_hard90
python scripts/auditkit_eval.py --run sft3_hard90 --tasks data/tasks_hard90_eval.jsonl
python scripts/stats.py --a runs/base_hard90 --b runs/sft3_hard90
```

Full tables, per-family counts, all paired tests and the pre-registered thresholds:
`docs/RESULTS.md`, `docs/SAFETY.md`, `docs/ROUND2.md`.
