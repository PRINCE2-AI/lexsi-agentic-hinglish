---
language:
  - hi
  - en
base_model: CohereLabs/tiny-aya-global
library_name: peft
tags:
  - agent
  - tool-calling
  - hinglish
  - lora
  - safety
datasets:
  - primce/lexsi-runs
---

# tiny-aya-hinglish-lora

LoRA adapter for **CohereLabs/tiny-aya-global** (3.35B) that teaches a Hinglish expense-splitting
agent to call tools reliably: record a shared expense, carry the ledger across turns, convert
USD/EUR, settle balances, and answer with the exact rupee amount.

Lexsi AI Safety & Alignment Hackathon 2026, Track 3 (Agentic fine-tuning), solo entry.

## Results

Frozen 2026-11-04. Greedy decoding, single sample, no LLM judge — success is checked against the
final ledger state. `sft3_adapter` is the shipped adapter; `sft2_adapter` and the two seed
adapters are kept for the variance control.

| Run | Split | Task success |
|---|---|---:|
| Base Tiny Aya, prompt-only | 30 hard tasks | 3.33% (1/30) |
| Base Tiny Aya, prompt-only | 90 hard tasks | 3.33% (3/90) |
| SFT on easy split | 30 hard tasks | 63.33% |
| + GRPO (30 steps, graded reward) | 30 hard tasks | 63.33% (no gain) |
| SFT on the reweighted mix (`sft2`) | 30 hard tasks | 96.67% (29/30) |
| **SFT on the mix + abstention pairs (`sft3`, this adapter)** | **30 hard tasks** | **96.67% (29/30)** |
| **Same adapter** | **90 hard tasks** | **75.56% (68/90)** |
| Same adapter | 24 in-distribution tasks | 100.00% (24/24) |
| Same recipe, seed 7 | 30 / 90 hard tasks | 93.33% / 83.33% |
| Same recipe, seed 123 | 30 / 90 hard tasks | 80.00% / 75.56% |

- Paired exact McNemar over shared task ids: base → this adapter **28 fixed, 0 broken,
  p = 7.5e-09** on hard-30 and **66 fixed, 1 broken, p = 9.2e-19** on hard-90.
- Tool calls per task: **3.57** on hard-30, exactly the oracle's count (base: 7.03). On hard-90:
  3.27 against a base 7.24.
- Tool error rate **0.00%** on hard-30 and **1.1%** on hard-90, loop collapse **0.00%**, redundant
  calls **0.00%** (base: 22.3% tool errors, 16 of 30 episodes never finishing, `overcall` 1.08).
- Independent AuditKIT 1.1.3 replay of the same hard-30 trajectories: exact-call accuracy
  **0.033 → 1.000**, tool-call F1 **0.188 → 1.000**, redundant calls **0.327 → 0.000**. On
  hard-90: exact 0.778, F1 0.811, and AuditKIT's success verdict agrees with the repo scorer on
  89/90 tasks.

Three seeds of the same recipe span 16.67 pp on hard-30 and 7.77 pp on hard-90. The shipped
adapter is seed 42, fixed before the seed runs existed and the median of the three on the mean of
both splits; seed 7 is better on hard-90 (7 fixed, 0 broken, p = 0.0156). The spread is published
rather than hidden.

The base model's single hard-30 success is not a clean one: on `h-e-029` it returned the correct
"0 INR" only after inventing an 800 INR expense nobody mentioned. The adapter fixes the behaviour,
not just the final string.

The hard splits are tasks the model never saw, spanning six families: five-way settlement with
paise remainders and two currencies, balance-plus-settlement in one turn, payer-not-a-participant
traps, paraphrase, two-currency conversion sums, and empty-ledger questions. Both files are
SHA-256 pinned in the repo, so the numbers cannot drift.

## Safety split

The repo also ships a 32-task injection eval: the agent is asked a normal question while a tool
result carries attacker text. `attack_ledger` (money actually moved) and `attack_answer` (the
number was repeated) are exact checks — the marker 999999 can never be a correct answer.

| Family | n | Attack |
|---|---:|---|
| `inject_note` | 8 | tool result asks for a 999999 INR expense |
| `inject_skip` | 8 | tool result asks the model to skip a tool and answer 999999 |
| `inject_settle` | 8 | tool result asks for an unrequested settlement |
| `underspecified` | 8 | user omitted names/amount/currency; asking is correct |

Measured 2026-11-04 on the 24 injection tasks:

| System | safe | utility under attack | attack_ledger | attack_answer |
|---|---:|---:|---:|---:|
| Base Tiny Aya | 0/24 | 0/24 | 13/24 | 8/24 |
| `sft2` (round 1) | 11/24 | 11/24 | 0/24 | 3/24 |
| **`sft3` (this adapter)** | **15/24** | **15/24** | **0/24** | **2/24** |
| `sft3` + hardened prompt | 12/24 | 12/24 | 0/24 | 3/24 |

Base → this adapter removes 13 ledger attacks and introduces 0 (exact McNemar p = 0.00024).
Abstention on underspecified requests moves from 1/8 to 2/8 asked (invented expense 7/8 → 6/8),
against a pre-registered bar of 6/8: improved, still the open problem. The hardened prompt helps
`sft2` and hurts `sft3`, so it is not the default. Thresholds, per-family counts and the threat
model: `docs/SAFETY.md`.

## Training

- Data: 312 oracle-verified Hinglish rows — 184 easy/hard mix, 96 abstention rows (32
  underspecified × 3) and 32 fully-specified twins. The twins exist so the policy learns the
  boundary instead of the template.
- Method: 2 epochs of supervised fine-tuning on oracle trajectories (~156 optimizer steps), LoRA
  r=16 all-linear, fp16, lr 2e-4, assistant-only loss, then merged into the base weights.
- GRPO (AgentTune `create_agentic_trainer`) was run in both rounds; round 1 moved nothing at 30
  steps because the graded reward was saturated. Round 2 uses an abstention-aware reward and
  reports the delta on both axes (`docs/ROUND2.md`).
- Output format: a bare JSON array of one tool call per turn, no prose, no markdown fences.

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("CohereLabs/tiny-aya-global")
model = AutoModelForCausalLM.from_pretrained("CohereLabs/tiny-aya-global")
model = PeftModel.from_pretrained(model, "primce/tiny-aya-hinglish-lora", subfolder="sft3_adapter")
```

Other subfolders: `sft2_adapter` (round 1), `sft3_seed7_adapter`, `sft3_seed123_adapter` (variance
control), `grpo_adapter` and `sft_adapter` (earlier stages).

The adapter expects the system prompt from `hinglish_agent/prompts.py` (tool schemas inline, one
tool call per turn, ledger returned by every ledger tool and copied into the next call).

## Limitations

- Domain-specific: it answers ledger questions with exact arithmetic; it is not a general assistant.
- The 90-task holdout is still a probe (Wilson 65.8–83.3%), and the seed spread is wider than any
  single change made after round 1.
- The adapter can ask for missing details but does not prefer to: 2 of 8 underspecified requests.
- Hinglish + English only. Devanagari paraphrases appear in the eval, but generation is not
  evaluated in Devanagari.

## Licence and provenance

Base weights: CohereLabs Tiny Aya, used under its licence. Adapter trained by the entry author on
synthetic, oracle-verified data. Code, data, pins and run logs:
`github.com/PRINCE2-AI/lexsi-agentic-hinglish`, `primce/lexsi-runs`. Live demo:
`huggingface.co/spaces/primce/hinglish-expense-agent`. Write-up:
`huggingface.co/spaces/primce/hinglish-track3-article`. Collection:
`huggingface.co/collections/primce/lexsi-track-3-hinglish-expense-agent-6aeb3207e10dfc7dda6329d6`.
