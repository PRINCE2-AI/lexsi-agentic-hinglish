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

Frozen 2026-11-03. Greedy decoding, single sample, no LLM judge — success is checked against the
final ledger state.

| Run | Split | Task success |
|---|---|---:|
| Base Tiny Aya, prompt-only | 20 easy tasks | 25.00% |
| SFT on easy split | 30 hard tasks | 63.33% |
| + GRPO (30 steps, graded reward) | 30 hard tasks | 63.33% (no gain) |
| **SFT on the reweighted mix (this adapter)** | **30 hard tasks** | **96.67%** |
| Same adapter | 24 in-distribution tasks | 100.00% |

- Paired McNemar over the 30 shared hard tasks: **10 fixed, 0 broken, two-sided p = 0.00195**.
- Tool calls per task: **3.57** (the oracle needs 3.5).
- Tool error rate **0.00%**, loop collapse **0.00%**, unnecessary calls **0.00%**.

The hard split is 30 tasks the model never saw, spanning six families: five-way settlement with
paise remainders and two currencies, balance-plus-settlement in one turn, payer-not-a-participant
traps, paraphrase, two-currency conversion sums, and empty-ledger questions. The file is SHA-256
pinned in the repo, so the number cannot drift.

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

Results, thresholds and the threat model: `docs/SAFETY.md` in the repo.

## Training

- Data: 184 oracle-verified Hinglish tasks (100 easy + 84 hard), family-reweighted.
- Method: 2 epochs of supervised fine-tuning on oracle trajectories, then merged into the base
  weights. GRPO was run and did not improve the score; the write-up explains why.
- Output format: a bare JSON array of one tool call per turn, no prose, no markdown fences.

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("CohereLabs/tiny-aya-global")
model = AutoModelForCausalLM.from_pretrained("CohereLabs/tiny-aya-global")
model = PeftModel.from_pretrained(model, "primce/tiny-aya-hinglish-lora", subfolder="sft2_adapter")
```

The adapter expects the system prompt from `hinglish_agent/prompts.py` (tool schemas inline, one
tool call per turn, ledger returned by every ledger tool and copied into the next call).

## Limitations

- Domain-specific: it answers ledger questions with exact arithmetic; it is not a general assistant.
- 30 hard tasks is a probe. Every number carries a wide Wilson interval and there is no seed sweep.
- The adapter is not trained to abstain. On underspecified questions it tends to guess; the safety
  split measures exactly that, and a hardened system prompt is the tested mitigation.
- Hinglish + English only. Devanagari paraphrases appear in the eval, but generation is not
  evaluated in Devanagari.

## Licence and provenance

Base weights: CohereLabs Tiny Aya, used under its licence. Adapter trained by the entry author on
synthetic, oracle-verified data. Code, data, pins and run logs:
`github.com/PRINCE2-AI/lexsi-agentic-hinglish`, `primce/lexsi-runs`.
