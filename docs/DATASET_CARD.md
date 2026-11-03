---
language:
  - hi
  - en
tags:
  - agent
  - tool-calling
  - evaluation
  - hinglish
  - safety
size_categories:
  - n<1K
---

# lexsi-runs

Run logs for the Hinglish expense-splitting agent (Lexsi AI Safety & Alignment Hackathon 2026,
Track 3). Every reported number in the write-up comes from `summary.json` in this dataset, and every
trajectory can be re-scored offline without a GPU or a model download.

## Layout

| Folder | What it is |
|---|---|
| `baseline/` | base Tiny Aya, 20 starter tasks |
| `sft_hard30/` | SFT on the easy split, 30 hard tasks |
| `grpo_hard30/` | SFT + GRPO, same 30 hard tasks |
| `sft2_hard30/` | frozen adapter (`sft2_adapter`), 30 hard tasks |
| `sft2_eval/` | frozen adapter, 24 in-distribution tasks |
| `oracle/` | ground-truth replay, 30 hard tasks |
| `safety_base/` | base model on the 32-task injection split |
| `safety_sft2/` | frozen adapter on the injection split |
| `safety_sft2_hardened/` | frozen adapter + hardened system prompt |

Each folder holds:

- `trajectories.jsonl` — one row per task: `task`, `steps` (tool name, arguments, result),
  `final_response`, and `metadata` with the full conversation and per-task scores. This is the
  AgentTune/AuditKIT-compatible format.
- `results.jsonl` — the per-task score rows.
- `summary.json` — the aggregate, including `seconds` (wall time) and, for safety runs, a
  per-family breakdown.

## Provenance

| Field | Value |
|---|---|
| Base model | `CohereLabs/tiny-aya-global` (3.35B, gated) |
| Adapter | `primce/tiny-aya-hinglish-lora`, subfolder `sft2_adapter`, tag `sft2-frozen` |
| Code | `github.com/PRINCE2-AI/lexsi-agentic-hinglish`, commit `8c78979` at freeze |
| Decoding | greedy, fp16 (`--no-4bit`), max 8 agent steps, max 256 new tokens |
| Hardware | Kaggle T4 x2, one process per run |
| Hard eval | `data/tasks_hard_eval.jsonl`, sha256 `37f52b47…506865` |
| Safety eval | `data/tasks_safety_eval.jsonl`, sha256 `bb8a3858…a7fe` |

## Load one run

```python
import json, urllib.request

URL = "https://huggingface.co/datasets/primce/lexsi-runs/resolve/main/sft2_hard30/trajectories.jsonl"
rows = [json.loads(line) for line in urllib.request.urlopen(URL).read().decode().splitlines()]

row = rows[0]
print(row["task"])
for step in row["steps"]:
    print(step["name"], step["arguments"], step["result"].get("ok"))
print(row["metadata"]["scores"]["success"], row["final_response"])
```

## Re-score without a model

The scoring functions are pure Python, so the numbers are reproducible from the trajectories alone:

```bash
python scripts/stats.py --a runs/sft_hard30 --b runs/sft2_hard30   # Wilson CI + McNemar
python scripts/safety_table.py                                      # injection comparison
```

## Notes and caveats

- These are evaluation logs, not training data. The training split lives in the GitHub repo
  (`data/tasks_*.jsonl`) and is regenerated deterministically from a fixed seed.
- Scores in `metadata.scores` were written by the code at the freeze commit. `overcall` and
  `parse_failures` in the earlier runs were recomputed post-hoc; the `_v2` runs carry them natively.
- Safety runs are single-sample greedy. An attack-success rate of 0% here is not a safety proof.

## Attribution

Built on AgentTune and AuditKIT (Lexsi Labs) with Cohere Labs Tiny Aya.
