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

| Folders | What they are |
|---|---|
| `base_hard30/`, `base_hard90/` | base Tiny Aya, prompt-only, on both hard splits |
| `oracle/` | ground-truth replay of the oracle actions, 30 hard tasks |
| `sft_hard30/`, `sft_eval/` | SFT on the easy split (pre-round-1) |
| `grpo_hard30/`, `grpo_eval/`, `grpo3_hard30/`, `grpo3_hard90/` | GRPO on top of SFT, two rounds |
| `sft2_hard30/` (+`_v2`), `sft2_eval/` (+`_v2`), `sft2_hard90/` | round-1 adapter |
| `sft3_hard30/`, `sft3_hard90/`, `sft3_eval/` | round-2 adapter (mix2) |
| `sft4_hard30/`, `sft4_hard90/`, `sft4_eval/` | **round-3 adapter, the one that ships** |
| `sft5_hard30/`, `sft5_hard90/`, `sft5_eval/`, `sft5_sanity/` | mixture ablation (fewer ask rows) |
| `sft3_seed7_hard30/`, `sft3_seed7_hard90/`, `sft3_seed123_hard30/`, `sft3_seed123_hard90/` | seed control on the mix2 recipe |
| `safety_base/`, `safety_sft2/`, `safety_sft3/`, `safety_sft4/`, `safety_sft5/`, `safety_grpo3/` | 32-task injection split, default prompt |
| `safety_sft2_hardened/`, `safety_sft3_hardened/`, `safety_sft4_hardened/`, `safety_sft5_hardened/` | same split, hardened system prompt |
| `auditkit_*` | AuditKIT 1.1.3 replays of the same recorded episodes, no judge model |

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
| Base model | `CohereLabs/tiny-aya-global` (3.35B) |
| Shipped adapter | `primce/tiny-aya-hinglish-lora`, subfolder `sft4_adapter` |
| Code | `github.com/PRINCE2-AI/lexsi-agentic-hinglish`, commit `78876f7` for the shipped round |
| Decoding | greedy, fp16 (`--no-4bit`), max 8 agent steps, max 256 new tokens |
| Hardware | rented T4 (HF Jobs) and one Kaggle T4; on a T4 x2 machine `CUDA_VISIBLE_DEVICES=0` pins one card |
| Hard eval (30) | `data/tasks_hard_eval.jsonl`, sha256 `37f52b47…506865` |
| Hard eval (90) | `data/tasks_hard90_eval.jsonl`, sha256 `2a4fc8fa…5e8877c` |
| Safety eval (32) | `data/tasks_safety_eval.jsonl`, sha256 `bb8a3858…a84a7fe` |

The two hard splits share no task text; hard-90 was built afterwards as a wider sample of the same
six families and is pinned separately. Both hard splits and the safety split carry a SHA-256 file
in the repo, so a number cannot silently drift to a different task set.

## Load one run

```python
import json, urllib.request

URL = "https://huggingface.co/datasets/primce/lexsi-runs/resolve/main/sft4_hard90/trajectories.jsonl"
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
python scripts/stats.py --a runs/sft3_hard90 --b runs/sft4_hard90   # Wilson CI + exact McNemar
python scripts/safety_table.py                                      # injection comparison
python scripts/auditkit_eval.py --run sft4_hard90 --tasks data/tasks_hard90_eval.jsonl
```

`scripts/job_upload.py` is what wrote these folders: it prints each run's `summary.json` keys and
uploads the folder plus the adapter subfolder, so a re-run reproduces the same layout.

## Notes and caveats

- These are evaluation logs, not training data. The training split lives in the GitHub repo
  (`data/tasks_*.jsonl`) and is regenerated deterministically from a fixed seed.
- Scores in `metadata.scores` were written by the code at the freeze commit. `overcall` and
  `parse_failures` in the earlier runs were recomputed post-hoc; the `_v2` runs carry them natively.
- Safety runs are single-sample greedy. An attack-success rate of 0% here is not a safety proof.
- The seed control is on the mix2 recipe (`sft3_seed*`). The mix3 recipe that ships has one seed;
  a mix3 re-train finished on Kaggle but its held-out evaluation was cut off by the GPU quota, so
  no mix3 seed folder exists and none is claimed.

## Attribution

Built on AgentTune and AuditKIT (Lexsi Labs) with Cohere Labs Tiny Aya.
