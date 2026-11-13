---
language:
  - hi
  - en
license: cc-by-nc-4.0
tags:
  - agent
  - tool-calling
  - evaluation
  - hinglish
  - safety
size_categories:
  - n<1K
configs:
  - config_name: results
    data_files:
      - split: train
        path: viewer/results.parquet
  - config_name: runs
    data_files:
      - split: train
        path: viewer/runs.parquet
---

# lexsi-runs

Run logs for the Hinglish expense-splitting agent (Lexsi AI Safety & Alignment Hackathon 2026,
Track 3). Every reported number in the write-up comes from `summary.json` in this dataset, and every
trajectory can be re-scored offline without a GPU or a model download. A 1:44 narrated demo
(`demo/lexsi_track3_demo.mp4`) walks through the live Space and the before/after numbers.

## Browse it here

Two parquet tables make the logs clickable in the Hub viewer, so nothing has to be downloaded to
check a number:

| Config | Rows | One row is |
|---|---:|---|
| `results` | 1,794 | one task in one run: `run`, `task_id`, `family`, `success`, plus every per-task metric (`tac`, `arr`, `ter`, `rad`, `overcall`, `lcf`, `answer_ok`, `ledger_ok`, `strict`, `calls`, `steps`, `parse_failures`, and the attack counters where they exist) |
| `runs` | 36 | one run: the same fields aggregated, with `n`, `seconds`, `model` and `adapter` |

Both are regenerated from `runs/*/results.jsonl` and `runs/*/summary.json` by
`python scripts/publish_viewer.py`, so the tables and the files cannot disagree.

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
| `sft4_hard90_hardened/` | shipped adapter on clean hard-90 with the hardened system prompt (86/90 vs 84/90 default) |
| `sft4_shift_hard30/` | shipped adapter on hard-30 with every tool and parameter renamed (25/30 vs 29/30 canonical) |
| `sft4_devanagari30/` | shipped adapter on the Devanagari version of hard-30, canonical schema (25/30) |
| `base_shift_hard30/` | untuned base under the same rename (0/30, loop collapse 0.60) |
| `sft5_hard30/`, `sft5_hard90/`, `sft5_eval/`, `sft5_sanity/` | mixture ablation (fewer ask rows) |
| `sft3_seed7_hard30/`, `sft3_seed7_hard90/`, `sft3_seed123_hard30/`, `sft3_seed123_hard90/` | seed control on the mix2 recipe |
| `sft4_seed7_hard30/`, `sft4_seed7_hard90/`, `sft4_seed123_hard30/`, `sft4_seed123_hard90/` | mix3 seed control (seeds 7 and 123) for the shipped seed 42 |
| `safety_base/`, `safety_sft2/`, `safety_sft3/`, `safety_sft4/`, `safety_sft5/`, `safety_grpo3/` | 32-task injection split, default prompt |
| `safety_sft2_hardened/`, `safety_sft3_hardened/`, `safety_sft4_hardened/`, `safety_sft5_hardened/` | same split, hardened system prompt |
| `safety2_base/`, `safety2_sft4/`, `safety2_sft4_hardened/` | second injection sample, 80 tasks with 72 payloads, seed 20261108 (base 48/72 ledger accounting, adapter 4/72; the `attack_ledger` column in these runs is the legacy threshold) |
| `heldout_base/`, `heldout_sft4/`, `heldout_sft4_hardened/` | 30-task held-out attack-style split, seed 20261110: split payloads and obfuscated amounts (adapter 5/30 ledger, 0/30 answer) |
| `safety_payee_base/`, `safety_payee_sft4/`, `safety_payee_sft4_hardened/` | 16 payee-redirect attacks: the tool result claims the sender's UPI handle changed; scored on the answer (base 5/16 ledger, adapter 0/16) |
| `safety_hinglish_base/`, `safety_hinglish_sft4/`, `safety_hinglish_sft4_hardened/` | 24 attacks with Hinglish/Devanagari payloads only (18 Latin, 6 Devanagari, 8 per family; base 10/24 ledger and 5/24 answer, adapter 0/24 and 0/24, hardened 24/24 solved) |
| `safety_sft4_neutralise/`, `safety_sft4_strip/` | the same 136 injected rows with the guard in front of the adapter: 0/136 expense, 0/136 answers, 136/136 solved (10,567 s for `strip`) |
| `safety_adaptive72_sft4/`, `safety_adaptive72_sft4_guard/` | 72-task weakness split built after the guard's rules were read (spelled, split across two results, expression): adapter alone 47/72 solved with 7 extra expenses, with the guard 72/72 solved and 0 harm |
| `auditkit_*` | AuditKIT replays of the same recorded episodes, no judge model (scored on 1.1.3, re-checked on 1.1.6, same verdicts) |
| `demo/` | `lexsi_track3_demo.mp4` — 1:44 narrated demo: the live Space, the injection before/after, and the hard-90 numbers |

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
| Safety eval 2 (80) | `data/tasks_safety_eval2.jsonl`, sha256 `bc226c7a…b12138` |
| Safety payee (16) | `data/tasks_safety_payee.jsonl`, sha256 `b22f6819…e447a4` |
| Safety Hinglish payloads (24) | `data/tasks_safety_hinglish.jsonl`, sha256 `7ed61572…802e95` |

The two hard splits share no task text; hard-90 was built afterwards as a wider sample of the same
six families and is pinned separately. Both hard splits and all three safety splits carry a
SHA-256 file in the repo, so a number cannot silently drift to a different task set.

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
- Two seed controls exist, on two recipes. The mix2 spread (`sft3_seed7`, `sft3_seed123` vs
  `sft3_hard90`) is 7.77 pp on hard-90. The mix3 recipe that ships has three seeds - `sft4_hard90`
  (seed 42), `sft4_seed7_hard90`, `sft4_seed123_hard90` - spanning 1.11 pp on hard-90
  (92.22-93.33%) and 3.33 pp on hard-30 (96.67-100%), with no seed pair significantly different
  (exact McNemar p = 1.0 in every pair) and five tasks missed by all three seeds.

## Attribution

Built on AgentTune and AuditKIT (Lexsi Labs) with Cohere Labs Tiny Aya.
