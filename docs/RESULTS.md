# Frozen results — sft2 (submission candidate)

**Frozen on 2026-11-03.** `runs/sft2_merged` + its LoRA adapter is the submission
candidate. Nothing below is estimated; every number comes from a Kaggle T4 x2 run of
`laxsi-ai-round2.ipynb`.

## Frozen artifact and provenance

| Item | Value |
|---|---|
| Training / eval code | `PRINCE2-AI/lexsi-agentic-hinglish` @ `b9574c9` |
| Adapter | `primce/tiny-aya-hinglish-lora` → `sft2_adapter` |
| Eval logs | `primce/lexsi-runs` → `sft2_hard30`, `sft2_eval` |
| Hard eval set | `data/tasks_hard_eval.jsonl`, sha256 `37f52b47937d231cce468c2b222d6e979ac51dffd116705fb98151b4a5056865` |
| Base model | `CohereLabs/tiny-aya-global` (3.35B) |

The hard-30 pin is checked inside `data/build_hard.py`, so an eval-set drift fails the
run instead of silently changing the denominator.

## Data

| Split | Rows | Families |
|---|---:|---|
| `tasks_train.jsonl` (easy) | 100 | single/multi expense, balance, settle, convert |
| `tasks_eval.jsonl` (easy, held out) | 24 | same distribution as easy train |
| `tasks_hard_train.jsonl` | 40 | settle5, two_query, convert_sum, empty_balance, trap_combo, paraphrase |
| `tasks_hard_eval.jsonl` (held out) | 30 | 6 families × 5: settle5 6, two_query 6, convert_sum 3, empty_balance 3, trap_combo 6, paraphrase 6 |
| `tasks_train_mix.jsonl` | 184 | easy 100 + hard 84 (hard rows repeated per family) |

Mix family share: easy 0.543, two_query 0.13, convert_sum 0.087, trap_combo 0.087,
empty_balance 0.065, settle5 0.043, paraphrase 0.043. A hard-train row that appears in
the hard-eval set aborts `data/build_mix.py`, so the repetition cannot leak eval rows.

## Training runs

| Stage | Data | Epochs / steps | Wall time | Notes |
|---|---|---|---:|---|
| SFT-easy | 100 easy rows | 3 epochs, 75 steps | 1595 s | LoRA r=16, all-linear, fp16, lr 2e-4 |
| GRPO | 184 mix rows | 30 steps (T4 profile caps 60 → 30) | 10 686 s | lr 1e-6, graded + efficiency rewards, 2 generations |
| **SFT-mix (frozen)** | 184 mix rows | 2 epochs, 92 steps | 2252 s | same SFT config as SFT-easy, then merged |

## Results

Held-out tasks, greedy decoding, `--no-4bit`, 0 tool errors in every run.

| Model | hard-30 success | eval-24 success | answer_ok (hard) | calls (hard) | strict_rate (hard) |
|---|---:|---:|---:|---:|---:|
| SFT-easy | 63.33% (19/30) | 100% (24/24) | 66.67% | 3.47 | 1.000 |
| SFT-easy + GRPO | 63.33% (19/30) | 100% (24/24) | 66.67% | 3.47 | 1.000 |
| **SFT-mix (frozen)** | **96.67% (29/30)** | **100% (24/24)** | **96.67%** | **3.57** | 0.983 |

Oracle ceiling on hard-30 is 30/30 with 3.57 calls, so the frozen model reaches oracle
call efficiency exactly.

## What moved the number

SFT-easy left three hard families at zero: two_query 0/6, convert_sum 0/3, empty_balance
2/3. Tool calls were already correct in those episodes — the final answer dropped the
second value (answer_ok 66.67%). GRPO on top of SFT-easy moved nothing: 30 steps is
~0.33 epoch of the mix, and with a saturated graded reward both generations in a group
scored alike, so there was no advantage to learn from.

Reweighting the same 40 hard rows into an SFT run fixed it in 92 steps: +33.3 points on
hard-30 with eval-24 unchanged. The bottleneck was data coverage of the answer shape,
not the RL stage.

## Caveats to state in the write-up

- hard-30 rows are **held-out instances** (new amounts, names, currencies) of the same six
  templates seen in hard-train. Generalization is across instances, not across formats.
- eval-24 is in-distribution with easy-train, which is why it stays at 100%.
- `strict_rate` 0.983 means one assistant turn in the run was not bare JSON; `strict`
  (all-turns-clean) per task is in `sft2_hard30/summary.json`.
- Full per-metric summaries (`tac`, `arr`, `ter`, `rad`, `lcf`) are in the uploaded
  `summary.json` files, not repeated here.

## Reproduce

```bash
python data/build_hard.py
python data/build_mix.py
python scripts/train_sft.py --tasks data/tasks_train_mix.jsonl \
    --epochs 2 --out runs/sft2 --merge-out runs/sft2_merged
python scripts/run_baseline.py --model runs/sft2_merged \
    --tasks data/tasks_hard_eval.jsonl --no-4bit --out runs/sft2_hard30
python scripts/run_baseline.py --model runs/sft2_merged \
    --tasks data/tasks_eval.jsonl --no-4bit --out runs/sft2_eval
```

No GPU is needed to confirm the data: `python scripts/train_sft.py --tasks
data/tasks_train_mix.jsonl --check` prints `rows: 184  assistant turns: 630`.
