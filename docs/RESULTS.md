# Frozen results — sft2 (submission candidate)

**Frozen on 2026-11-03.** `runs/sft2_merged` + its LoRA adapter is the submission
candidate. Every number comes from a Kaggle T4 x2 run: training and the tuned evals from
`laxsi-ai-round2.ipynb`, the base-model before-column and the reproducibility re-run from
`laxsi-ai-base-bench.ipynb` (2026-11-04).

## Frozen artifact and provenance

| Item | Value |
|---|---|
| Training / eval code | `PRINCE2-AI/lexsi-agentic-hinglish` @ `b9574c9` |
| Adapter | `primce/tiny-aya-hinglish-lora` → `sft2_adapter` |
| Eval logs | `primce/lexsi-runs` → `base_hard30`, `sft2_hard30`, `sft2_eval`, plus the 2026-11-04 re-runs `sft2_hard30_v2`, `sft2_eval_v2` |
| Hard eval set | `data/tasks_hard_eval.jsonl`, sha256 `37f52b47937d231cce468c2b222d6e979ac51dffd116705fb98151b4a5056865` |
| Base model | `CohereLabs/tiny-aya-global` (3.35B) |

The 2026-11-04 run loaded the published adapter, merged it with `merge_and_unload()` and
re-ran hard-30: `success` 0.9667, `calls` 3.57, `strict_rate` 0.9833 — identical to the
frozen run. Anyone can reproduce the submission number from the public adapter alone.

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

| Stage | Data | Epochs / steps | Train time | Notes |
|---|---|---|---:|---|
| SFT-easy | 100 easy rows | 3 epochs, 75 steps | 1595 s | LoRA r=16, all-linear, fp16, lr 2e-4 |
| GRPO | 184 mix rows | 30 steps (T4 profile caps 60 → 30) | 10 686 s | lr 1e-6, graded + efficiency rewards, 2 generations |
| **SFT-mix (frozen)** | 184 mix rows | 2 epochs, 92 steps | 2252 s | same SFT config as SFT-easy, then merged |

## Results

Held-out tasks, greedy decoding, `--no-4bit`, 0 tool errors in every run.

| Model | hard-30 success | eval-24 success | answer_ok (hard) | calls (hard) | strict_rate (hard) |
|---|---:|---:|---:|---:|---:|
| Base Tiny Aya, prompt-only | 3.33% (1/30) | - | 6.67% | 7.03 | 0.534 |
| SFT-easy | 63.33% (19/30) | 100% (24/24) | 66.67% | 3.47 | 1.000 |
| SFT-easy + GRPO | 63.33% (19/30) | 100% (24/24) | 66.67% | 3.47 | 1.000 |
| **SFT-mix (frozen)** | **96.67% (29/30)** | **100% (24/24)** | **96.67%** | **3.57** | 0.983 |

Oracle ceiling on hard-30 is 30/30 with 3.57 calls, so the frozen model reaches oracle
call efficiency exactly.

The base model's single success is not a clean one: on `h-e-029` it answered "0 INR"
correctly, but only after inventing an 800 INR expense nobody mentioned. It also averaged
7.03 calls against the oracle's 3.57, with 16 of 30 episodes never reaching `finish`.

Run timings on the same T4 x2, greedy, `--no-4bit`: base hard-30 2428.4 s, tuned hard-30
1648.6 s, tuned eval-24 1212.5 s. The tuned model is both more accurate and faster because
it stops at the oracle call count instead of looping.

## Safety split

The repo also ships a 32-task injection eval. Headline: base writes attacker money into the
ledger in 13/24 injection tasks, the frozen adapter in 0/24 (exact McNemar p = 0.00024), while
utility under attack rises from 0/24 to 11/24. Abstention on underspecified requests stays at
1/8 for both tuned variants — the open problem. Full table, per-family counts and the threat
model: `docs/SAFETY.md`.

Run logs: `primce/lexsi-runs` → `safety_base`, `safety_sft2`, `safety_sft2_hardened`.

## What moved the number

Base Tiny Aya solves 1/30. Its tool error rate is 22.3%, `overcall` 1.075 and 16 episodes
never finish, so the before-column is a genuine failure, not a close miss.

SFT-easy then left three hard families at zero: two_query 0/6, convert_sum 0/3,
empty_balance 2/3. Tool calls were already correct in those episodes — the final answer
dropped the second value (answer_ok 66.67%). GRPO on top of SFT-easy moved nothing: 30
steps is ~0.33 epoch of the mix, and with a saturated graded reward both generations in a
group scored alike, so there was no advantage to learn from.

Reweighting the same 40 hard rows into an SFT run fixed it in 92 steps: +33.3 points on
hard-30 with eval-24 unchanged. The bottleneck was data coverage of the answer shape,
not the RL stage.

## Statistics

| Model | hard-30 success | Wilson 95% CI |
|---|---:|---|
| Base Tiny Aya | 3.33% (1/30) | 0.6% – 16.7% |
| SFT-easy | 63.33% (19/30) | 45.5% – 78.1% |
| SFT-easy + GRPO | 63.33% (19/30) | 45.5% – 78.1% |
| **SFT-mix (frozen)** | **96.67% (29/30)** | **83.3% – 99.4%** |

eval-24 is 100% (24/24) with a Wilson CI of 86.2% – 100%. The 30-task denominator is
small, which is exactly why the interval is reported instead of the point estimate alone.

Paired exact McNemar on the uploaded trajectories:

- base → SFT-mix (`base_hard30` vs `sft2_hard30_v2`): **28 fixed, 0 broken, two-sided
  p = 7.45e-09**.
- easy-SFT + GRPO → SFT-mix (`grpo_hard30` vs `sft2_hard30`): **10 fixed, 0 broken,
  two-sided p = 0.00195**.

Both improvements are nested — every hard task the earlier model passed, the frozen model
still passes.

```bash
python scripts/stats.py --a runs/base_hard30 --b runs/sft2_hard30
python scripts/stats.py --a runs/grpo_hard30 --b runs/sft2_hard30
```

Per family, same paired files:

| Family | n | Base | SFT-easy + GRPO | SFT-mix |
|---|---:|---:|---:|---:|
| two_query | 6 | 0/6 | 0/6 | **6/6** |
| convert_sum | 3 | 0/3 | 0/3 | **2/3** |
| empty_balance | 3 | 1/3 | 2/3 | **3/3** |
| trap_combo | 6 | 0/6 | 5/6 | **6/6** |
| settle5 | 6 | 0/6 | 6/6 | 6/6 |
| paraphrase | 6 | 0/6 | 6/6 | 6/6 |

The one remaining failure is `h-e-027` (`convert_sum`: "Total kitna INR hoga agar 19 USD aur
37 EUR convert karein?"), so the next data edit should target that template, not the others.

## Caveats to state in the write-up

- hard-30 rows are **held-out instances** (new amounts, names, currencies) of the same six
  templates seen in hard-train. Generalization is across instances, not across formats.
- eval-24 is in-distribution with easy-train, which is why it stays at 100%.
- `strict_rate` 0.983 means one assistant turn in the run was not bare JSON; `strict`
  (all-turns-clean) per task is in `sft2_hard30/summary.json`.
- `overcall` and `parse_failures` were added on 2026-11-03, after the frozen eval ran, so
  the frozen `summary.json` files do not contain them. Recomputed from the uploaded
  trajectories they are already 0: frozen `overcall` 0.00 and 0 parse failures, against
  0.0167 and 0 for the SFT-easy + GRPO run. The next eval writes them directly.
- Full per-metric summaries (`tac`, `arr`, `ter`, `rad`, `lcf`) are in the uploaded
  `summary.json` files, not repeated here.

## Reproduce

```bash
python data/build_hard.py
python data/build_mix.py
python scripts/run_baseline.py --model CohereLabs/tiny-aya-global \
    --tasks data/tasks_hard_eval.jsonl --no-4bit --out runs/base_hard30
python scripts/train_sft.py --tasks data/tasks_train_mix.jsonl \
    --epochs 2 --out runs/sft2 --merge-out runs/sft2_merged
python scripts/run_baseline.py --model runs/sft2_merged \
    --tasks data/tasks_hard_eval.jsonl --no-4bit --out runs/sft2_hard30
python scripts/run_baseline.py --model runs/sft2_merged \
    --tasks data/tasks_eval.jsonl --no-4bit --out runs/sft2_eval
python scripts/stats.py --a runs/base_hard30 --b runs/sft2_hard30
```

No GPU is needed to confirm the data: `python scripts/train_sft.py --tasks
data/tasks_train_mix.jsonl --check` prints `rows: 184  assistant turns: 630`.
