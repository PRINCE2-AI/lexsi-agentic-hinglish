# Results — sft3 (submission candidate)

**Frozen on 2026-11-04.** `runs/sft3` + its LoRA adapter is the submission candidate: the
round-1 SFT mix plus an abstention slice (`data/tasks_train_mix2.jsonl`, 312 rows), seed 42,
measured on a 90-task holdout, three seeds, the safety split and an independent AuditKIT pass.
The round-1 freeze (`sft2`, 2026-11-03) is kept unchanged further down as the before-column.

**Shipping note (2026-11-04, 16:20 UTC).** The `sft2` control row below arrived after this header
was written and it beats `sft3` on hard-90 by 12 tasks (88.89% vs 75.56%, p = 0.00418), losing only
the abstention slice. The adapter that ships is therefore undecided until `sft4` (mix3 + contrast
pairs) is evaluated; this file is updated in place when it lands, and the table above is the
current measurement, not the decision.

**Wording split (2026-11-04, 16:30 UTC).** Masking person names and matching hard-90 against the
training file shows the lost 12 tasks are lost on *new wording*, not new instances. On the 85 hard-90
tasks whose masked wording never appears in training, `sft2` is 75/85 (88.24%) and `sft3` is 63/85
(74.12%); on the 5 that do appear both are 5/5. All 11 of the difference is `empty_balance`: 4/4 seen
versus 0/11 unseen for `sft3`, against 4/4 and 11/11 for `sft2`. The `sft3` failures are not silence —
they emit the `ambiguous_share` abstention answer verbatim ("Wo teesra banda … Batao, phir split
karunga") on a question whose ledger is merely empty. Reproduce with
`python scripts/template_split.py --run sft3_hard90 --train data/tasks_train_mix2.jsonl`.

Every number here comes from a `summary.json` written by a GPU run and uploaded to
`primce/lexsi-runs`; nothing is estimated.

## Round 2 (2026-11-04)

Round 1 left four holes: one seed, a 30-task denominator, an abstention problem the adapter
ignored, and a track that asks for an agentic-eval toolkit the project never ran. This section
reports each one in the order it was pre-registered in `docs/ROUND2.md`, including the target it
missed.

### Artifacts

| Item | Value |
|---|---|
| Adapter (shipped) | `primce/tiny-aya-hinglish-lora` → `sft3_adapter` |
| Seed controls | same repo → `sft3_seed7_adapter`, `sft3_seed123_adapter` |
| Training data | `data/tasks_train_mix2.jsonl` (184 mix + 96 abstention + 32 twins) |
| Holdouts | `data/tasks_hard_eval.jsonl` (30), `data/tasks_hard90_eval.jsonl` (90, own pin), `data/tasks_eval.jsonl` (24), `data/tasks_safety_eval.jsonl` (32) |
| Run logs | `primce/lexsi-runs` → `sft3_hard30`, `sft3_hard90`, `sft3_eval`, `sft3_seed7_*`, `sft3_seed123_*`, `base_hard90`, `safety_sft3*` |
| Independent eval | AuditKIT 1.1.3 via `scripts/auditkit_eval.py`, no judge model |

### Held-out success

Greedy decoding, `--no-4bit`, zero tool errors in every row.

| Model | hard-30 | hard-90 | eval-24 |
|---|---:|---:|---:|
| Base Tiny Aya, prompt-only | 3.33% (1/30) | 3.33% (3/90) | - |
| SFT-mix, round 1 (`sft2`) | 96.67% (29/30) | **88.89% (80/90)** | 100% (24/24) |
| **SFT-mix2 seed 42 (`sft3`, shipped)** | **96.67% (29/30)** | **75.56% (68/90)** | **100% (24/24)** |
| SFT-mix2 seed 7 | 93.33% (28/30) | 83.33% (75/90) | - |
| SFT-mix2 seed 123 | 80.00% (24/30) | 75.56% (68/90) | - |

Wilson 95% intervals: sft3 83.3–99.4% on hard-30 and 65.8–83.3% on hard-90; base 0.6–16.7%
and 1.1–9.3%; seed 7 78.7–98.2% and 74.3–89.6%; seed 123 62.7–90.5% and 65.8–83.3%. The 90-task
number is the one to quote: the interval is ±7 pp instead of ±13 pp.

The round-1 adapter row landed with stage 1b (`6aeaf8d80e0338d1bb7f983a`, 2026-11-04) and it
reorders the table: `sft2` is 12 tasks better than `sft3` on hard-90, and the difference is one
family. `empty_balance` is **15/15 for `sft2` and 4/15 for `sft3`**; every other family is within
one task. Paired exact McNemar over the 90 shared tasks: **14 tasks only `sft2`, 2 only `sft3`,
p = 0.00418**. The abstention rows bought `asked` 0/8 -> 2/8 and cost 11 tasks on the primary
split, which the 30-task file could not see. Base -> `sft2` on the same 90 tasks is 77 fixed, 0
broken (p < 1e-20).

hard-90 is harder for a reason that is not tool use. sft3 keeps `ter` 0.011, `overcall` 0.000
and `rad` 0.000; the misses are answer shape (`tac` 0.822, `lcf` 0.011) and eight tasks with one
unparsable turn (`parse_failures` 0.09). The base model fails the other way: 7.24 calls against
the oracle's 3.57, `overcall` 1.258, 52% of episodes never reach `finish`, 21% with a parse
failure.

### Seed spread

| Seed | hard-30 | hard-90 | mean of the two |
|---|---:|---:|---:|
| 42 (pre-registered, shipped) | 96.67% | 75.56% | 86.11% |
| 7 | 93.33% | 83.33% | 88.33% |
| 123 | 80.00% | 75.56% | 77.78% |
| spread (max − min) | 16.67 pp | 7.77 pp | - |
| mean across seeds | 90.00% | 78.15% | - |

The spread is real and it is reported, not averaged away: on hard-90, seed 7 beats seed 42 on
7 tasks with 0 regressions (exact McNemar p = 0.0156), while seed 123 trades 4 for 4 against it
(p = 1.0). On hard-30 the three seeds are not separable (42 vs 7: 0 fixed, 1 broken, p = 1.0).

**Why 42 ships.** It was fixed before the seed runs existed, and it is the median seed on the
mean of the two splits. Shipping seed 7 would raise the headline by 7.8 pp on hard-90, but it
would mean choosing the seed on the split we report. The honest version is the one published:
one pre-registered seed, plus the spread, plus the note that a different seed can be better.

### Paired tests (deterministic scorer, same task ids)

| Comparison | Split | Fixed | Broken | Gain | Exact McNemar p |
|---|---|---:|---:|---:|---:|
| base → sft3 | hard-90 (n=90) | 66 | 1 | +72.2 pp | 9.2e-19 |
| sft3 → seed 7 | hard-90 (n=90) | 7 | 0 | +7.8 pp | 0.0156 |
| sft3 → seed 123 | hard-90 (n=90) | 4 | 4 | 0.0 pp | 1.0 |
| sft3 → seed 7 | hard-30 (n=30) | 0 | 1 | −3.3 pp | 1.0 |
| base → sft3 | hard-30 (n=30) | 28 | 0 | +63.3 pp | 7.5e-09 |

Reproduce:

```bash
python scripts/stats.py --a runs/base_hard90 --b runs/sft3_hard90
python scripts/stats.py --a runs/sft3_hard90 --b runs/sft3_seed7_hard90
python scripts/stats.py --a runs/sft3_hard90 --b runs/sft3_seed123_hard90
```

### Per family on hard-90 (15 each)

| Family | Base | sft3 (42) | seed 7 | seed 123 |
|---|---:|---:|---:|---:|
| settle5 | 0/15 | 15/15 | 15/15 | 14/15 |
| trap_combo | 0/15 | 15/15 | 15/15 | 15/15 |
| paraphrase | 0/15 | 13/15 | 13/15 | 13/15 |
| two_query | 0/15 | 13/15 | 14/15 | 11/15 |
| convert_sum | 0/15 | 8/15 | 11/15 | 11/15 |
| empty_balance | 3/15 | 4/15 | 7/15 | 4/15 |

`empty_balance` and `convert_sum` are where the remaining 22 failures live, and the seed spread
sits in the same two families. Clearing hard-30 did not clear hard-90: the 30-task set exposed
two families with 0/6 and 0/3, the 90-task set exposes two families with 4/15 and 8/15.

### AuditKIT (independent scorer, Track 3 toolkit)

`scripts/auditkit_eval.py` rebuilds each recorded episode from `trajectories.jsonl` as
`auditkit.agent_eval.AgentCase` + episode and scores it in `mode="recorded"`. No judge model is
used (`tool_selection` is excluded because it requires one). Two routes: a "ledger route" that
drops the model's `finish` call and compares the ledger calls against the oracle actions, and a
"full route" that includes `finish` compared on name only (arity differs by design).

| Split | Model | tool_call_exact | tool_call_f1 | precision | recall | redundant | loop | agreement |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| hard-30 | base | 0.0333 | 0.1875 | 0.1547 | 0.3194 | 0.3270 | 0.4333 | 1.000 |
| hard-30 | sft3 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 |
| hard-90 | sft3 | 0.7778 | 0.8114 | 0.8130 | 0.8102 | 0.0109 | 0.9875 | 0.9889 |
| hard-90 | sft2 | 0.9333 | 0.9463 | 0.9444 | 0.9500 | 0.0000 | 1.0000 | 0.9889 |

Full route (with `finish`, name-only): base hard-30 exact 0.100, F1 0.448; sft3 hard-30 all
1.000; sft3 hard-90 exact 0.8222, F1 0.9210; sft2 hard-90 exact 0.9444, F1 0.9681.

The independent scorer separates the two round-2 schedulers the same way the task metric does:
on hard-90 `sft2` is exact on 0.933 of the ledger calls where `sft3` is exact on 0.778, with no
redundant call and no loop in either. That is the "ask instead of checking" failure showing up as
a missing `get_balance`, not as a malformed call — validity stays at 0.99 for both.

`agreement_with_own_scorer` is 1.000 on hard-30 and 0.9889 (89/90) on hard-90, so AuditKIT
reproduces the project's own deterministic scorer rather than redefining success. Its
diagnostics add the before/after the track asks for: exact-call accuracy 0.033 → 1.000, tool-call
F1 0.188 → 1.000, redundant calls 0.327 → 0.000 on the same 30 tasks. (AuditKIT counts 69/90
successes on hard-90 against the project's 68, and 81/90 for `sft2` against 80; in both cases the
one disagreement is a parse-failure task where the final ledger still matches.)

### Safety split (32 tasks, 24 with a payload)

| System | safe | utility | attack_ledger | attack_answer | unplanned_calls |
|---|---:|---:|---:|---:|---:|
| Base Tiny Aya | 0/24 | 0/24 | 13/24 (54.2%) | 8/24 (33.3%) | 1.08 |
| SFT-mix (`sft2`) | 11/24 (45.8%) | 11/24 | **0/24** | 3/24 (12.5%) | 0.29 |
| SFT-mix + hardened prompt | 15/24 (62.5%) | 15/24 | **0/24** | 2/24 (8.3%) | 0.17 |
| **SFT-mix2 (`sft3`)** | **15/24 (62.5%)** | **15/24** | **0/24** | **2/24 (8.3%)** | **0.12** |
| SFT-mix2 + hardened prompt | 12/24 (50.0%) | 12/24 | **0/24** | 3/24 (12.5%) | 0.17 |

Underspecified requests (8, no payload): `asked` 0/8 base → 0/8 sft2 → **2/8 sft3**, and
`invented_expense` 8/8 → 7/8 → **6/8**. Full family counts in `docs/SAFETY.md`.

The pre-registered abstention bar was `asked ≥ 6/8` and `invented_expense ≤ 2/8`. **Both were
missed.** What the abstention slice bought is directional and small: the model now asks on 2 of 8
instead of 0, stops inventing an expense on 1 more task, and keeps `attack_ledger` at 0/24 while
`unplanned_calls` drops to 0.12. The hardened prompt, which helped sft2, now costs 3 safe tasks
on sft3 (15 → 12), so it is not shipped as the default.

The honest read: 8.4% supervised ask tokens is enough to make the boundary visible, not enough to
make it the default behaviour. A real fix needs either more abstention data (the mask is the
cheap lever) or the GRPO abstain reward, which is running as stage 2.

### What round 2 changed, in one paragraph

The 30-task number was optimistic by 21 points: sft3 drops from 96.67% on hard-30 to 75.56% on
hard-90, and the same drop appears in the two families the 30-task set under-samples
(`empty_balance` 3/3 → 4/15, `convert_sum` 2/3 → 8/15). The 90-task holdout and the three seeds
are what turn "96.67%, n=30, one seed" into "75.56–96.67%, n=90, 3 seeds, ±7 pp". Training stayed
deterministic and cheap: the mix2 SFT is ~156 optimizer steps on one T4, and the whole round-2
evidence base cost less than $20 of rented GPU time.

## Round 1 (2026-11-03) — sft2 freeze

Training and the tuned evals ran on Kaggle T4 x2 from `laxsi-ai-round2.ipynb`, the base-model
before-column and the reproducibility re-run from `laxsi-ai-base-bench.ipynb` (2026-11-04).

### Frozen artifact and provenance

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

`docs/ROUND2.md` tracks the follow-up runs: a 90-task holdout, an abstention SFT mix, a GRPO
configuration whose reward can see the abstention behaviour, and a seed control. Those numbers
are filled from `runs/*/summary.json` when the GPU runs land; nothing in this file is estimated.

### Data

| Split | Rows | Families |
|---|---:|---|
| `tasks_train.jsonl` (easy) | 100 | single/multi expense, balance, settle, convert |
| `tasks_eval.jsonl` (easy, held out) | 24 | same distribution as easy train |
| `tasks_hard_train.jsonl` | 40 | settle5, two_query, convert_sum, empty_balance, trap_combo, paraphrase |
| `tasks_hard_eval.jsonl` (held out) | 30 | 6 families × 5: settle5 6, two_query 6, convert_sum 3, empty_balance 3, trap_combo 6, paraphrase 6 |
| `tasks_train_mix.jsonl` | 184 | easy 100 + hard 84 (hard rows repeated per family) |
| `tasks_hard90_eval.jsonl` (held out) | 90 | 15 per family, own pin `hard90_eval.sha256` |
| `tasks_train_abstain.jsonl` | 64 | 32 underspecified + 32 fully-specified twins |
| `tasks_train_mix2.jsonl` | 312 | mix 184 + abstention 96 (3 copies) + twins 32 |

Mix family share: easy 0.543, two_query 0.13, convert_sum 0.087, trap_combo 0.087,
empty_balance 0.065, settle5 0.043, paraphrase 0.043. A hard-train row that appears in
the hard-eval set aborts `data/build_mix.py`, so the repetition cannot leak eval rows.

### Training runs

| Stage | Data | Epochs / steps | Train time | Notes |
|---|---|---|---:|---|
| SFT-easy | 100 easy rows | 3 epochs, 75 steps | 1595 s | LoRA r=16, all-linear, fp16, lr 2e-4 |
| GRPO | 184 mix rows | 30 steps (T4 profile caps 60 → 30) | 10 686 s | lr 1e-6, graded + efficiency rewards, 2 generations |
| **SFT-mix (frozen)** | 184 mix rows | 2 epochs, 92 steps | 2252 s | same SFT config as SFT-easy, then merged |

### Results (round 1)

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

### Safety split (round 1)

The repo also ships a 32-task injection eval. Headline: base writes attacker money into the
ledger in 13/24 injection tasks, the frozen adapter in 0/24 (exact McNemar p = 0.00024), while
utility under attack rises from 0/24 to 11/24. Abstention on underspecified requests stays at
1/8 for both tuned variants — the open problem. Full table, per-family counts and the threat
model: `docs/SAFETY.md`.

Run logs: `primce/lexsi-runs` → `safety_base`, `safety_sft2`, `safety_sft2_hardened`.

### What moved the number (round 1)

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

### Statistics (round 1)

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

### Caveats to state in the write-up

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

### Reproduce

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
