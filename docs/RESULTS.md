# Results — sft4 (shipped)

**Frozen on 2026-11-05.** `sft4` is the submission adapter: the round-1 SFT mix plus an abstention
slice and minimal pairs that separate "the ledger is empty" from "the request is missing a
number" (`data/tasks_train_mix3.jsonl`, 374 rows), seed 42. It is measured on the 90-task
holdout, the 30-task holdout, the easy split, the safety split, and an independent AuditKIT pass.

**Why it ships.** `sft4` is 84/90 on hard-90 (93.33%, 95% CI 86.2–96.9%) against 80/90 for the
round-1 adapter and 68/90 for `sft3`; it restores `empty_balance` to 15/15 while keeping `asked`
at 4/8, and it is the only adapter that writes zero attacker values into an answer on the
injection split (0/24). The `sft5` ablation scored one task higher on hard-30 and one lower on
hard-90 but collapsed abstention back to 0/8, so it does not ship. Every number below comes from
a `summary.json` written by a GPU run and uploaded to `primce/lexsi-runs`; nothing is estimated.

## Shipped artifacts

| Item | Value |
|---|---|
| Adapter (shipped) | `primce/tiny-aya-hinglish-lora` → `sft4_adapter` |
| Seed controls | same repo → `sft3_seed7_adapter`, `sft3_seed123_adapter` (mix2 recipe, seeds 7 and 123) |
| Training data | `data/tasks_train_mix3.jsonl` (100 easy + 84 hard + 96 abstention + 94 contrast) |
| Holdouts | `data/tasks_hard_eval.jsonl` (30), `data/tasks_hard90_eval.jsonl` (90, own pin), `data/tasks_eval.jsonl` (24), `data/tasks_safety_eval.jsonl` (32) |
| Run logs | `primce/lexsi-runs` → `sft4_hard30`, `sft4_hard90`, `sft4_eval`, `sft3_seed7_*`, `sft3_seed123_*`, `grpo3_*`, `safety_sft4*`, `auditkit_sft4_hard90` |
| Independent eval | AuditKIT 1.1.3 via `scripts/auditkit_eval.py`, no judge model |

## Round 3 (2026-11-05) — sft4

Round 2 ended with a contradiction worth stating plainly: the abstention slice that bought
`asked` 0/8 → 2/8 cost 12 tasks on hard-90, and the losses were all `empty_balance` (4/15 against
15/15 for the adapter without abstention rows). The hypothesis was that 96 identical ask rows
taught the *wording* of an abstention answer rather than the *condition* for asking, and that the
model could not tell "empty ledger, so ask" from "empty ledger, so report zero" without a
minimal pair showing both. Round 3 pre-registered one bar: **`sft4` ≥ 80/90 on hard-90 with
`asked` ≥ 2/8**. It measured 84/90 and 4/8.

### Held-out success

Greedy decoding, `--no-4bit`, zero tool errors in every tuned row.

| Model | hard-30 | hard-90 | eval-24 |
|---|---:|---:|---:|
| Base Tiny Aya, prompt-only | 3.33% (1/30) | 3.33% (3/90) | - |
| SFT-mix, round 1 (`sft2`) | 96.67% (29/30) | 88.89% (80/90) | 100% (24/24) |
| SFT-mix2, round 2 (`sft3`) | 96.67% (29/30) | 75.56% (68/90) | 100% (24/24) |
| **SFT-mix3 seed 42 (`sft4`, shipped)** | **96.67% (29/30)** | **93.33% (84/90)** | **100% (24/24)** |
| SFT-mix5 seed 42 (`sft5`, ablation) | 100% (30/30) | 92.22% (83/90) | 100% (24/24) |
| SFT-mix2 seed 7 | 93.33% (28/30) | 83.33% (75/90) | - |
| SFT-mix2 seed 123 | 80.00% (24/30) | 75.56% (68/90) | - |

Wilson 95% intervals: `sft4` 83.3–99.4% on hard-30 and **86.2–96.9%** on hard-90; `sft5`
88.6–100% and 84.8–96.2%; base 0.6–16.7% and 1.1–9.3%. The 90-task number is the one to quote:
the interval is ±5.4 pp on the shipped adapter instead of ±13 pp on the 30-task set.

`convert_sum` is the family that moved: 7/15 (`sft2`) → 8/15 (`sft3`) → **13/15** (`sft4`), with
`empty_balance` back to **15/15** from 4/15. The cost is two `two_query` tasks (14/15 → 12/15).
`sft4` keeps `tac` 0.967, `ter` 0.000, `overcall` 0.000, `rad` 0.000, `lcf` 0.000 and
`parse_failures` 0.000 — all nine misses are answer-shape misses, not tool misses.

### Per family on hard-90 (15 each)

| Family | Base | `sft2` | `sft3` | **`sft4`** | `sft5` |
|---|---:|---:|---:|---:|---:|
| settle5 | 0/15 | 15/15 | 15/15 | 15/15 | 15/15 |
| trap_combo | 0/15 | 15/15 | 15/15 | 15/15 | 15/15 |
| paraphrase | 0/15 | 14/15 | 13/15 | 14/15 | 14/15 |
| two_query | 0/15 | 14/15 | 13/15 | 12/15 | 13/15 |
| convert_sum | 0/15 | 7/15 | 8/15 | **13/15** | 11/15 |
| empty_balance | 3/15 | 15/15 | 4/15 | **15/15** | 15/15 |

### Paired tests (deterministic scorer, same task ids)

| Comparison | Split | Fixed | Broken | Gain | Exact McNemar p |
|---|---|---:|---:|---:|---:|
| base → `sft4` | hard-90 (n=90) | 81 | 0 | +90.0 pp | < 1e-20 |
| `sft2` → `sft3` | hard-90 (n=90) | 2 | 14 | −13.3 pp | 0.00418 |
| `sft2` → `sft4` | hard-90 (n=90) | 7 | 3 | +4.4 pp | 0.34375 |
| `sft3` → `sft4` | hard-90 (n=90) | 18 | 2 | +17.8 pp | 0.0004 |
| `sft4` → `sft5` | hard-90 (n=90) | 1 | 2 | −1.1 pp | 1.0 |
| `sft3` → `grpo3` | hard-90 (n=90) | 0 | 0 | 0.0 pp | 1.0 |

The round-2 → round-3 fix is unambiguous (18 / 2, p = 0.0004). The round-1 → round-3 difference
is not: 7 tasks fixed against 3 broken gives p = 0.34, so `sft4` should be described as *at least
as good as* the round-1 adapter, with a directional +4-task edge, not as a significant win. That
is stated here and in the article rather than rounded up into a claim.

```bash
python scripts/stats.py --a runs/sft2_hard90 --b runs/sft4_hard90
python scripts/stats.py --a runs/sft3_hard90 --b runs/sft4_hard90
python scripts/stats.py --a runs/sft4_hard90 --b runs/sft5_hard90
```

### Wording split: did the fix generalize or memorize?

`scripts/template_split.py` masks person names and matches each hard-90 task against the
training file, separating tasks whose wording was seen from tasks whose wording was not.

| Model | seen wording | unseen wording | total |
|---|---:|---:|---:|
| `sft2` | 10/10 | 70/80 (87.50%) | 80/90 |
| `sft3` | 5/10 (50.00%) | 63/80 (78.75%) | 68/90 |
| **`sft4`** | **10/10** | **74/80 (92.50%)** | **84/90** |
| `sft5` | 10/10 | 73/80 (91.25%) | 83/90 |

`sft3` was the only adapter that failed *seen* wording — 4/9 on `empty_balance` and 0/6 on the
unseen six — which is what "learned the ask answer, not the condition" looks like numerically.
`sft4` is 9/9 and 6/6 on the same cells, so the contrast pairs fixed the condition, and its
92.50% on unseen wording is higher than `sft2`'s 87.50%. Reproduce:

```bash
python scripts/template_split.py --run sft4_hard90 --train data/tasks_train_mix3.jsonl
```

Caveat: defining `tasks_train_mix3.jsonl` as "seen" is the strict version, because the mix
contains paraphrases of evaluation templates. One hard-90-only phrasing also appears in the
contrast set. The split is a diagnostic, not a clean generalization measurement.

### Seed spread

The round-2 seed controls (same mix2 recipe, three seeds) put the recipe's own variance at
16.67 pp on hard-30 and 7.77 pp on hard-90:

| Seed | hard-30 | hard-90 | mean of the two |
|---|---:|---:|---:|
| 42 (pre-registered) | 96.67% | 75.56% | 86.11% |
| 7 | 93.33% | 83.33% | 88.33% |
| 123 | 80.00% | 75.56% | 77.78% |
| spread (max − min) | 16.67 pp | 7.77 pp | - |

The three seeds above are the mix2 recipe. The mix3 recipe that ships was trained once: a seed-7
re-train finished on Kaggle but its held-out evaluation did not complete before the GPU quota ran
out, and both free-compute budgets (Kaggle quota, HF Jobs credit) are now exhausted, so no mix3
seed spread exists and none is claimed. Shipping seed 42 stays the rule: it was fixed before any
seed run existed, and choosing the best seed on the split that gets reported would be selection.

### AuditKIT (independent scorer, Track 3 toolkit)

`scripts/auditkit_eval.py` rebuilds each recorded episode from `trajectories.jsonl` as
`auditkit.agent_eval.AgentCase` + episode and scores it in `mode="recorded"`. No judge model is
used (`tool_selection` is excluded because it requires one). Two routes: a ledger route that
drops the model's `finish` call and compares ledger calls against the oracle actions, and a full
route that includes `finish` compared on name only.

| Run | n | AuditKIT success | agreement | ledger tool_f1 | validity | full F1_name | loop |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base — hard-30 | 30 | 3.33% (1/30) | 1.0000 | 0.1875 | 0.9833 | 0.4476 | 0.4333 |
| `sft2` — hard-30 | 30 | 96.67% (29/30) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `sft2` — hard-90 | 90 | 90.00% (81/90) | 0.9889 | 0.9463 | 0.9944 | 0.9681 | 1.0000 |
| `sft3` — hard-90 | 90 | 76.67% (69/90) | 0.9889 | 0.8114 | 0.9875 | 0.9210 | 0.9875 |
| **`sft4` — hard-90** | 90 | **95.56% (86/90)** | **0.9778** | **0.9804** | **1.0000** | **0.9963** | **1.0000** |

`agreement_with_own_scorer` is the number that matters: two independent scorers on the same
recorded episodes agree on 88 of 90 decided tasks (0.9778). AuditKIT counts 86 successes against
the project's 84; the two disagreements are the known parse-failure pair, not a definition of
success that drifts. Its diagnostics reproduce the narrative: exact ledger calls 0.778 → 0.980,
`tool_call_validity` 1.000, zero redundant calls, zero loops.

### Safety split (32 tasks, 24 with a payload)

| System | safe | utility | attack_ledger | attack_answer | unplanned_calls |
|---|---:|---:|---:|---:|---:|
| Base Tiny Aya | 0/24 | 0/24 | 13/24 (54.2%) | 8/24 (33.3%) | 1.08 |
| SFT-mix (`sft2`) | 11/24 (45.8%) | 11/24 | **0/24** | 3/24 (12.5%) | 0.29 |
| SFT-mix2 (`sft3`) | 15/24 (62.5%) | 15/24 | **0/24** | 2/24 (8.3%) | 0.12 |
| **SFT-mix3 (`sft4`)** | **14/24 (58.3%)** | **14/24** | **0/24** | **0/24** | 0.25 |
| SFT-mix3 + hardened prompt | 14/24 (58.3%) | 14/24 | **0/24** | 2/24 (8.3%) | 0.21 |
| SFT-mix5 (`sft5`) | 11/24 (45.8%) | 11/24 | **0/24** | 3/24 (12.5%) | 0.50 |

Underspecified requests (8, no payload) — the abstention bar was `asked ≥ 6/8` and
`invented_expense ≤ 2/8`, pre-registered in `docs/ROUND2.md`:

| System | asked | invented_expense |
|---|---:|---:|
| Base | 0/8 | 8/8 |
| `sft2` | 0/8 | 7/8 |
| `sft3` | 2/8 | 6/8 |
| **`sft4`** | **4/8** | **4/8** |
| **`sft4` + hardened prompt** | **7/8** | **0/8** |
| `sft5` | 0/8 | 8/8 |

`sft4` alone misses the bar on both halves (4/8 and 4/8). `sft4` with the hardened system prompt
clears it (7/8 and 0/8) and also raises overall safety from 18/32 to 22/32, at the cost of 2/24
`attack_answer` against 0/24 without it. The honest configuration statement is therefore: the
adapter ships as the default, and the hardened prompt is documented as the abstention
configuration with its measured trade-off, because a hard-90 run of the hardened prompt has not
been done and the task-success cost of switching the default is unmeasured. Full family counts:
`docs/SAFETY.md`.

### What round 3 changed

Twelve tasks on hard-90 were the price of teaching "ask". Round 3 identified that the model had
learned the *surface* of an abstention (a verbatim answer) and not its *trigger*, fixed it with
contrast pairs that put an empty ledger and a missing number side by side, and recovered the
twelve plus four more (84/90) while keeping the ask behaviour. The remaining nine misses are all
answer shape in `convert_sum` (13/15) and `two_query` (12/15), with zero tool errors, zero
redundant calls and zero loops — so the next lever is the answer format, not the tool policy.

---

## Round 2 (2026-11-04) — sft3, kept for provenance

Round 1 left four holes: one seed, a 30-task denominator, an abstention problem the adapter
ignored, and a track that asks for an agentic-eval toolkit the project never ran. Round 2
reported each in the order pre-registered in `docs/ROUND2.md`. The headline result was negative:
`sft3` (the abstention mix) dropped 12 tasks on hard-90 against `sft2`, and the loss was one
family, `empty_balance` (4/15 against 15/15). This section is unchanged from the 2026-11-04
freeze; round 3 above explains the failure and fixes it.

### Round-2 held-out numbers

| Model | hard-30 | hard-90 | eval-24 |
|---|---:|---:|---:|
| Base Tiny Aya, prompt-only | 3.33% (1/30) | 3.33% (3/90) | - |
| SFT-mix, round 1 (`sft2`) | 96.67% (29/30) | **88.89% (80/90)** | 100% (24/24) |
| SFT-mix2 seed 42 (`sft3`) | **96.67% (29/30)** | **75.56% (68/90)** | **100% (24/24)** |
| SFT-mix2 seed 7 | 93.33% (28/30) | 83.33% (75/90) | - |
| SFT-mix2 seed 123 | 80.00% (24/30) | 75.56% (68/90) | - |

Wilson 95% intervals: `sft3` 83.3–99.4% on hard-30 and 65.8–83.3% on hard-90; base 0.6–16.7%
and 1.1–9.3%; seed 7 78.7–98.2% and 74.3–89.6%; seed 123 62.7–90.5% and 65.8–83.3%.

Paired exact McNemar on the same 90 tasks: **14 tasks only `sft2`, 2 only `sft3`,
p = 0.00418**. The abstention rows bought `asked` 0/8 → 2/8 and cost 11 tasks on the primary
split, which the 30-task file could not see. Base → `sft2` on the same 90 tasks is 77 fixed, 0
broken (p < 1e-20).

`sft3` keeps `ter` 0.011, `overcall` 0.000 and `rad` 0.000; the misses are answer shape
(`tac` 0.822, `lcf` 0.011) and eight tasks with one unparsable turn (`parse_failures` 0.09). The
base model fails the other way: 7.24 calls against the oracle's 3.57, `overcall` 1.258, 52% of
episodes never reach `finish`, 21% with a parse failure.

### Round-2 seed spread

| Seed | hard-30 | hard-90 | mean of the two |
|---|---:|---:|---:|
| 42 (pre-registered, shipped at the time) | 96.67% | 75.56% | 86.11% |
| 7 | 93.33% | 83.33% | 88.33% |
| 123 | 80.00% | 75.56% | 77.78% |
| spread (max − min) | 16.67 pp | 7.77 pp | - |
| mean across seeds | 90.00% | 78.15% | - |

The spread is real and it is reported, not averaged away: on hard-90, seed 7 beats seed 42 on
7 tasks with 0 regressions (exact McNemar p = 0.0156), while seed 123 trades 4 for 4 against it
(p = 1.0). On hard-30 the three seeds are not separable (42 vs 7: 0 fixed, 1 broken, p = 1.0).
That spread sets the scale for reading every later difference in this file: a 4-task gap on
hard-90 is smaller than the recipe's own seed noise.

```bash
python scripts/stats.py --a runs/base_hard90 --b runs/sft3_hard90
python scripts/stats.py --a runs/sft3_hard90 --b runs/sft3_seed7_hard90
python scripts/stats.py --a runs/sft3_hard90 --b runs/sft3_seed123_hard90
```

### Round-2 per family on hard-90

| Family | Base | `sft3` (42) | seed 7 | seed 123 |
|---|---:|---:|---:|---:|
| settle5 | 0/15 | 15/15 | 15/15 | 14/15 |
| trap_combo | 0/15 | 15/15 | 15/15 | 15/15 |
| paraphrase | 0/15 | 13/15 | 13/15 | 13/15 |
| two_query | 0/15 | 13/15 | 14/15 | 11/15 |
| convert_sum | 0/15 | 8/15 | 11/15 | 11/15 |
| empty_balance | 3/15 | 4/15 | 7/15 | 4/15 |

### Round-2 safety

Underspecified requests: `asked` 0/8 base → 0/8 `sft2` → **2/8 `sft3`**, `invented_expense`
8/8 → 7/8 → **6/8**. The pre-registered bar (`asked ≥ 6/8`, `invented ≤ 2/8`) was missed. What
the abstention slice bought was directional and small, and it cost 12 tasks on hard-90 — the
motivation for round 3.

### What round 2 changed, in one paragraph

The 30-task number was optimistic by 21 points: `sft3` drops from 96.67% on hard-30 to 75.56% on
hard-90, and the drop appears in the two families the 30-task set under-samples (`empty_balance`
3/3 → 4/15, `convert_sum` 2/3 → 8/15). The 90-task holdout and the three seeds are what turn
"96.67%, n=30, one seed" into "75.56–96.67%, n=90, 3 seeds, ±7 pp". Training stayed deterministic
and cheap: mix2 SFT is ~156 optimizer steps on one T4, and the whole round-2 evidence base cost
less than $20 of rented GPU time.


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
