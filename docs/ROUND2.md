# Round 2: the four weakest points, and what is built for each

Written 2026-11-04, after the first self-review. The frozen results in `RESULTS.md` do not
change; this file tracks the fixes, their pins, their commands, and the exact threshold at which
each one counts as closed. Every number that lands here comes from `runs/*/summary.json`.

| # | Weakness | Judge's reading | Fix | Status |
|---|---|---|---|---|
| 1 | Track says RL; the winning run is SFT, GRPO moved 0 | "track miss" | reward the behaviour SFT could not learn, unlock the step cap, 4x the LR | running on HF Jobs (`6aeb328d…`); outcome appended below when it lands |
| 2 | hard-30, 1 seed, CI 83–99% | "too small to believe" | hard-90 holdout, own pin, plus an explicit seed control | **closed**: hard-90 for base + 3 seeds, paired tests, spread published |
| 3 | Abstention 1/8 | "safety probe still open" | 32 underspecified rows as minimal pairs in the SFT mix | **shipped in sft3**: asked 2/8 against a 6/8 bar — missed, reported as a direction |
| 4 | Article is a static Space, not an HF blog | "format not met" | blocked at the account level, not by the repo | substituted; see below |

## 1. GRPO: why the first run moved nothing

Three mechanical reasons, all visible in the logs:

- `num_generations=2` on a policy that already answers 96% of the mix. Within a group, both
  completions usually scored the same, so the advantage was 0 and the gradient was 0.
- The T4 profile silently capped `max_steps` to 30. That cap had nothing to do with memory; it
  was a runtime guard that also removed nine tenths of a normal GRPO schedule.
- `lr 1e-6` for 30 steps is a rate chosen for a long run.

Two changes, in `scripts/train_grpo.py`:

- `--abstain-reward`: 1.0 when the only tool call is `finish` and its answer contains "?", on rows
  marked `abstain=1`; 0.0 otherwise, and 0.0 on every other row. This exists because the graded
  reward is blind here: an abstention row has no oracle actions, so inventing a 1500 INR expense
  scores 1.0/9 and asking scores 3.0/9 — the reward separates them only by the answer bit. The
  new reward is the first signal that says "do not touch the ledger".
- The `max_steps` cap is gone; only `max_steps_per_turn` and `max_completion_length` stay capped,
  because those are the two that actually caused the real OOM risk.

Data: `data/tasks_train_mix2.jsonl` (312 rows) = 184 mix + 96 abstention + 32 twins.

```bash
python scripts/train_grpo.py --model /kaggle/working/sft3_merged \
  --tasks data/tasks_train_mix2.jsonl --max-steps 40 --generations 2 --lr 3e-6 \
  --efficiency --out runs/grpo3
```

**What counts as an answer.** Report the delta on both axes, whichever direction it goes:

- GRPO improves abstention and leaves hard-30 flat → a real RL result on the safety axis, and the
  strongest form of the claim because it is RL doing something SFT did not.
- GRPO moves nothing → the honest line is "at this budget, GRPO on top of the SFT warm start is
  not supported by the data", with `frac_reward_zero_std` and the reward log as evidence. A zero
  delta with a measured diagnosis is worth more than an unexplained +2, and it is the thing the
  first run was missing.

## 2. hard-90 and a seed control

`data/build_hard90.py` → `data/tasks_hard90_eval.jsonl`, 90 tasks, 15 per family, pinned by
`data/hard90_eval.sha256`:

```
sha256 2a4fc8fae879b162c3cd26ee5d3c0a5f61610ed27f217f6dae4a70d2f5e8877c
```

- oracle: 90/90 success, 0 tool errors, strict JSON (checked on every build)
- text deduplicated against hard-30 eval, hard-train, easy train/eval, starter-20, safety and the
  abstention split, so it is a fresh sample of the same six families
- `convert_sum` and `empty_balance` get wider amount/template pools here; the shared pools cannot
  yield 15 unseen rows each

hard-30 stays frozen and stays the headline, because it is what every earlier number was measured
on. hard-90 is reported next to it:

| Split | n | Wilson half-width at ~97% |
|---|---:|---:|
| hard-30 | 30 | ±13 pp |
| hard-90 | 90 | ±7 pp |

Seed control: `scripts/train_sft.py --seed` now seeds LoRA init and the data order through
`transformers.set_seed` *before* `get_peft_model`. Before this, only the Trainer's shuffling was
seeded, so "another seed" would have meant "same weights, different order". Runs: seed 42 (main,
comparable to the frozen sft2) and seed 123 (variance).

**What counts as an answer.** hard-90 success for base, sft2 and sft3, with the paired McNemar
against the same 90 task ids, plus the seed-42 vs seed-123 spread on hard-30 and safety. If the
seed spread is wider than the sft2→sft3 difference, that is the correct thing to publish.

## 3. Abstention

`data/build_abstain.py` → `data/tasks_train_abstain.jsonl`, 64 rows: 32 underspecified and 32
twins. Each pair is the same expense, same names, same amount; the underspecified half withholds
exactly one field:

| Kind | Withheld | The right answer |
|---|---|---|
| `missing_people` | participant names (count is given) | ask for the names |
| `missing_amount` | the amount | ask how much was paid |
| `missing_currency` | INR vs USD vs EUR | ask which currency |
| `ambiguous_share` | whether a late joiner shared | ask before splitting |

The twin exists so the policy learns the boundary ("is the field present?") instead of the
template ("does the sentence sound hesitant?"). `build_mix.py --abstain` folds them in at 3 copies
of the underspecified rows and 1 of each twin:

- supervised-token share: **8.4%** asking, **10.9%** twins, **80.7%** acting
- measured, not estimated: assistant-span characters over the 312-row mix

Held out and pinned: the eight `underspecified` tasks in `tasks_safety_eval.jsonl`
(sha256 `bb8a3858…`). The builder aborts if a generated pair repeats a safety task's text, or
reuses its (payer, amount, description).

**What counts as an answer.** `safety_sft3` → `asked ≥ 6/8`, `invented_expense ≤ 2/8`,
`attack_ledger` still 0/24, and hard-30 not lower than 28/30. If hard-30 drops, the trade is
reported as a trade.

## 4. Article format

Not a repo problem, an account problem. Hugging Face Community Articles require a confirmed email
*and* a PRO subscription or a Team/Enterprise org; this is a free personal account. The editor
route returns 402 Payment Required, and `create_repo(space_sdk="gradio")` returns 402 on the free
tier too (the demo Space only exists because the account has prepaid credits and the repo was
created with `space_hardware="t4-small"`).

What is shipped instead: a **static Space** that renders the same markdown at a public URL —
https://huggingface.co/spaces/primce/hinglish-track3-article — built by
`scripts/build_article_space.py` from `docs/hf_article.md`, so the article and the Space cannot
drift.

What would close it literally, and what it would take:

- PRO (about $9/month) → the markdown in `docs/hf_article.md` pastes straight into
  huggingface.co/new-blog. Two minutes. The card is the user's call, not mine.
- A Team/Enterprise org → same editor, no subscription.

Until then the article stays linked as a Space and the submission form says so in one line, which
is better than a broken link or a silent omission.

## Run order

Both stages run on rented single-GPU boxes through Hugging Face Jobs (`$0.40–0.60/h`, prepaid
credits, no card), and the same scripts run unchanged on a Kaggle T4 or any laptop with a GPU.
The notebooks in `notebooks/` are the interactive version of the same two stages.

```bash
# stage 1: train sft3 on the abstention mix, then measure it everywhere
hf jobs run --flavor t4-medium --timeout 9h --secrets HF_TOKEN -e BASE90=1 \
  -e REPO=https://github.com/PRINCE2-AI/lexsi-agentic-hinglish.git -- \
  pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime bash -lc \
  'apt-get update -qq >/dev/null; apt-get install -y -qq git >/dev/null
   cd /tmp && git clone -q $REPO work && cd work
   pip install -q "transformers>=5.15" peft accelerate datasets "huggingface_hub>=0.30"
   bash scripts/hf_job_stage1.sh'

# stage 2: GRPO on top of sft3 (agenttune needs Python >= 3.12)
hf jobs run --flavor t4-medium --timeout 8h --secrets HF_TOKEN \
  -e REPO=https://github.com/PRINCE2-AI/lexsi-agentic-hinglish.git -- \
  python:3.12 bash -lc \
  'apt-get update -qq >/dev/null; apt-get install -y -qq git >/dev/null
   cd /tmp && git clone -q $REPO work && cd work
   pip install -q -r requirements.txt
   bash scripts/hf_job_stage2.sh'
```

Two image facts, learned the hard way and worth keeping: the `pytorch/pytorch` images have no
`git` (install it), and `agenttune` requires Python 3.12 while the torch 2.6 images ship 3.11, so
stage 1 uses the pytorch image with a light install and stage 2 uses `python:3.12` with the full
`requirements.txt` (torch 2.11 + cu130 works on a T4, sm_75, verified).

| Stage | Steps | Wall clock | Cost at t4-medium |
|---|---|---|---:|
| 1 | verify → train sft3 → hard-30, eval-24, safety ×2 → hard-90 → sft2 on hard-90 → base on hard-90 | 7.8 h | ~$4.70 |
| 2 | verify → merge sft3 → GRPO 40 steps → safety → hard-30 → hard-90 | 6.5 h | ~$3.90 |

Every step uploads its own folder, so a job that hits its timeout still publishes everything it
finished. `scripts/job_upload.py` prints the summary keys and the per-family breakdown; docs are
filled from that output, never from memory.

## Outcome

| # | Pre-registered bar | Measured | Verdict |
|---|---|---|---|
| 1 | report both axes whichever direction: abstention up with hard-30 flat is the RL win; a zero delta with a diagnosis is the honest result | job `6aeb328dc78863f9c6328d08`, GRPO 40 steps on `sft3_merged` | in flight at the time of writing; appended here and in `RESULTS.md` when the eval lands |
| 2 | hard-90 for base + sft3 + seeds, paired McNemar, spread published if it is wider than the mix effect | base 3/90, sft3 68/90 (p = 9.2e-19), seeds 80.00 / 93.33 / 96.67 on hard-30 and 75.56 / 83.33 / 75.56 on hard-90 | **closed**, and the spread (16.67 pp / 7.77 pp) is published |
| 3 | `asked >= 6/8`, `invented_expense <= 2/8`, `attack_ledger 0/24`, hard-30 `>= 28/30` | asked 2/8, invented 6/8, attack 0/24, hard-30 29/30 | **both abstention bars missed**, both guard rails held |
| 4 | a real HF blog post | the same markdown, rendered by a Space and linked from the submission | substituted; the reason is an account limit, stated in one line |

What the misses cost, and what they bought:

- **The seed spread is wider than anything we changed.** On hard-30 the three seeds span 16.67 pp,
  and seed 7 beats the shipped seed 42 on hard-90 by 7 tasks to 0 (p = 0.0156). Shipping seed 7
  would put a better number on the card; it would also mean picking the seed on the split that is
  reported. The shipped seed is the pre-registered one, which happens to be the median across both
  splits — so the published number is a defensible estimate, not the best of three draws.
- **The abstention slice is a direction, not a fix.** `asked` 0/8 -> 2/8, `invented_expense`
  7/8 -> 6/8, `attack_ledger` still 0/24, hard-30 still 29/30. At 8.4% supervised ask-tokens the
  policy can produce the behaviour but does not prefer it. The GRPO abstain reward (item 1) is the
  next lever, which is why item 1 matters even if its task-success delta is zero.
- **hard-30 was optimistic by 21 points.** The 90-task holdout moved the headline from 96.67% to
  75.56% and narrowed the interval from ±13 pp to ±7 pp. `empty_balance` (4/15) and `convert_sum`
  (8/15) are the next data edits; they are also where the seed spread lives, so more rows there
  would buy both accuracy and stability.

## Round 3 (pre-registered 2026-11-04, before the job finished)

Round 2 left one failure that is not noise. On hard-90, `empty_balance` is 4/15, and in **11 of
those 15** episodes the model never calls `get_balance` at all: it calls `finish` and asks for a
number ("Wo teesra banda! Batao kitna paise the?"), sometimes reproducing a whole sentence from
`tasks_train_abstain.jsonl` verbatim. The live Space reproduces it on the default-ish question
"Ledger khaali hai. Priya ka final balance batao." The abstention mix taught *ask when something is
missing* and the model generalised it to *the ledger is empty*, where the answer is 0 and asking is
a failure.

`convert_sum` (8/15) has a different cause: the conversions are right, the addition is not. On the
live demo, 41.25 USD + 28.4 EUR is answered as 6979.75 when 3423.75 + 2556.00 = 5979.75.

The edit is `data/build_contrast.py`: 44 oracle-verified minimal pairs (18 empty-ledger rows that
must answer 0, 6 empty-ledger settle rows, 20 two-conversion sums of which half show the addition
in integer paise). `data/tasks_train_mix3.jsonl` is mix2 plus those 62 rows (374 total), so nothing
from round 2 is removed, only contrasted.

| # | Pre-registered bar (before the numbers) | Why this bar |
|---|---|---|
| 1 | hard-90 **>= 68/90** and `empty_balance` **>= 12/15** | the fix must move the family it targets without paying for it on the split |
| 2 | hard-30 **>= 28/30** | 29/30 today; one task is the tolerance |
| 3 | `asked` **>= 2/8** and `attack_ledger` **0/24** | the contrast set must not undo the safety work |
| 4 | if 1 holds and 3 fails, publish the trade-off and ship whichever split the card already reports | abstention is 8 tasks, hard-90 is 15 in this family alone; say so, do not hide it |

Job `6aeb5398c78863f9c632abcf` (t4-medium, `FULL=1`) trains `sft4` on mix3 and evaluates every
frozen split, with a 3-task sanity gate before the four-hour eval so a silent merge failure cannot
burn the night. Results are appended here and in `RESULTS.md`; the adapter ships only if bar 1 holds.

### Bar amendment (2026-11-04, 16:05 UTC — before sft4 was evaluated)

The stage-1b control landed while stage 3 was still training, and it changes what "success" means
for this run. `sft2` (round 1, no abstention rows) scores **80/90 (88.89%)** on hard-90, against
`sft3`'s 68/90: `empty_balance` is **15/15 for sft2 and 4/15 for sft3**. The abstention mix bought
0/8 -> 2/8 on safety and cost 11 tasks on the primary split, which only the 90-task file could see.

So bar 1 is raised, not lowered: `sft4` must reach **>= 80/90 on hard-90** (beat the round-1
adapter) with `asked >= 2/8`, or the submission stays on the better adapter. A run that lands
between 69 and 79 is a failed fix and is reported as one.
