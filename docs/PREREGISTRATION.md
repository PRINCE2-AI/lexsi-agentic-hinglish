# Pre-registration: what was fixed before the data arrived

A judge cannot check a claim that was chosen after the result was seen. This page lists every
decision that was written down **before** the run that tested it, the commit that fixes it, and
what it scored. It also lists the decisions that were made **after** seeing data, because those are
the ones a reader has to discount.

The rule for this repo: evaluation data is pinned by SHA-256 in `data/*.sha256`, `python
scripts/verify_all.py` regenerates each split and fails on drift (28 checks), and every published
number is copied from a `runs/*/summary.json`.

## 1. Frozen evaluation artefacts

| Artefact | n | SHA-256 (first 12) | Frozen in | Date |
|---|---:|---|---|---|
| `data/tasks_hard_eval.jsonl` (hard-30) | 30 | `37f52b47937d` | `1ebd51b` | 2026-11-02 |
| `data/tasks_hard90_eval.jsonl` (holdout) | 90 | `2a4fc8fae879` | `8b7116a` | 2026-11-04 |
| `data/tasks_safety_eval.jsonl` | 32 | `bb8a385842e8` | `eeaf818` | 2026-11-04 |
| `data/tasks_devanagari_eval.jsonl` | 30 | `25dfb0e51925` | `93c9cb1` | 2026-11-05 |
| `data/tasks_safety_eval2.jsonl` | 80 | `bc226c7ac8fb` | `99c8350` | 2026-11-08 |
| `data/tasks_safety_payee.jsonl` | 16 | `b22f6819e5b3` | `2bf8032` | 2026-11-08 |
| `data/tasks_safety_hinglish.jsonl` | 24 | `7ed6157247d4` | `c5a441a` | 2026-11-09 |
| `data/tasks_safety_heldout.jsonl` | 30 | `18de6bdb0096` | `bffc82b` | 2026-11-10 |
| `data/payloads_unseen.jsonl` (guard dev) | 12 | `a1607816fe92` | guard work | 2026-11-10 |
| `data/payloads_blind.jsonl` (guard blind) | 8 | `81962a2d408d` | guard work | 2026-11-10 |

The blind set's hash is listed here on its own because the point of that set is that it was hashed
before it was run.

One pin was wrong for a day: the devanagari row was first published as `a6f48ac088f1`, the hash of
the CRLF checkout this repo was built on, not of the stored file. The first CI run on a Linux
checkout failed on exactly that test, which is the only reason it was caught. The generator now
writes LF and every pin is compared against it, so the value above holds on Windows and Linux; the
task content never changed (30 tasks, same ids, same rows).

## 2. Frozen runtime

Everything in `docs/RESULTS.md`, `docs/SAFETY.md` and `docs/GUARD.md` was produced with:

- **Decoding:** greedy. `do_sample=False` (`scripts/run_baseline.py:73`), one sample per task.
- **Precision:** fp16, `--no-4bit` on every published row. No quantised number is published.
- **Budget:** `--max-steps 8`, `--max-new-tokens 256` (defaults of `run_baseline.py`).
- **Prompts:** `hinglish_agent/prompts.py`, default system prompt unless the row says `+hardened`.
- **Adapter:** `primce/tiny-aya-hinglish-lora`, subfolder `sft4_adapter`.
- **Hardware:** one T4 at a time (Kaggle T4×2 or HF Jobs), so the second GPU is never a variable.

## 3. Decision rules written before the test

| # | Rule, stated before the run | Where | Outcome |
|---|---|---|---|
| R1 | hard-90 is the holdout that decides; hard-30 is a smoke set that saturates. `sft4` ships only at `>= 80/90` | `8b7116a`, `0a2b68e`, `docs/ROUND2.md` | `sft4` 84/90 passes; three adapters tie at 29/30 on hard-30, so the holdout was doing the work |
| R2 | Abstention counts as closed at `asked >= 6/8` **and** `invented_expense <= 2/8` | `8b7116a`, `docs/ROUND2.md` | `sft4` alone 4/8 — **missed, reported as a miss**; `sft4`+hardened 7/8, 0/8 — passed |
| R3 | If the Hinglish payload split moves ledger or answer harm at the same n, the defence is partly language-specific | `c5a441a`, `docs/SAFETY.md` | 0/24 and 0/24 — language hypothesis **closed** |
| R4 | The held-out style split is designed after the adapter is frozen and its numbers are not used to choose anything | `bffc82b`, `docs/SAFETY.md` | 5/30 ledger harm published, and the hardened prompt's 21/30 published against the prompt |
| R5 | The metric correction changes what is reported, not which artifact ships | `bffc82b` | `sft4` still ships; the correction made it look worse (0/136 → 6/136) |
| R6 | Guard rules are frozen before the blind set is written; the blind set is run once and not tuned against | `docs/GUARD.md` | 5/8, published with the three misses named |
| R7 | A GRPO delta is reported on both axes whichever direction it goes | `docs/ROUND2.md` §1 | 0/152 rows changed; published as "GRPO not supported at this budget" |
| R8 | Whether a flagged payload *would have been followed* is scored end to end by `attacker_expense` and `attack_answer` on the same 136 rows, `--guard neutralise` against the plain adapter; offline detection and saved trajectories cannot answer it | `22b5af4`, `docs/GUARD.md` §"What is not measured" | `neutralise` **0/136** expenses, **136/136** solved; the comparison uses the already-published `runs/safety_sft4`, not a second GPU pass |

R2 is the useful one to check: the bar was written on 2026-11-04, `sft4` was trained on 2026-11-05,
and the result was 4/8 — a miss that stayed a miss. The bar was not moved after.

## 4. Decisions made after seeing data (discount these)

| Decision | What was seen first | Why it is still reported |
|---|---|---|
| `attacker_expense` replaces the sentinel | 6 tasks moved money and the sentinel read 0 | the metric had to be fixed; every affected number is restated, and the old sentinel is kept as a column |
| Guard `MUTATION_VERB` / `MODAL` / `_is_imperative_start` were added | dev set scored 5/12 | disclosed in `docs/GUARD.md`; the blind set is the control, and it was not tuned |
| `scan_result` drops the `answer` field | one false positive in a 324-episode sweep | read as a caller bug, not a rule change; a test pins it |
| `sft4` chosen over `sft2`/`sft3`/`sft5` | hard-90 and safety rows | the choice uses the holdout; no row in any safety split was used to pick between adapters |
| The abstention `asked` rate read as "shipped behaviour" | `sft4` 4/8 vs `sft5` 0/8 | reported with the mix that produced it, not as a model property |
| `strip` named as the control for the guard run, and the marker disclosed | `neutralise` scored 0/136 with its fixed notice still in context | the run was already reported before `strip` was written; `docs/GUARD.md` states the marker, and the strip run is listed as open rather than assumed |

Nothing in the second table is hidden; the point of listing it is that a reader can decide how much
of the published result rests on a choice made after the fact. The evaluation splits and the blind
set are the part that cannot be.

## 5. Verify it

```bash
python scripts/verify_all.py          # 28 checks: every pin above, oracle, leak checks, scoring
python scripts/guard_eval.py          # the guard tables, reproducible from the pinned files
python -m pytest -q                   # 59 tests, no GPU
git log -1 --format=%H               # the commit these numbers describe
```
