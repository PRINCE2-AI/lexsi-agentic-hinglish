# Demo Day — questions and one-line answers

Every answer below points at a published artifact; nothing here depends on a memory of the run.
Keep the number, then the file. Do not volunteer a number that is not in `docs/STATS.md`.

## Likely questions

**1. "Tiny Aya is a Global model — where is the multilingual work?"**
The user turns are Roman-script Hinglish, which is the code-mixed case that breaks tool calling.
Two Devanagari probes are published. The first is a 30-task rewrite of hard-30: **25/30** for the
adapter, no base run on that file. The second was built on 15 Nov with a new seed and the base
measured on the same ids: **29/30 against 1/30**, 28 fixed and 0 broken (p = 7.5e-9), and AuditKIT
returns 29/30 under both 1.1.3 and 1.1.6. The second probe's template fixes two grammar nits from
the first, so they are reported side by side, not merged. Hindi beyond Devanagari, other Indic
languages and other scripts are not measured and the write-up says so. `docs/RESULTS.md`.

**2. "The 257 tasks are yours — isn't that your own benchmark?"**
The split was built on 2026-11-13 from a fresh seed, after the adapter was frozen on 2026-11-05, and
pinned by sha256 before the run. The base scores **0/257** on the same ids and the organiser's
scorer agrees on all 257 (agreement **1.0000**). `docs/PREREGISTRATION.md`, `docs/RESULTS.md`.

**3. "0/257 for the base looks too clean."**
It is consistent with 1/30 and 3/90: the base finds one accidental success in each smaller split
and none at 257. It is published as a floor with a **0.0–1.5%** Wilson interval, not as a rate.
The base's behaviour is described from hard-30, where its one hit came after inventing an 800 INR
expense. `docs/RESULTS.md#hard-257`.

**4. "Your guard scores 5/8 on the blind set — is that a security boundary?"**
No. It is a filter, not a proof: 166/166 on published payloads is in-sample, **5/8** on the blind
set is the generalisation estimate, and **6/8** on a probe written with the source open. It flags
0 of 1,010 clean tool results. That is why it is a second line under the model, not the claim.
`docs/GUARD.md`.

**5. "14 of 22 misses are one phrasing — isn't the adapter brittle?"**
Yes, and it is published with the task ids rather than patched. The 90-task split never sampled
that construction; the 257-task split found it at 14/14. Without those rows the run is
235/243 = 96.7%. A fix is a training-data change after the freeze, so it stays a named limitation.
`docs/RESULTS.md`, `docs/WHAT_DIDNT_WORK.md`.

**6. "Why no LLM judge?"**
Success is the final integer-paise ledger against a deterministic oracle, and a second scorer
(AuditKIT, recorded mode) re-derives it: **235/257 with agreement 1.0000**. An LLM judge would add
variance, not independence. `docs/RESULTS.md#auditkit`, `scripts/auditkit_eval.py`.

**7. "Is this reproducible, or just a notebook?"**
`python scripts/verify_all.py` runs 30 checks with no GPU and no token (31 with `--with-tests`);
`python scripts/bootstrap.py`
rebuilds every interval and paired test from `primce/lexsi-runs`; `python scripts/manifest.py
--check` re-hashes 87 files and fails on drift. `docs/JUDGE.md`.

**8. "What does it cost?"**
**$7.39 per 1,000 tasks** self-hosted on one T4 at $0.53/GPU-hour, 3.38 calls and 50.2 s per task on
hard-90; 72x cheaper per solved task than the cheapest closed tier because of accuracy, not the
price list. The 257-task run took 61.7 s per task. `docs/roi.md`.

**9. "Where would this be used?"**
Any expense-splitting or reimbursement agent where the assistant reads tool results written by
someone else; the injection half applies to every tool-using agent, not only ledgers. The guard is
a standalone function and has a no-GPU playground in the demo Space.

**10. "What would you do with another week?"**
Add the two currency-sum phrasings to training data and re-freeze, extend the Hindi probe beyond one
split, and run the guard against a second independent attack author. Those are named as not done in
`docs/WHAT_DIDNT_WORK.md`, not hidden.

## Run of show (2:50, hard stop 3:03)

| Time | Beat | What is on screen |
|---|---|---|
| 0:00–0:20 | Problem: a Hinglish expense agent that reads tool results | Article Space, first paragraph |
| 0:20–1:00 | Live demo: base → adapter on one frozen episode | Demo Space, Frozen episode |
| 1:00–1:50 | Numbers: 3.33% → 91.44%, 235 fixed / 0 broken, AuditKIT 1.0000 | `docs/STATS.md` |
| 1:50–2:30 | Safety: 13/208 → 0/208, guard playground | Guard panel, blind `b-05` |
| 2:30–2:50 | Limits and one next step | `docs/RESULTS.md` misses table |

Fallback: the 1:44 narrated video is published at
`https://huggingface.co/datasets/primce/lexsi-runs/resolve/main/demo/lexsi_track3_demo.mp4`.
If the Space is quota-blocked, play the video and keep the same beats.

## Rehearsal plan (three passes, 27 minutes total)

The stage-1 deadline is 16 Nov, 17:29 IST; Demo Day is 22 Nov. Rehearsals go after the submission is
filed, so a late number change never reaches a rehearsed script.

| When | Pass | Pass/fail |
|---|---|---|
| 16 Nov, evening | Warm run: open both Spaces cold, run the frozen episode and the guard `b-05` panel once | Everything loads in under 60 s without touching a terminal |
| 19 Nov | Timed run: the full run of show with a visible timer, questions 1–5 asked by someone else | Lands at 2:50 or earlier; no answer longer than 30 s |
| 20 Nov | Failure run: start from the video fallback, then switch to the live Space mid-talk | Same beats, no dead air, numbers still land on `docs/STATS.md` |

Checklist before each pass: demo Space responds, frozen episode replays base then adapter, guard panel
returns `b-05` flagged with the ledger unchanged, the video URL returns HTTP 200, and `docs/STATS.md`
is open in a second tab so every number said aloud is on screen. If a published number changes
between passes, say the new number and point at the file; do not re-quote from memory.
