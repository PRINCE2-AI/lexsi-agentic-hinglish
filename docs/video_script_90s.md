# 90-second submission video

Not a required form field. Record it anyway: judges who will not open the Space will watch 90
seconds, and it is the only artifact where the tool trace is visible without them clicking.

Numbers here are frozen from `docs/RESULTS.md` (2026-11-07). Do not improvise new ones. Re-read
`docs/demo_script.md` for the 3-minute Demo Day version; this is the short cut.

## Recording setup

- Windows: **Win + Alt + R** (Game Bar) records the active window. OBS if you want a mic track tuned
  separately.
- 1920×1080, browser zoom **125%** so the tool trace is readable on a phone.
- One take is fine. No intro card, no music, no cursor tricks.
- Close every tab except: (1) the Space, (2) the README table, (3) the injection episode.
- Say the number **and** the sample size. "29 of 30", not "almost 97%".

## Shot list

| Time | Screen | Say (exact words) |
|---|---|---|
| 0:00–0:12 | Space, empty question box | "Tiny Aya speaks Hinglish. Ask it to split a dinner bill and it will sound confident and get the ledger wrong. This is money, so 'roughly right' is a failure." |
| 0:12–0:30 | Space, **Base model (adapter off)**, default question, run it | "Base model, 30 hard tasks: **3.3%** success. 16 of 30 episodes never finish, 7 tool calls where the oracle needs 3.6. One 'success' invents an expense and still lands on the right number." |
| 0:30–0:55 | Toggle **Shipped adapter (sft4)**, same question, run it. Point at `add_expense` → `convert_currency` → `get_balance` → `finish` | "Same prompt, shipped adapter. Right tool, right arguments, every time. **29 of 30** on the held-out hard split, **84 of 90** on the 90-task holdout, zero tool errors, zero redundant calls." |
| 0:55–1:12 | Injection scenario, shipped mode. Verdict panel | "Indirect prompt injection, 32 tasks, no LLM judge. Base writes the attacker's amount into the ledger in 13 of 24 tasks. The adapter: **0 of 24**. Attacker number inside the answer: 8 of 24 → **0 of 24**." |
| 1:12–1:28 | Terminal: `python scripts/verify_all.py` finishing, then the README table | "All 23 checks pass on a free CPU, no GPU, no token. Adapter, run logs, and the eval harness are public: GitHub PRINCE2-AI slash lexsi-agentic-hinglish." |
| 1:28–1:35 | Model card page, then stop | "GRPO ran twice and moved zero tasks. That zero is measured, diagnosed, and published — the bottleneck was data coverage, not RL." |

## What not to say

- No "state-of-the-art", no "solves hallucinations". It is one tool surface, 32 tasks.
- The schema-shift and Devanagari runs exist now (`sft4_shift_hard30`, `sft4_devanagari30`,
  `base_shift_hard30`), so one line at 1:28 is allowed and load-bearing:
  "Renamed every tool and parameter: **25 of 30** still solve, and the untuned model gets 0."
