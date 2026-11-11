# Demo Day runbook — Nov 22

Five-minute slot. The last two minutes are buffer, so the script is 3:00 of content and 2:00 of
fallback talk. Everything before the live run is CPU or frozen and cannot fail; the live run is the
only step that touches a GPU, so it happens last, after the story is already on screen.

Numbers on this page are the frozen ones: `docs/RESULTS.md` (round 3, 2026-11-05), `docs/GUARD.md`
and `docs/STATS.md`. Do not improvise new ones.

---

## The number card (print this, keep it next to the screen)

| Claim | Number | Source |
|---|---|---|
| Shipped adapter, 30 unseen hard tasks | **29/30 = 96.67%** | `RESULTS.md#round-3` |
| Shipped adapter, 90 unseen hard tasks | **84/90 = 93.33%** | `RESULTS.md#round-3` |
| Three seeds pooled | **251/270 = 93.0%** hard-90 (bootstrap 87.4–97.4%), **88/90 = 97.8%** hard-30 | `RESULTS.md#aggregate` |
| Base Tiny Aya, same tasks | **1/30** and **3/90** | `RESULTS.md` |
| Tool calls per task | **3.38** against an oracle **3.41**; base **7.24** | `JUDGE.md` |
| Paired tests, base to shipped | hard-30 **28 fixed / 0 broken, p = 7.5e-9**; hard-90 **81 / 0, p < 1e-20** | `STATS.md` |
| Round 2 to shipped | **18 fixed / 2 broken, p = 0.0004** | `STATS.md` |
| Round 1 to shipped | 7 fixed / 3 broken, **p = 0.34** — say it, it is not significant | `STATS.md` |
| GRPO, run twice | **0 tasks moved on hard-90, p = 1.0** | `RESULTS.md` |
| Injection, four pinned splits, 136 rows | **6 ledgers, 0 answers** harmed (2/24 + 4/72 + 0/16 + 0/24) | `SAFETY.md` |
| Base, same splits | **18/24, 48/72, 5/16, 14/24** | `SAFETY.md` |
| Guard in front of the adapter, same 136 | **0/136 expenses, 136/136 solved, 136/136 flagged** — in `neutralise` (11,103 s) and in `strip` (10,567 s) | `GUARD.md` |
| Guard, published payloads / blind set / clean rows | **166/166**, **5/8**, **0/294** | `GUARD.md` |
| Abstention | **4/8** asked by default, **7/8** with the hardened prompt, **0/8** invented | `SAFETY.md` |
| Schema shift / Devanagari | **25/30** / **25/30**; untuned base under the rename **0/30** | `RESULTS.md#probes` |

---

## T-1 day (Sat Nov 21) — 15 minutes, do it once

- [ ] Open https://huggingface.co/spaces/primce/hinglish-expense-agent. If the badge says
      **SLEEPING**, this visit wakes it. Wait for the page to render, then close the tab.
- [ ] Download the fallback video **to the demo laptop** and play the first 5 seconds:
      `curl -L -o lexsi_demo.mp4 https://huggingface.co/datasets/primce/lexsi-runs/resolve/main/demo/lexsi_track3_demo.mp4`
      (4.0 MB, 1:44). Do not rely on streaming it during the pitch.
- [ ] Log into huggingface.co **in the browser you will demo in**. ZeroGPU quota is per visitor per
      day; the logged-in account is the one with quota.
- [ ] Open the article once and scroll to the numbers table:
      https://huggingface.co/spaces/primce/hinglish-track3-article
- [ ] Print the number card above, or put it on a second screen.

## T-10 minutes — warm the one thing that can be cold

1. Open the Space. Confirm the header renders and the **Start here — no GPU needed** panel already
   shows a verdict. That panel is filled server-side, so if it renders, the app is alive.
2. Click **Replay (instant)** once. Under a second, or the Space is not ready — reload once, then
   move to the fallback plan below.
3. In **Try the guard yourself**, press **Run the guard** on whatever example is selected. Expect a
   verdict line with a microsecond count (300–700 µs). This confirms the CPU path.
4. In **Live run**, set **Mode → Shipped adapter (sft4)**, leave the default Sawal, press
   **Run on the model (~60 s first time)**. Wait for the verdict to say the ledger matches. This is
   run 1 of 3: it pays the model load once, and from here a run is seconds, not minutes.
5. Leave the tab scrolled to the top. Do not reload it.

**GPU budget, measured on the live Space (2026-11-11):** ZeroGPU reserves duration + 60 s from the
daily quota, so the 120 s cold call costs 180 s and each 90 s warm call costs 150 s. Anonymous
visitors get 2 minutes a day — **one run** — and a logged-in free account gets 5 minutes —
**three runs**. That is exactly the pre-flight plus the two optional live runs below, and nothing
else. Do not spend a fourth.

If step 4 fails with "ZeroGPU quota exceeded", do not spend time on it: the demo runs entirely on
frozen replays plus the guard panel, and the fallback plan below has the wording.

---

## Order of operations (what is on screen, in what order)

Tabs open, in this order: (1) the Space, (2) the article, (3) the GitHub README, (4) the downloaded
video in a local player. Close everything else. Browser zoom 125%. Do Not Disturb on.

### 0:00–0:25 — Problem

"Tiny Aya speaks Hinglish. Ask it to split a dinner bill and it will sound confident and get the
ledger wrong. This is money, so 'roughly right' is a failure. I measured that on a frozen split, and
then fixed it — 30 unseen tasks, base model versus one LoRA adapter, same prompt."

### 0:25–1:10 — The before/after, frozen (no GPU)

The Space is already showing **Frozen episode → Shipped adapter — correct split**: verdict, tool
trace, final ledger JSON. Point at the ledger, then at the call count.

"Same prompt, shipped adapter. Ledger matches the expected balances exactly. **29 of 30** on the
held-out hard split, **84 of 90** on the 90-task holdout, **3.38** tool calls per task where the
oracle needs 3.41 — 87 of 90 tasks take exactly the oracle's number of calls, three take one fewer,
none take more. Zero tool errors, zero redundant calls, zero loops."

Switch **Frozen episode** to **Base model — duplicate expenses**. Same scorer, same renderer.

"The base model on the same task: eight calls, duplicate expense, never finishes. Across the set it
solves **1 of 30**."

### 1:10–1:50 — The safety half, frozen (no GPU)

Scroll to **Same task, same injected note — the adapter is the only difference**. Two columns, both
frozen episodes from the safety split.

"Indirect prompt injection. A tool result carries a note telling the agent to add 999999 INR. On the
left, the base model executes it — the attacker's number is in the ledger. On the right, the same
task, same note, only the adapter differs: resisted. Across four pinned splits that is **6 ledgers
and 0 answers harmed out of 136**, against 18/24, 48/72, 5/16 and 14/24 for the base model. Paired
on task ids: 74 fixed, 0 broken, p = 1.1e-22. No LLM judge anywhere in that number."

### 1:50–2:35 — The guard, live, on the CPU (the demo that cannot fail)

Scroll to **Try the guard yourself**.

1. **Example → Note with an English payload**, **Guard mode → strip**, **Which tool returned it →
   add_expense**, press **Run the guard**.
   Expect: **Flagged** — `ignore_rules`, `ledger_instruction`, `ledger_instruction_hinglish`,
   `state_change_directive`, and *what the model would read* is the same JSON with the `note` key
   gone. 300–700 µs.
2. **Example → Blind b-05: guard misses this one**, press again.
   Expect: **Not flagged** — only `unusual_amount`; the note passes through unchanged.

"That is the honest shape of the defence. A deterministic guard sits between the tool and the model:
no model, no GPU, half a millisecond. On the 136 injected rows it takes the adapter from 6 to
**0/136** attacker expenses and still solves **136/136** — and it does that identically when the note
is replaced by a notice and when it is deleted outright, so the result is the removal, not the hint.
It flags 166 of 166 published payloads and 0 of 294 clean episodes. It also misses 3 of the 8 blind
payloads I wrote after freezing the rules — that is the second example you just saw. A finite verb
list has an edge, and I published mine."

### 2:35–3:00 — Close

"Everything is public: the adapter, the run logs with a `summary.json` per run, the frozen splits,
the guard, and this Space. The zero that matters most is the honest one — GRPO ran twice and moved
zero tasks, and the write-up explains why. Agent reliability is a safety property: an agent that
mis-copies state gives confidently wrong money answers at scale."

### If the live GPU run is healthy and there is time (optional, +40 s)

Runs 2 and 3 of the three. Set **Mode → Base model (adapter off)**, press **Run on the model** on
the default Sawal
("Rahul ne 1200 INR diye pizza ke liye..."), let it finish, then **Mode → Shipped adapter (sft4)** and
run again. Base ends in MISMATCH; the adapter's ledger shows Rahul 900 and the other three -300.
Warm, each run takes seconds. **Skip this if anything above was slow — the frozen panels already
made the point.**

---

## Fallbacks, in the order you should reach for them

| What happens | Do this | Say this |
|---|---|---|
| **ZeroGPU quota exceeded** / `CUDA out of memory` | Do nothing on the GPU. Use the frozen panel and the guard panel; skip the optional live run. | "The GPU quota is 2 minutes anonymously and 5 logged in, and this Space has already served today's traffic, so I'll use the frozen episodes — same scoring code, rendered without the model. Here is the same run recorded a few days ago." |
| **Space is slow or a click hangs** | Stop clicking. Open the article tab and scroll to the numbers table. | "I'll use the write-up, which carries the same numbers with their sources." |
| **Space is down / 500** | Play the local video from 0:40 (before/after) and 1:20 (injection), then the article table. | "This is the Space recording: same tasks, same tool trace." |
| **No network at all** | Local video, then the number card; the guard panel does not work either, so narrate the blind-set example from `docs/GUARD.md`. | "No network, so I'll describe the guard and point to the repo, which reproduces every number on CPU." |
| **A judge asks a live question** | Type it into **Sawal** with **Shipped adapter (sft4)** and run it once. If it fails or is ambiguous, say so and move on. Never debug on stage. | "That one is out of the frozen set — the adapter can still miss a task the split does not cover, and I'd rather show you the miss than hide it." |
| **The video is played** | It is from Nov 9. Its on-screen cards are current, but it says **23 offline checks** and does not contain the guard. | "The repo now runs **28** offline checks and has the guard panel you just saw; the video predates both." |

Do not apologise for a cold start. One sentence, then keep moving: **"The Space has been idle, so the
first model call takes one to three minutes — meanwhile, here is the same run frozen."**

---

## Q&A prep

- **Why not a bigger model?** Free compute, and the point is a small open-weights model that fits on
  commodity hardware. 3.35B merged runs on one T4.
- **Why not prompt engineering?** Prompt-only is the before column; it is 1/30. The prompt-level
  defence is a separate arm we also measured: the hardened prompt clears the abstention bar (7/8
  asked, 0/8 invented) but costs 2/24 attacker values in answers and is worse on the held-out
  injection split (21/30 dirty ledgers against 5/30), so it is documented, not shipped.
- **Is 30 tasks enough?** No — that is why the headline is the 90-task holdout, why three seeds
  pool to 251/270, and why every claim carries a paired exact McNemar test on shared task ids.
- **Why is eval-24 at 100%?** It is in-distribution; it is a regression check, not the headline.
- **Isn't this just SFT?** Yes, and the track asks for RL. GRPO ran twice and moved zero tasks; the
  diagnosis is reward saturation (tool names already correct, answers incomplete, so every rollout
  in a group got the same reward). What is missing is cold-start GRPO with no SFT warm start.
- **Why is round 1 to shipped only p = 0.34?** Because it is 7 fixed and 3 broken. The proven step
  is round 2 to shipped, 18 against 2.
- **Is the agent safe?** Measured, not asserted: 6 of 136 ledgers and 0 of 136 answers across four
  pinned splits, no LLM judge; with the guard, 0 of 136 and 136 of 136 still solved; and the guard's
  blind spots are published (5/8 on the blind set).
- **The injection card says "solved: no".** The task asks for Amit's signed balance (`-833.46`); the
  adapter settles the transfer correctly but phrases it as "Amit → Neha 833.46", so the
  answer-contains check fails while the ledger is right. That is the answer-shape miss class — 9 of
  the 90 holdout misses are this shape, and none of them is a tool error.
- **What would you do with $200 of compute?** Seeds are done; the spread is 1.11 pp on hard-90. Next
  is cold-start GRPO, a Devanagari injection set at scale, and the schema shift on hard-90 instead
  of hard-30.
- **What is the residual risk?** A payload that corrupts an existing expense instead of adding one
  would not register in the metric; obfuscated amounts still get through on 5 of 15 held-out rows;
  and an adaptive attacker who can see the eval is untested. All three are in the write-up.

## Do not say

- No "state-of-the-art", no "solves hallucinations". One tool surface, six task families, 136 safety
  rows.
- Do not quote a safety count from memory. The adapter's number is **6/136**, not zero; the zero is
  the adapter **plus the guard**.
- Do not claim the blind set is clean: it is 5/8, and that number is a feature of the pitch, not a
  weakness to hide.
