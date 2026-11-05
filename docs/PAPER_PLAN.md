# Paper-grounded plan — what to build next and why

**Read:** 2026-11-02. Both PDFs read end to end (21 + 19 pages), not just abstracts.

| Paper | Venue / id | Pages read |
|---|---|---|
| Chu et al., *SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training* | ICML 2025, arXiv:2501.17161v2 | 21/21 |
| Qian et al., *ToolRL: Reward is All Tool Learning Needs* | arXiv:2504.13958v1 | 19/19 |

Repo state when this was written: `033ae85`, clean tree. SFT already measured at
**success 100% (24/24) on generated eval-24** and **12.5% baseline (3/24)**. That
100% is the problem this plan fixes: it is measured on the same templated
generator that produced the training set, which is exactly the condition under
which Chu et al. show SFT looks good and then collapses out of distribution.

---

## 0. The thesis in one sentence

> SFT installs the **protocol** (strict JSON, ledger copying, turn order);
> GRPO installs the **policy** (which tool, which arguments, when to stop).
> We prove this with a frozen hard/OOD split and a fine-grained tool reward,
> and we report the case where it fails.

Five decisions the papers force, before any more GPU time is spent:

1. **Stop treating eval-24 as evidence.** SFT scored 100% there. Build a hard,
   frozen, disjoint split and report the drop. Chu et al. measure SFT OOD drops
   of −8.1, −79.5, −5.6 and −33.2 points across four settings while RL gains
   +3.5, +11.0, +3.0 and +9.3 on the same shifts. If our SFT drops, that is not
   a bug in the model; that is the paper's central finding reproduced on a new
   task.
2. **Keep SFT light and keep it as the main arm.** Chu et al. show RL without
   SFT fails when the backbone does not follow instructions (their Fig. 9), and
   RL cannot rescue an extremely overfit SFT checkpoint (their Fig. 19). Our SFT
   is 75 optimizer steps of LoRA on 100 rows, so it is a warm start, not a
   collapsed checkpoint — but the cold-start control (A2) is mandatory to prove
   it.
3. **Replace the binary tool reward with ToolRL's decomposition.** Current
   `tool_reward` gives 1.0 if every expected tool name appears anywhere
   (`scripts/train_grpo.py:62-69`). ToolRL shows coarse/binary correctness
   rewards train worse than name + parameter-name + parameter-value matching,
   and that format should carry ~1/6 of the correctness signal.
4. **Do not add a length reward.** ToolRL's first ablation: length rewards
   increase reasoning length and *hurt* tool-use accuracy (e.g. Qwen2.5-1.5B
   46.20% → 33.23%). Our tasks are short; length is pure risk.
5. **Add a schema-shift evaluation.** ToolRL trains on Hammer-masked data
   (randomised tool/parameter names) precisely to stop models memorising
   strings. We cannot retrain on masked data in time, but we can *measure*
   whether our model follows the schema it is given or the names it memorised.
   This is the cleanest evidence that SFT learned a protocol vs. a template.

---

## 1. What each paper actually gives us

### 1.1 Chu et al. — claims, exact numbers, use

| Finding | Number from the paper | What we do with it |
|---|---|---|
| RL with outcome reward generalizes to unseen **rules** | GP-L OOD 11.5→15.0 (+3.5); V-IRL-L 80.8→91.8 (+11.0); GP-VL 11.2→14.2 (+3.0); V-IRL-VL 35.7→45.0 (+9.3) | Hard split must change *rules*, not just wording: payer-excluded + outsider + 5 expenses + two questions in one episode |
| SFT memorizes and degrades OOD | GP-L 11.5→3.4 (−8.1); V-IRL-L 80.8→1.3 (−79.5); GP-VL 11.2→5.6 (−5.6); V-IRL-VL 35.7→2.5 (−33.2) | Report SFT easy-eval vs SFT hard-eval side by side; the gap is the result |
| RL also generalizes to **visual** shifts | +17.6 (23.6→41.2) and +61.1 (16.7→77.8); SFT −9.9 and −5.6 | Our analogue is **script/surface shift**: Devanagari and unseen Hinglish word order |
| SFT is necessary for RL when the base does not follow instructions | End-to-end RL on base fails; outputs are long, tangential, unparseable (Fig. 9, 20) | Justifies SFT→GRPO as the main arm; A2 cold start is the control that shows our base *can* emit JSON (strict_rate 0.584), unlike their Llama |
| RL cannot rescue an extremely overfit SFT checkpoint | OOD per-step accuracy stays <1% (Fig. 19, 21) | If A3 fails to move hard-OOD, do not blame RL blindly: diagnose whether SFT collapsed, and run A2 |
| More verification iterations improve OOD | +0.48% (1 iter), +2.15% (3), +2.99% (5), +5.99% (10) | Keep the repair loop; report parse-failure repairs and a `--max-steps 4` ablation, not a new mechanism |
| SFT on sub-optimal trajectories still memorizes | Fig. 15: more diverse SFT data does not fix OOD | Do **not** spend a run on noisy SFT data as a fix. Fix the *policy* with RL |
| Compute-matched comparison | `X_train = 6ND`, `X_infer = 2ND` | Ship a compute table; it is cheap and judges can check it |

### 1.2 ToolRL — claims, exact numbers, use

| Finding | Number from the paper | What we do with it |
|---|---|---|
| Fine-grained reward beats binary | Qwen2.5-1.5B 46.20% vs coarse 36.72%; Qwen2.5-3B 52.98% vs 51.40% | Replace binary `tool_reward` with name / param-name / param-value components |
| Correctness should outweigh format | format ∈ {0,1}, correctness ∈ [−3,3]; equalising them ("Equal max") drops 46.20→39.47 | Total reward: format 1, correctness 6, answer 2 |
| Length reward hurts | 46.20→33.23 (1.5B), 52.98→48.89 (3B) | No length reward. Say so in the write-up |
| Abrupt two-stage scaling hurts; smooth dynamic helps slightly | 46.20→38.85 (two-stage), 45.71 (dynamic); 52.98→50.66 / 53.81 | Skip the schedule at 60 steps; report as a deliberate limitation |
| SFT-initialised GRPO can generalize *worse* than cold start | 7B API-Bank: SFT4k+GRPO 32.0% vs cold start 72.0% | A2 is not optional. If A2 beats A3 on hard-OOD, that is the headline |
| Remove KL, temperature 1.0 | Table 8: KL false, 4 rollouts, LR 1e-6 | Try `beta=0` if AgentTune forwards it; otherwise keep default and say so |
| Tool-name masking prevents memorisation | Hammer (masked) is 1 of 3 training sets | Schema-shift eval; honest diagnostic, not a headline claim |
| Relevance / irrelevance detection is a first-class metric | BFCL column | Empty-ledger and no-mutation tasks in the hard split; report "invented expense" rate |

---

## 2. Build the missing benchmark (no GPU)

### 2.1 Why

`tasks_eval.jsonl` shares the generator, templates, amount pools and phrasing
with `tasks_train.jsonl`. 100% there measures template fit. The hard split must
change **rules and surface form**, and must be frozen before any training run.

### 2.2 Files

| File | Contents |
|---|---|
| `data/build_hard.py` | deterministic generator, seed `20261103`; writes `tasks_hard_train.jsonl` (40), `tasks_hard_eval.jsonl` (30), `tasks_train_all.jsonl` (100 easy + 40 hard) |
| `data/hard_eval.sha256` | hash of the frozen eval file, written once, committed before training |
| `data/task_builder.py` | extend `finalize` with new query types (below) |
| `tests/test_hard_data.py` | oracle + disjointness + freeze-hash assertions |

### 2.3 Hard families (5 × 6 eval = 30; 5 × 8 train = 40)

| Id | Family | What changes vs. train-100 | Query type |
|---|---|---|---|
| H1 | `settle_5` | 5 expenses, 4–6 people, mixed INR/USD/EUR, at least one payer excluded | `settle` |
| H2 | `two_query` | one episode asks for a balance **and** the settlement; both must be in `finish` | `balance_and_settle` (new) |
| H3 | `trap_combo` | payer-out **and** outsider query **and** a zero-sum pair, all in one task | `balance` / `settle` |
| H4 | `paraphrase` | Devanagari script, postposed word order, `hisab` / `paisa` / `kitna dena hai`; never a template string from train | `balance` / `settle` |
| H5 | `convert_multi` | two conversions that must be summed; plus one empty-ledger task ("koi expense nahi hua") | `convert_sum` (new) / `empty_balance` (new) |

Rules:

- Disjoint from `tasks.jsonl`, `tasks_train.jsonl`, `tasks_eval.jsonl`,
  `tasks_hard_train.jsonl`. Dedupe on normalised text **and** on
  `(payer, amount, currency, tuple(sorted(participants)), description)`.
- Every row passes `run_oracle` + `score_episode` with `success=1`, `ter=0`,
  `strict=1`.
- `two_query`: reject a draw where the queried balance equals any transfer
  amount (otherwise `answer_contains` needle collapse makes the check weaker).
- H5 `convert_sum`: `answer_text` is the summed INR total; `answer_contains`
  is the single total; `expected_tools=["convert_currency"]`; two calls.
- H5 `empty_balance`: no actions except `get_balance` on the empty ledger;
  `checks_ledger=False`, answer `0.00`.

### 2.4 `finalize` extensions (`data/task_builder.py`)

```python
# balance_and_settle
expected_settlements = settle_ledger(ledger)          # not [] as today
answer = f"{person} ka balance {bal:.2f} INR hai. " + settlement_text
contains = [fmt(bal)] + [fmt(t["amount_inr"]) for t in transfers]

# convert_sum
total = sum(execute_tool("convert_currency", item)["value"] for item in query["items"])
answer = f"Total {total:.2f} INR."
contains = [fmt(total)]

# empty_balance
answer = f"{person} ka balance 0.00 INR hai."
contains = ["0.00"]
```

Add `require_answer: bool` to the emitted task for `balance_and_settle`, and in
`hinglish_agent/runner.py::score_episode`:

```python
if task.get("checks_ledger"):
    success = ledger_ok and settlement_ok and (answer_ok or not task.get("require_answer"))
```

### 2.5 Acceptance

```bash
python data/build_hard.py
python - <<'PY'
import json, hashlib, pathlib
d = pathlib.Path("data")
def rows(p): return [json.loads(l) for l in (d/p).read_text(encoding="utf-8").splitlines() if l.strip()]
sets = {p: rows(p) for p in ["tasks.jsonl","tasks_train.jsonl","tasks_eval.jsonl","tasks_hard_train.jsonl","tasks_hard_eval.jsonl"]}
all_train = {t["task"].strip().lower() for p in ["tasks.jsonl","tasks_train.jsonl","tasks_hard_train.jsonl"] for t in sets[p]}
leak = [t["id"] for t in sets["tasks_hard_eval.jsonl"] if t["task"].strip().lower() in all_train]
print("hard-eval leak:", leak); assert not leak
print("hard eval n:", len(sets["tasks_hard_eval.jsonl"]), "hard train n:", len(sets["tasks_hard_train.jsonl"]))
print("sha256:", hashlib.sha256((d/"tasks_hard_eval.jsonl").read_bytes()).hexdigest())
PY
python scripts/run_oracle_eval.py --tasks data/tasks_hard_eval.jsonl
python tests/test_hard_data.py
```

Expected: `0` leaks, `30 / 40` rows, oracle `30/30` and `40/40` success,
`ter=0`, `strict=1`.

### 2.6 Freeze

```bash
python data/build_hard.py --freeze   # writes data/hard_eval.sha256, refuses to overwrite
```

From that moment the hard eval file is read-only by convention, never used in
rewards, never used for checkpoint selection. Selection uses `tasks_eval.jsonl`
or train loss only; the hard number is reported, not optimised.

---

## 3. Replace the reward (ToolRL §3.3 → our contract)

### 3.1 Formula

For one completion, with predicted calls `P`, oracle calls `G` (task actions +
`finish`), ledger key excluded from matching (it is environment state, not a
decision):

| Component | Definition | Scale |
|---|---|---|
| `format` | 1.0 strict bare JSON array, exactly 1 call; 0.5 parseable but fenced/prose; 0.0 unparseable | 0 / 0.5 / 1 |
| `name` | `|N_P ∩ N_G| / |N_P ∪ N_G|` over tool-name sets | 0…1 → ×3 |
| `param` | greedy optimal matching of `P` to `G`; mean key-Jaccard over matched pairs | 0…1 → ×1.5 |
| `value` | same matching; exact value agreement (numbers ±0.01, participant lists as sets, enums case-insensitive) | 0…1 → ×1.5 |
| `valid` | 0.5 if any call has an unknown tool, non-dict parameters, bad enum, negative amount, or empty participants; else 1.0 | multiplier on `param+value` |
| `answer` | existing numeric containment vs `answer_text`, on the finish turn only | 0…1 → ×2 |
| `efficiency` | 1.0 if no duplicate `(tool, args)` in the turn (A5 only) | 0…1 → ×0.5 |

`R = format + 3·name + 1.5·valid·param + 1.5·valid·value + 2·answer (+0.5·efficiency)`

Range 0…9.5, correctness 6× format — matches ToolRL's 6:1 weighting without
their negative range (we never punish a valid intermediate step, because a
ledger episode legitimately needs several).

### 3.2 Implementation

- Keep one function per component so TRL logs each curve separately:
  `format_reward`, `name_reward`, `args_reward`, `answer_reward`, `efficiency_reward`.
- Dataset rows gain `oracle_actions_json` (JSON string; nested dict columns are
  brittle in `datasets`). Build it in `train_grpo.py::build_dataset` from
  `task["actions"]` plus a synthetic `finish` action with the oracle answer.
- `--reward-mode {binary,graded}`: binary keeps today's functions for A4.
- `--check-rewards`: run every component on (a) the oracle replay, (b) a
  fenced-JSON replay, (c) a wrong-ledger replay, (d) prose. Print the vector.
  No GPU, no agenttune required.
- Pure functions, no globals: unit-testable on Windows.

### 3.3 Acceptance

```bash
python scripts/train_grpo.py --check-rewards
python tests/test_rewards.py
```

Expected: oracle turn → `format=1, name=3, args≈3, answer=2`; fenced → `0.5`;
prose → `0,0,0,0`; wrong-ledger → `args < 1.5`; unknown tool → `valid=0.5`.

### 3.4 Deliberately not doing

- No length reward (ToolRL ablation 1).
- No dynamic reward schedule (60 steps is too short to schedule; ToolRL's gain
  was +0.83 on one model, −0.7 on another).
- No LLM judge anywhere. Every reward is deterministic.

---

## 4. Experiment matrix

| Arm | Init | Train data | Reward | Steps | Proves | GPU est. |
|---|---|---|---|---:|---|---:|
| A0 | base | — | — | 0 | before number on hard-30 + schema shift | 1.5 h |
| A1 | SFT (existing `runs/sft`) | — | — | 0 | SFT fits ID, drops on hard-OOD (Chu) | 1.5 h |
| A2 | base | easy+hard train | graded | 20 | cold-start RL works because our base emits JSON; ToolRL's cold-start result | 2.5 h |
| A3 | SFT | easy+hard train | graded | 60 | main result: RL recovers hard-OOD and keeps efficiency | 4 h |
| A4 | SFT | easy+hard train | binary | 30 | ToolRL granularity ablation at our scale | 2 h |
| A5 | SFT | easy+hard train | graded + efficiency | 30 | efficiency claim (calls ≤ A3) | 2 h |

Run order: A0/A1 evals → reward smoke → A3 → A2 → A4 → A5 only if time.
Drop A4 first if GPU quota is tight; A2 never (it is the control that makes the
paper comparison honest).

Commands:

```bash
# A0 / A1 on the hard split
python scripts/run_baseline.py --model CohereLabs/tiny-aya-global \
  --tasks data/tasks_hard_eval.jsonl --out runs/a0_hard
python scripts/run_baseline.py --model CohereLabs/tiny-aya-global \
  --adapter runs/sft --no-4bit --tasks data/tasks_hard_eval.jsonl --out runs/a1_hard

# smoke: 5 steps, graded reward, from SFT
python scripts/train_grpo.py --model runs/sft_merged --tasks data/tasks_train_all.jsonl \
  --reward-mode graded --max-steps 5 --out runs/grpo_smoke

# A3
python scripts/train_grpo.py --model runs/sft_merged --tasks data/tasks_train_all.jsonl \
  --reward-mode graded --max-steps 60 --out runs/a3_grpo

# A2
python scripts/train_grpo.py --model CohereLabs/tiny-aya-global --tasks data/tasks_train_all.jsonl \
  --reward-mode graded --max-steps 20 --out runs/a2_cold

# A4
python scripts/train_grpo.py --model runs/sft_merged --tasks data/tasks_train_all.jsonl \
  --reward-mode binary --max-steps 30 --out runs/a4_binary
```

### Gates

| Gate | Condition | If it fails |
|---|---|---|
| G1 rewards | `--check-rewards` matches §3.3 | fix reward code, do not start GRPO |
| G2 variance | `frac_reward_zero_std < 1` within 5 steps | raise `--generations` to 4, temperature 1.0, or mix easier rows; drop A2 if base is hopeless |
| G3 main | A3 hard success ≥ A1 + 10 pts | ship A1; report GRPO as a negative result with the reward curves |
| G4 efficiency | A3 calls ≤ A1 calls + 0.5 at no more than −2 pts success | efficiency claim is dropped, not fudged |
| G5 cold start | if A2 > A3 by ≥5 pts on hard-OOD, report it as the finding | changes the headline, not the write-up honesty |

### SFT rules from the papers

- Do **not** retrain SFT on the hard split before A3. The memorisation contrast
  needs SFT trained on easy data only. If A3 fails G3 and A2 also fails, the
  fallback is a 1-epoch SFT on `tasks_train_all.jsonl`, clearly labelled as a
  different arm (A1b), not silently swapped in.
- Keep SFT at 3 epochs / 75 steps. Chu's Fig. 19 warns that a very overfit
  checkpoint cannot be rescued by RL; longer SFT is not free.

---

## 5. Measurements to add

| Metric | Where | Why |
|---|---|---|
| `overcall` = `max(0, calls − (len(actions)+1))` | `runner.score_episode` | Track 3 efficiency; ToolRL reports tool-call count |
| `parse_failures` | `runner.score_episode` (steps with `name=None`) | separates format failure from tool error (`ter`) |
| `invented_expense` | new flag: `ledger.expense_count > oracle expense count` | relevance / irrelevance analogue |
| Wilson 95% CI on success | `metrics.summarise` | n=30 needs an interval; stops over-claiming |
| compute table | `scripts/compute_accounting.py` | `6ND` train, `2ND` inference; tokens from logs |
| reward components | TRL logs per component | shows *why* an arm moved |

Tables to ship (each row traceable to a `summary.json`):

1. **Main:** A0 / A1 / A3 × {starter-20, easy-eval-24, hard-eval-30} — success,
   strict, ter, calls, overcall, seconds, n, CI.
2. **Ablation:** A2 (cold start), A4 (binary reward), A5 (efficiency).
3. **Schema shift:** A0 / A1 / A3 under canonical vs aliased tool names.
4. **Reward curves:** format / name / args / answer means + `frac_reward_zero_std`.
5. **Compute:** tokens, FLOPs, GPU-hours, wall-clock per arm.
6. **Error taxonomy:** hard-30, A1 vs A3, counted by class (no JSON, wrong tool,
   bad args, wrong ledger copy, wrong split, wrong number, loop).

---

## 6. Schema-shift evaluation (ToolRL Hammer-masked analogue)

New files: `hinglish_agent/aliases.py`, `scripts/eval_schema_shift.py`,
`tests/test_aliases.py`.

Alias map (one direction, model sees aliases, executor sees canonical):

```text
add_expense→split_cost      convert_currency→fx_convert   get_balance→net_balance
settle_up→settle_all        finish→respond
ledger→state  payer→paid_by  amount→total  currency→unit
participants→people  description→note  person→member  answer→message
from_currency→from_unit  to_currency→to_unit
```

- Build an aliased system prompt from `TOOL_SCHEMAS` (same rules, new names).
- `build_messages` / `run_episode` gain an optional `system_prompt` argument
  (default `SYSTEM_PROMPT`, so nothing else changes).
- Translate predicted calls back to canonical before `execute_tool`.
- Score with the same `score_episode`; report success + strict under both prompts.

Run on A0, A1, A3 (24 easy or 30 hard tasks, ~1 h each). Interpret carefully:

- A1 collapses, A3 recovers → SFT memorised names, RL learned the schema. Strong.
- Both collapse → the task is too far OOD; report as a limitation, not a failure.
- Both hold → the model genuinely reads schemas. Also a good result.

Do not put schema-shift in the headline. It is a diagnostic with n≤30.

---

## 7. Write-up mapping (claim → evidence → citation)

| Section | Claim | Evidence | Cite |
|---|---|---|---|
| Method | SFT warm start then GRPO; SFT is the format teacher | A1 vs A0 strict JSON | Chu §5.4 (SFT necessary when base ignores instructions) |
| Method | Fine-grained reward: name + param names + values, correctness 6× format | reward functions + G1 | ToolRL §3.3 |
| Results | SFT fits ID but drops on rule shifts | A1 easy-24 vs A1 hard-30 | Chu §5.1 exact OOD deltas |
| Results | GRPO recovers hard-OOD | A3 vs A1 hard-30, with CI | Chu §5.1 |
| Results | GRPO does not buy efficiency for free | A3/A5 calls, overcall | Track 3; ToolRL reports call counts |
| Ablation | Binary reward is worse than graded | A4 vs A3 | ToolRL §5.3 |
| Ablation | Cold start vs SFT init | A2 vs A3 | ToolRL §4.3 (SFT-init can generalise worse) |
| Diagnostic | Schema shift separates memorised names from schema following | canonical vs aliased | ToolRL §4.1 (Hammer masked) |
| Limitations | n=30 hard, 60 GRPO steps, three mix3 seeds (1.11 pp spread), fixed FX, six names | CI + logs | — |
| Limitations | No length reward, no dynamic schedule | deliberate, cited | ToolRL §5.1, §5.2 |

Allowed sentences: "on our frozen 30-task hard split, SFT drops from X to Y and
GRPO raises it to Z (±CI)". Not allowed: "RL generalises" as a general claim,
"SOTA", or any competitor comparison.

---

## 8. Schedule (Nov 3–14) and GPU budget

Kaggle 2×T4, one GPU at a time, ~30 GPU-h/week.

| Date | Work | GPU-h |
|---|---|---:|
| Nov 2 (today) | this plan; local design only | 0 |
| Nov 3 | `build_hard.py` + `finalize` extensions + tests + freeze hash; A0 hard eval; A1 hard eval | 3.0 |
| Nov 4 | graded reward + `--check-rewards` + tests; schema-shift harness; A0/A1 schema-shift | 3.0 |
| Nov 5 | GRPO 5-step smoke (G2); A3 block 1 (30 steps) | 3.5 |
| Nov 6 | A3 block 2 (30 steps); A3 hard eval | 3.5 |
| Nov 7 | A2 20 steps; A2 hard eval | 3.5 |
| Nov 8 | A4 30 steps + eval; A5 only if quota allows | 3.0 |
| Nov 9 | final evals (starter-20, easy-24, hard-30) for the chosen arms; error taxonomy | 3.0 |
| Nov 10 | freeze results; AuditKIT; compute table | 0.5 |
| Nov 11 | HF model / dataset / Space / Collection | 0 |
| Nov 12 | article + recording | 0 |
| Nov 13 | submit early | 0 |
| Nov 14 | buffer only | 0 |

Total ≈ 23 GPU-h, inside 30. If a session dies, A3 restarts from the last
checkpoint or from SFT (SFT is intact and cheap). Never run two notebooks on the
same account.

---

## 9. Failure modes and the honest fallbacks

| Risk | Signal | Response |
|---|---|---|
| Hard split is unfair / unsolvable | oracle gate fails | fix the generator, never hand-edit the JSONL |
| Hard eval leaks into training | dedupe test fails | regenerate with a new seed, re-freeze |
| Graded reward is gamed (e.g. spams `add_expense`) | name/args high, success flat; `rad`/overcall rise | add `valid` penalty for stale ledger, keep efficiency arm, inspect completions |
| GRPO reward variance stays 0 | `frac_reward_zero_std=1` | A2 first; otherwise SFT is too good on easy rows — sample harder rows only |
| A3 worse than A1 | G3 fails | ship SFT-only; the negative result is still a submission with a real ablation |
| A2 beats A3 | G5 | headline becomes "cold-start GRPO > SFT-init on OOD", still consistent with ToolRL |
| Schema shift 0% for everyone | diagnostic floor | report as limitation, do not hide |
| GPU quota exhausted | Kaggle refuses | A0/A1/A3 only; drop A4/A5 and schema shift on A3 |
| Time overrun | Nov 10 | freeze whatever is measured; write-up > extra run |

---

## 10. Work order (do in this sequence)

1. [ ] `data/task_builder.py`: add `balance_and_settle`, `convert_sum`,
   `empty_balance`, `require_answer`. `tests/test_task_builder.py`.
2. [ ] `data/build_hard.py`: 5 families, 40 train / 30 eval, dedupe vs all
   splits, oracle gate, `--freeze` hash.
3. [ ] `tests/test_hard_data.py`: counts, disjointness, hash, oracle.
4. [ ] `hinglish_agent/runner.py`: `overcall`, `parse_failures`,
   `invented_expense`, `require_answer` success rule.
5. [ ] `hinglish_agent/metrics.py`: Wilson CI; new keys in table.
6. [ ] `scripts/train_grpo.py`: graded reward components, `--reward-mode`,
   `--check-rewards`, `oracle_actions_json` in `build_dataset`. Keep binary path.
7. [ ] `tests/test_rewards.py`: oracle / fenced / prose / wrong-ledger / unknown tool.
8. [ ] `hinglish_agent/aliases.py` + `scripts/eval_schema_shift.py` +
   `tests/test_aliases.py`; optional `system_prompt` in `build_messages` /
   `run_episode`.
9. [ ] `scripts/compute_accounting.py`: token counts → `6ND` / `2ND`, GPU-hours.
10. [ ] Run A0/A1 on hard-30 + schema shift; fill table 1 and 3.
11. [ ] Reward smoke, then A3; then A2; then A4/A5 if quota.
12. [ ] Freeze Nov 10; update `README.md`, `docs/hf_article.md`,
    `docs/PLANBOOK.md` from `summary.json` only.
13. [ ] HF model/dataset/Space/Collection; submit Nov 13.

**Definition of done:** every number in the write-up comes from a committed
`runs/*/summary.json`; the hard eval hash in the article matches
`data/hard_eval.sha256`; A1 vs A3 on hard-30 has a Wilson interval; and the
article states one thing that did not work.
