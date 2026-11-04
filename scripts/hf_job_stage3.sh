#!/usr/bin/env bash
# Stage 3 on a rented GPU: does the contrast set fix empty_balance without
# un-learning abstention?
#
#   bash scripts/hf_job_stage3.sh          # train sft4 + probe + full hard-30
#   FULL=1 bash scripts/hf_job_stage3.sh   # also hard-90, eval-24 and both safety rows
#
# The probe is the 30 hard-90 rows of the two families that carry almost every
# miss in round 2 (`empty_balance` 4/15, `convert_sum` 8/15). It costs ~5 h and
# answers the only question that decides whether a full run is worth it: did the
# model stop asking for a number on an empty ledger?
#
# Requires cwd = repo root and HF_TOKEN in the environment. A failing step prints
# FAILED and the rest still run, so a partial night still uploads its numbers.
set -uo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MODEL=CohereLabs/tiny-aya-global

step() {
  local label="$1"
  shift
  echo "=== $label: $(date -u +%H:%M:%S) ==="
  local start=$SECONDS
  "$@" || echo "$label FAILED"
  echo "=== $label done in $((SECONDS - start)) s ==="
}

step "verify" python scripts/verify_all.py

if [ ! -f runs/sft4_merged/config.json ]; then
  step "train sft4 (mix3, seed 42)" python scripts/train_sft.py \
    --tasks data/tasks_train_mix3.jsonl --epochs 2 --seed 42 \
    --out runs/sft4 --merge-out runs/sft4_merged
fi

[ -f runs/sft4_merged/config.json ] || { echo "no sft4, stopping"; exit 1; }

# The probe: only the two families round 2 lost, both from hard-90.
step "probe: hard-90 weak families" python scripts/run_baseline.py --model runs/sft4_merged \
  --tasks data/tasks_hard90_eval.jsonl --families empty_balance,convert_sum \
  --no-4bit --out runs/sft4_probe

step "sft4 hard-30" python scripts/run_baseline.py --model runs/sft4_merged \
  --tasks data/tasks_hard_eval.jsonl --no-4bit --out runs/sft4_hard30

step "upload sft4 (part 1)" python scripts/job_upload.py \
  --run sft4_probe --run sft4_hard30 --adapter runs/sft4:sft4_adapter

if [ "${FULL:-0}" = "1" ]; then
  step "sft4 hard-90" python scripts/run_baseline.py --model runs/sft4_merged \
    --tasks data/tasks_hard90_eval.jsonl --no-4bit --out runs/sft4_hard90
  step "sft4 eval-24" python scripts/run_baseline.py --model runs/sft4_merged \
    --tasks data/tasks_eval.jsonl --no-4bit --out runs/sft4_eval
  step "sft4 safety" python scripts/run_baseline.py --model runs/sft4_merged \
    --tasks data/tasks_safety_eval.jsonl --safety --no-4bit --out runs/safety_sft4
  step "sft4 safety + hardened prompt" python scripts/run_baseline.py --model runs/sft4_merged \
    --tasks data/tasks_safety_eval.jsonl --safety --hardened-prompt --no-4bit \
    --out runs/safety_sft4_hardened
  step "upload sft4 (part 2)" python scripts/job_upload.py \
    --run sft4_hard90 --run sft4_eval --run safety_sft4 --run safety_sft4_hardened
fi

echo "=== stage 3 finished: $(date -u +%H:%M:%S) ==="
