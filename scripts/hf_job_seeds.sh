#!/usr/bin/env bash
# Seed control for whichever SFT mix ends up shipping.
#
#   MIX=data/tasks_train_mix5.jsonl NAME=sft5 bash scripts/hf_job_seeds.sh
#
# Trains the same recipe at seeds 123 and 7 (the shipped seed stays 42) and evaluates
# each on hard-90 and hard-30, so the seed-spread claim in docs/RESULTS.md covers the
# adapter that actually ships instead of a retired one.
set -uo pipefail

MIX="${MIX:?set MIX to the training file}"
NAME="${NAME:?set NAME to the run prefix, e.g. sft5}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

step() {
  local label="$1"
  shift
  echo "=== $label: $(date -u +%H:%M:%S) ==="
  local start=$SECONDS
  "$@" || echo "$label FAILED"
  echo "=== $label done in $((SECONDS - start)) s ==="
}

run_seed() {
  local seed="$1"
  local out="runs/${NAME}_seed${seed}"
  local merged="runs/${NAME}_seed${seed}_merged"

  step "train ${NAME} seed ${seed}" python scripts/train_sft.py \
    --tasks "$MIX" --epochs 2 --seed "$seed" --out "$out" --merge-out "$merged"
  [ -f "$merged/config.json" ] || { echo "no ${merged}, skipping seed ${seed}"; return; }

  step "${NAME} seed ${seed} sanity" python scripts/adapter_sanity.py \
    --merged "$merged" --limit 3 --out "runs/${NAME}_seed${seed}_sanity/summary.json"
  step "${NAME} seed ${seed} hard-90" python scripts/run_baseline.py --model "$merged" \
    --tasks data/tasks_hard90_eval.jsonl --no-4bit --out "runs/${NAME}_seed${seed}_hard90"
  step "${NAME} seed ${seed} hard-30" python scripts/run_baseline.py --model "$merged" \
    --tasks data/tasks_hard_eval.jsonl --no-4bit --out "runs/${NAME}_seed${seed}_hard30"
  step "upload ${NAME} seed ${seed}" python scripts/job_upload.py \
    --run "${NAME}_seed${seed}_hard90" --run "${NAME}_seed${seed}_hard30" \
    --run "${NAME}_seed${seed}_sanity" \
    --adapter "$out:${NAME}_seed${seed}_adapter"
}

step "verify" python scripts/verify_all.py
run_seed 123
run_seed 7

echo "=== seeds finished: $(date -u +%H:%M:%S) ==="
