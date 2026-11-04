#!/usr/bin/env bash
# One extra SFT seed, measured on both held-out splits.
#
#   SEED=123 bash scripts/hf_job_seed.sh
#   SEED=7   bash scripts/hf_job_seed.sh
#
# Same recipe as stage 1 (mix2, 2 epochs, r=16, lr 2e-4); only the LoRA init and
# data order change, which is exactly the variance the single-seed result hides.
set -uo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SEED=${SEED:-123}
TAG=sft3_seed${SEED}

step() {
  local label="$1"
  shift
  echo "=== $label: $(date -u +%H:%M:%S) ==="
  local start=$SECONDS
  "$@" || echo "$label FAILED"
  echo "=== $label done in $((SECONDS - start)) s ==="
}

step "train $TAG" python scripts/train_sft.py \
  --tasks data/tasks_train_mix2.jsonl --epochs 2 --seed "$SEED" \
  --out "runs/$TAG" --merge-out "runs/${TAG}_merged"

if [ -f "runs/${TAG}_merged/config.json" ]; then
  step "$TAG hard-30" python scripts/run_baseline.py --model "runs/${TAG}_merged" \
    --tasks data/tasks_hard_eval.jsonl --no-4bit --out "runs/${TAG}_hard30"
  step "$TAG hard-90" python scripts/run_baseline.py --model "runs/${TAG}_merged" \
    --tasks data/tasks_hard90_eval.jsonl --no-4bit --out "runs/${TAG}_hard90"
fi

step "upload $TAG" python scripts/job_upload.py \
  --run "${TAG}_hard30" --run "${TAG}_hard90" --adapter "runs/$TAG:${TAG}_adapter"

echo "=== $TAG finished: $(date -u +%H:%M:%S) ==="
