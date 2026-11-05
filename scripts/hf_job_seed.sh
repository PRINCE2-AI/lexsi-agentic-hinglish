#!/usr/bin/env bash
# One extra SFT seed, measured on both held-out splits.
#
#   SEED=123 bash scripts/hf_job_seed.sh
#   SEED=7 PREFIX=sft4_seed TASKS=data/tasks_train_mix3.jsonl bash scripts/hf_job_seed.sh
#
# Same recipe as the shipped run (2 epochs, r=16, lr 2e-4); only the LoRA init
# and the data order change, which is exactly the variance a single seed hides.
# hard-90 is evaluated and uploaded first so a timeout cannot cost the headline.
set -uo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SEED=${SEED:-123}
PREFIX=${PREFIX:-sft3_seed}
TASKS=${TASKS:-data/tasks_train_mix2.jsonl}
TAG=${PREFIX}${SEED}

step() {
  local label="$1"
  shift
  echo "=== $label: $(date -u +%H:%M:%S) ==="
  local start=$SECONDS
  "$@" || echo "$label FAILED"
  echo "=== $label done in $((SECONDS - start)) s ==="
}

step "train $TAG" python scripts/train_sft.py \
  --tasks "$TASKS" --epochs 2 --seed "$SEED" \
  --out "runs/$TAG" --merge-out "runs/${TAG}_merged"

if [ -f "runs/${TAG}_merged/config.json" ]; then
  step "$TAG hard-90" python scripts/run_baseline.py --model "runs/${TAG}_merged" \
    --tasks data/tasks_hard90_eval.jsonl --no-4bit --out "runs/${TAG}_hard90"
  step "upload $TAG hard-90" python scripts/job_upload.py \
    --run "${TAG}_hard90" --adapter "runs/$TAG:${TAG}_adapter"

  step "$TAG hard-30" python scripts/run_baseline.py --model "runs/${TAG}_merged" \
    --tasks data/tasks_hard_eval.jsonl --no-4bit --out "runs/${TAG}_hard30"
  step "upload $TAG hard-30" python scripts/job_upload.py --run "${TAG}_hard30"
fi

echo "=== $TAG finished: $(date -u +%H:%M:%S) ==="
