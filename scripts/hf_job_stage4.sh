#!/usr/bin/env bash
# Stage 4 — only if sft4 misses the round-3 bar (hard-90 >= 80/90 with asked >= 2/8).
#
# The round-1 adapter scores 15/15 on empty_balance and the abstention rows in mix2
# cost 11 of those tasks. mix4 keeps the specified twins and the contrast pairs but
# drops the ask rows, so the model is walked back to "answer what is answerable"
# without losing the 44 minimal pairs. The safety cost of that is measured, not
# assumed: `asked` is expected to return to 0/8 and is reported as such.
#
#   bash scripts/hf_job_stage4.sh
set -uo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

step() {
  local label="$1"
  shift
  echo "=== $label: $(date -u +%H:%M:%S) ==="
  local start=$SECONDS
  "$@" || echo "$label FAILED"
  echo "=== $label done in $((SECONDS - start)) s ==="
}

step "verify" python scripts/verify_all.py

if [ ! -f runs/sft5_merged/config.json ]; then
  step "train sft5 (mix4, seed 42)" python scripts/train_sft.py \
    --tasks data/tasks_train_mix4.jsonl --epochs 2 --seed 42 \
    --out runs/sft5 --merge-out runs/sft5_merged
fi

[ -f runs/sft5_merged/config.json ] || { echo "no sft5, stopping"; exit 1; }

step "sft5 sanity (3 easy tasks)" python scripts/adapter_sanity.py \
  --merged runs/sft5_merged --limit 3 --out runs/sft5_sanity/summary.json

step "sft5 hard-90" python scripts/run_baseline.py --model runs/sft5_merged \
  --tasks data/tasks_hard90_eval.jsonl --no-4bit --out runs/sft5_hard90
step "sft5 hard-30" python scripts/run_baseline.py --model runs/sft5_merged \
  --tasks data/tasks_hard_eval.jsonl --no-4bit --out runs/sft5_hard30
step "sft5 safety" python scripts/run_baseline.py --model runs/sft5_merged \
  --tasks data/tasks_safety_eval.jsonl --safety --no-4bit --out runs/safety_sft5

step "upload sft5" python scripts/job_upload.py \
  --run sft5_hard90 --run sft5_hard30 --run safety_sft5 \
  --adapter runs/sft5:sft5_adapter

echo "=== stage 4 finished: $(date -u +%H:%M:%S) ==="
