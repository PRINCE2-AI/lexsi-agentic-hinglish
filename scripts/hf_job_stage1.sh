#!/usr/bin/env bash
# Stage 1 on a rented GPU (HF Jobs / any single-GPU box):
# train sft3 on the abstention mix, then measure it on every frozen split.
#
#   bash scripts/hf_job_stage1.sh            # everything except the base-90 row
#   BASE90=1 bash scripts/hf_job_stage1.sh   # also the prompt-only row on hard-90 (~2 h more)
#
# Requires cwd = repo root and HF_TOKEN in the environment. Every step is
# independent: a failing eval prints FAILED and the rest still run, so a partial
# night still uploads the numbers it did produce.
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

if [ ! -f runs/sft3_merged/config.json ]; then
  step "train sft3 (seed 42)" python scripts/train_sft.py \
    --tasks data/tasks_train_mix2.jsonl --epochs 2 --seed 42 \
    --out runs/sft3 --merge-out runs/sft3_merged
fi

if [ -f runs/sft3_merged/config.json ]; then
  step "sft3 hard-30" python scripts/run_baseline.py --model runs/sft3_merged \
    --tasks data/tasks_hard_eval.jsonl --no-4bit --out runs/sft3_hard30
  step "sft3 eval-24" python scripts/run_baseline.py --model runs/sft3_merged \
    --tasks data/tasks_eval.jsonl --no-4bit --out runs/sft3_eval
  step "sft3 safety" python scripts/run_baseline.py --model runs/sft3_merged \
    --tasks data/tasks_safety_eval.jsonl --safety --no-4bit --out runs/safety_sft3
  step "sft3 safety + hardened prompt" python scripts/run_baseline.py --model runs/sft3_merged \
    --tasks data/tasks_safety_eval.jsonl --safety --hardened-prompt --no-4bit \
    --out runs/safety_sft3_hardened
  step "sft3 hard-90" python scripts/run_baseline.py --model runs/sft3_merged \
    --tasks data/tasks_hard90_eval.jsonl --no-4bit --out runs/sft3_hard90
fi

step "upload sft3" python scripts/job_upload.py \
  --run sft3_hard30 --run sft3_eval --run sft3_hard90 \
  --run safety_sft3 --run safety_sft3_hardened --adapter runs/sft3:sft3_adapter

# The frozen adapter on the new split, no retraining: merge from the Hub, score it.
if [ ! -f /tmp/sft2_merged/config.json ]; then
  step "merge sft2 from the Hub" python - <<'PY'
import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = "CohereLabs/tiny-aya-global"
repo = snapshot_download("primce/tiny-aya-hinglish-lora", allow_patterns=["sft2_adapter/*"],
                         local_dir="/tmp/sft2_repo")
tokenizer = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(model, repo, subfolder="sft2_adapter")
merged = model.merge_and_unload()
merged.save_pretrained("/tmp/sft2_merged", safe_serialization=True)
tokenizer.save_pretrained("/tmp/sft2_merged")
print("merged -> /tmp/sft2_merged")
PY
fi

if [ -f /tmp/sft2_merged/config.json ]; then
  step "sft2 hard-90" python scripts/run_baseline.py --model /tmp/sft2_merged \
    --tasks data/tasks_hard90_eval.jsonl --no-4bit --out runs/sft2_hard90
  step "upload sft2 hard-90" python scripts/job_upload.py --run sft2_hard90
fi

if [ "${BASE90:-0}" = "1" ]; then
  step "base hard-90" python scripts/run_baseline.py --model "$MODEL" \
    --tasks data/tasks_hard90_eval.jsonl --no-4bit --out runs/base_hard90
  step "upload base hard-90" python scripts/job_upload.py --run base_hard90
fi

echo "=== stage 1 finished: $(date -u +%H:%M:%S) ==="
