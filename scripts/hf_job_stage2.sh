#!/usr/bin/env bash
# Stage 2 on a rented GPU: GRPO on top of sft3 with a reward that can see abstention.
#
#   bash scripts/hf_job_stage2.sh
#
# Requires cwd = repo root and HF_TOKEN in the environment. Reads the sft3 adapter
# from the Hub, so stage 1 must have uploaded it.
set -uo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
STEPS="${GRPO_STEPS:-40}"
MERGED=/tmp/sft3_merged

step() {
  local label="$1"
  shift
  echo "=== $label: $(date -u +%H:%M:%S) ==="
  local start=$SECONDS
  "$@" || echo "$label FAILED"
  echo "=== $label done in $((SECONDS - start)) s ==="
}

step "verify" python scripts/verify_all.py

if [ ! -f "$MERGED/config.json" ]; then
  step "merge sft3 from the Hub" python - <<'PY'
import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = "CohereLabs/tiny-aya-global"
repo = snapshot_download("primce/tiny-aya-hinglish-lora", allow_patterns=["sft3_adapter/*"],
                         local_dir="/tmp/sft3_repo")
tokenizer = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(model, repo, subfolder="sft3_adapter")
merged = model.merge_and_unload()
merged.save_pretrained("/tmp/sft3_merged", safe_serialization=True)
tokenizer.save_pretrained("/tmp/sft3_merged")
print("merged -> /tmp/sft3_merged")
PY
fi

if [ -f "$MERGED/config.json" ] && [ ! -d runs/grpo3 ]; then
  step "grpo $STEPS steps" python scripts/train_grpo.py --model "$MERGED" \
    --tasks data/tasks_train_mix2.jsonl --max-steps "$STEPS" --generations 2 --lr 3e-6 \
    --efficiency --out runs/grpo3
fi

if [ -d runs/grpo3 ]; then
  step "grpo safety" python scripts/run_baseline.py --model "$MERGED" --adapter runs/grpo3 \
    --tasks data/tasks_safety_eval.jsonl --safety --no-4bit --out runs/safety_grpo3
  step "grpo hard-30" python scripts/run_baseline.py --model "$MERGED" --adapter runs/grpo3 \
    --tasks data/tasks_hard_eval.jsonl --no-4bit --out runs/grpo3_hard30
  step "grpo hard-90" python scripts/run_baseline.py --model "$MERGED" --adapter runs/grpo3 \
    --tasks data/tasks_hard90_eval.jsonl --no-4bit --out runs/grpo3_hard90
  step "upload grpo" python scripts/job_upload.py \
    --run safety_grpo3 --run grpo3_hard30 --run grpo3_hard90 --adapter runs/grpo3:grpo3_adapter
fi

echo "=== stage 2 finished: $(date -u +%H:%M:%S) ==="
