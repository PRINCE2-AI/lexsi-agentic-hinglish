#!/usr/bin/env bash
# GRPO on top of the adapter that ships, with a group size that can carry a gradient.
#
#   SHIP_SUB=sft5_adapter NAME=grpo5 TASKS=data/tasks_train_grpo.jsonl \
#     bash scripts/hf_job_grpo4.sh
#
# Why this exists. The first GRPO run (`grpo3`, 40 steps on sft3) used `--generations 2`,
# and 80-95% of its groups came back with identical rewards (`frac_reward_zero_std` 0.8),
# so most steps had no advantage to learn from. The mixed reward also sat at 2.1-2.5 of a
# 9-point scale because the easy rows dominate a uniform sample. This run fixes both:
# 4 generations per prompt, and a task file with the easy rows removed so every group is
# drawn from the families the SFT still gets wrong.
#
# The base is read from the Hub (so this can run next to the SFT job that produced it) and
# the result is a LoRA on top of the merged adapter, evaluated on all four frozen splits.
set -uo pipefail

SHIP_SUB="${SHIP_SUB:?set SHIP_SUB to the Hub adapter subfolder, e.g. sft5_adapter}"
NAME="${NAME:?set NAME to the run prefix, e.g. grpo5}"
TASKS="${TASKS:-data/tasks_train_grpo.jsonl}"
STEPS="${GRPO_STEPS:-60}"
GENERATIONS="${GENERATIONS:-4}"
MERGED="/tmp/${NAME}_base_merged"
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

if [ ! -f "$MERGED/config.json" ]; then
  step "merge ${SHIP_SUB} from the Hub" python - <<PY
import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = "CohereLabs/tiny-aya-global"
repo = snapshot_download("primce/tiny-aya-hinglish-lora", allow_patterns=["${SHIP_SUB}/*"],
                         local_dir="/tmp/${NAME}_repo")
tokenizer = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(model, repo, subfolder="${SHIP_SUB}")
merged = model.merge_and_unload()
merged.save_pretrained("${MERGED}", safe_serialization=True)
tokenizer.save_pretrained("${MERGED}")
print("merged -> ${MERGED}")
PY
fi

[ -f "$MERGED/config.json" ] || { echo "no merged base, stopping"; exit 1; }

if [ ! -d "runs/${NAME}" ]; then
  step "grpo ${STEPS} steps" python scripts/train_grpo.py --model "$MERGED" \
    --tasks "$TASKS" --max-steps "$STEPS" --generations "$GENERATIONS" --lr 3e-6 \
    --efficiency --out "runs/${NAME}"
fi

[ -d "runs/${NAME}" ] || { echo "no adapter, stopping"; exit 1; }

step "${NAME} hard-90" python scripts/run_baseline.py --model "$MERGED" --adapter "runs/${NAME}" \
  --tasks data/tasks_hard90_eval.jsonl --no-4bit --out "runs/${NAME}_hard90"
step "${NAME} hard-30" python scripts/run_baseline.py --model "$MERGED" --adapter "runs/${NAME}" \
  --tasks data/tasks_hard_eval.jsonl --no-4bit --out "runs/${NAME}_hard30"
step "${NAME} eval-24" python scripts/run_baseline.py --model "$MERGED" --adapter "runs/${NAME}" \
  --tasks data/tasks_eval.jsonl --no-4bit --out "runs/${NAME}_eval"
step "${NAME} safety" python scripts/run_baseline.py --model "$MERGED" --adapter "runs/${NAME}" \
  --tasks data/tasks_safety_eval.jsonl --safety --no-4bit --out "runs/safety_${NAME}"

step "upload ${NAME}" python scripts/job_upload.py \
  --run "${NAME}_hard90" --run "${NAME}_hard30" --run "${NAME}_eval" --run "safety_${NAME}" \
  --adapter "runs/${NAME}:${NAME}_adapter"

echo "=== ${NAME} finished: $(date -u +%H:%M:%S) ==="
