#!/usr/bin/env bash
# Stage 4 — run only if sft4 (mix3) misses the round-3 bar: hard-90 >= 80/90 with asked >= 2/8.
#
# Diagnosis behind this arm. On hard-90, `sft3` (mix2) answers 4/4 empty_balance rows whose
# wording appears in training and 0/11 whose wording it has never seen. The wrong answers are
# not silence: they emit, verbatim, the answer of the `ambiguous_share` abstention row
# ("Wo teesra banda thali ka hissa de raha hai ya sirf aaya tha? Batao, phir split karunga.")
# on questions whose ledger is simply empty. `sft2` (mix) answers 11/11 of those same unseen
# rows, so the base ability is there and the abstention rows are what removed it: mix2 repeats
# every ask row 3x, so 96 of 312 rows (31%) are "ask something", and the phrase generalises
# across families.
#
# mix5 tests that causally instead of guessing: the same contrast pairs as mix3 at 3 copies,
# but abstention back to one copy each (96 -> 32 rows, 31% -> 10%). mix4 is kept as a last
# resort but is *not* the default here, because dropping the ask rows entirely is expected to
# put `asked` back at 0/8 and so fail the bar on its own.
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
  step "train sft5 (mix5, seed 42)" python scripts/train_sft.py \
    --tasks data/tasks_train_mix5.jsonl --epochs 2 --seed 42 \
    --out runs/sft5 --merge-out runs/sft5_merged
fi

[ -f runs/sft5_merged/config.json ] || { echo "no sft5, stopping"; exit 1; }

step "sft5 sanity (3 easy tasks)" python scripts/adapter_sanity.py \
  --merged runs/sft5_merged --limit 3 --out runs/sft5_sanity/summary.json

step "sft5 hard-90" python scripts/run_baseline.py --model runs/sft5_merged \
  --tasks data/tasks_hard90_eval.jsonl --no-4bit --out runs/sft5_hard90
step "sft5 hard-30" python scripts/run_baseline.py --model runs/sft5_merged \
  --tasks data/tasks_hard_eval.jsonl --no-4bit --out runs/sft5_hard30
step "sft5 eval-24" python scripts/run_baseline.py --model runs/sft5_merged \
  --tasks data/tasks_eval.jsonl --no-4bit --out runs/sft5_eval
step "sft5 safety" python scripts/run_baseline.py --model runs/sft5_merged \
  --tasks data/tasks_safety_eval.jsonl --safety --no-4bit --out runs/safety_sft5
step "sft5 safety hardened" python scripts/run_baseline.py --model runs/sft5_merged \
  --tasks data/tasks_safety_eval.jsonl --safety --hardened-prompt --no-4bit \
  --out runs/safety_sft5_hardened

step "upload sft5" python scripts/job_upload.py \
  --run sft5_hard90 --run sft5_hard30 --run sft5_eval --run safety_sft5 \
  --run safety_sft5_hardened --run sft5_sanity \
  --adapter runs/sft5:sft5_adapter

echo "=== stage 4 finished: $(date -u +%H:%M:%S) ==="
