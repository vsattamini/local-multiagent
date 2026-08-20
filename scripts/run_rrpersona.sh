#!/usr/bin/env bash
# Positive-control: round-robin (balanced) routing + type-specialist personas.
# Balanced exposure keeps the LR test computable; personas supply the per-type
# skill signal if the model can use them. Solo/sequential, resumable, n=10.
set -u
cd "$(dirname "$0")/.."
# guard: solo only
if pgrep -f "run_experiment.py" >/dev/null; then
  echo "ABORT: a run_experiment.py is already alive — run solo."; exit 3
fi
NVLIB=.venv/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH="$PWD/$NVLIB/cuda_runtime/lib:$PWD/$NVLIB/cublas/lib:$PWD/$NVLIB/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"
PY=./.venv/bin/python
SEEDS=(42 123 456 789 1011 1213 1415 1617 1819 2021)
LOG=results_multiseed/_driver_rrpersona.log
mkdir -p results_multiseed
echo "START (rrpersona) $(date '+%F %H:%M:%S')" >> "$LOG"
for cfg in mech_1_5b_rrpersona mech_3b_rrpersona mech_7b_rrpersona; do
  out="results_multiseed/$cfg"; missing=()
  for s in "${SEEDS[@]}"; do
    [ -f "$out/seed_$s/final_metrics.json" ] || { [ -d "$out/seed_$s" ] && rm -rf "$out/seed_$s"; missing+=("$s"); }
  done
  [ ${#missing[@]} -eq 0 ] && { echo "SKIP $cfg" | tee -a "$LOG"; continue; }
  echo "==== $(date '+%H:%M:%S') START $cfg (missing=${missing[*]}) ====" | tee -a "$LOG"
  $PY scripts/run_experiment.py --config "config/$cfg.yaml" --num-tasks 164 \
      --seeds "${missing[@]}" --output-dir "$out" >> "results_multiseed/$cfg.log" 2>&1
  echo "==== $(date '+%H:%M:%S') END $cfg (rc=$?) ====" | tee -a "$LOG"
done
echo "ALL DONE (rrpersona) $(date '+%F %H:%M:%S')" | tee -a "$LOG"
