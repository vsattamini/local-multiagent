#!/usr/bin/env bash
# PARALLEL small-model stream: the 1.5B differentiation mechanisms (n=10).
# Safe to run concurrently with the BIG stream WHILE BIG is on 3B work
# (3B ~2.5GB + 1.5B ~1.5GB + desktop ~1.6GB < 8GB). Do NOT overlap 7B.
# Resumable: skips any (config, seed) whose final_metrics.json exists.
set -u
cd "$(dirname "$0")/.."

NVLIB=.venv/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH="$PWD/$NVLIB/cuda_runtime/lib:$PWD/$NVLIB/cublas/lib:$PWD/$NVLIB/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"
PY=./.venv/bin/python

ALL_SEEDS=(42 123 456 789 1011 1213 1415 1617 1819 2021)
CONFIGS="mech_1_5b_typefilter mech_1_5b_persona"
LOG=results_multiseed/_driver_small.log

mkdir -p results_multiseed
echo "START (SMALL/1.5B) $(date '+%F %H:%M:%S')" >> "$LOG"

for cfg in $CONFIGS; do
  out="results_multiseed/$cfg"
  missing=()
  for s in "${ALL_SEEDS[@]}"; do
    if [ ! -f "$out/seed_$s/final_metrics.json" ]; then
      [ -d "$out/seed_$s" ] && rm -rf "$out/seed_$s"
      missing+=("$s")
    fi
  done
  if [ ${#missing[@]} -eq 0 ]; then
    echo "==== $(date '+%H:%M:%S') SKIP $cfg ====" | tee -a "$LOG"; continue
  fi
  echo "==== $(date '+%H:%M:%S') START $cfg (missing=${missing[*]}) ====" | tee -a "$LOG"
  $PY scripts/run_experiment.py --config "config/$cfg.yaml" --num-tasks 164 \
      --seeds "${missing[@]}" --output-dir "$out" >> "results_multiseed/$cfg.log" 2>&1
  echo "==== $(date '+%H:%M:%S') END   $cfg (rc=$?) ====" | tee -a "$LOG"
done
echo "ALL DONE (SMALL stream) $(date '+%F %H:%M:%S')" | tee -a "$LOG"
