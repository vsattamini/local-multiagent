#!/usr/bin/env bash
# 7B coverage, run STRICTLY SOLO (7B ~5GB must never share the 8GB GPU).
# Hard guard: aborts if any other run_experiment.py is alive. Resumable.
set -u
cd "$(dirname "$0")/.."

# --- GUARD: refuse to run in parallel with anything else ---
others=$(pgrep -f "run_experiment.py" | grep -v "^$$\$" | wc -l)
if [ "$others" -gt 0 ]; then
  echo "ABORT: $others other run_experiment.py process(es) alive — 7B must run solo." \
    | tee -a results_multiseed/_driver_7b.log
  pgrep -af "run_experiment.py" | grep -v pgrep | tee -a results_multiseed/_driver_7b.log
  exit 3
fi

NVLIB=.venv/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH="$PWD/$NVLIB/cuda_runtime/lib:$PWD/$NVLIB/cublas/lib:$PWD/$NVLIB/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"
PY=./.venv/bin/python

ALL_SEEDS=(42 123 456 789 1011 1213 1415 1617 1819 2021 2223 2425 2627 2829 3031 3233 3435 3637 3839 4041)
CONFIGS="exp_7b_baseline exp_7b_low_temp exp_7b_5_agents"
LOG=results_multiseed/_driver_7b.log
mkdir -p results_multiseed
echo "START (7B SOLO) $(date '+%F %H:%M:%S')" >> "$LOG"

for cfg in $CONFIGS; do
  out="results_multiseed/$cfg"; missing=()
  for s in "${ALL_SEEDS[@]}"; do
    if [ ! -f "$out/seed_$s/final_metrics.json" ]; then
      [ -d "$out/seed_$s" ] && rm -rf "$out/seed_$s"
      missing+=("$s")
    fi
  done
  if [ ${#missing[@]} -eq 0 ]; then echo "==== $(date '+%H:%M:%S') SKIP $cfg ====" | tee -a "$LOG"; continue; fi
  echo "==== $(date '+%H:%M:%S') START $cfg (missing=${missing[*]}) ====" | tee -a "$LOG"
  $PY scripts/run_experiment.py --config "config/$cfg.yaml" --num-tasks 164 \
      --seeds "${missing[@]}" --output-dir "$out" >> "results_multiseed/$cfg.log" 2>&1
  echo "==== $(date '+%H:%M:%S') END   $cfg (rc=$?) ====" | tee -a "$LOG"
done
echo "ALL DONE (7B SOLO) $(date '+%F %H:%M:%S')" | tee -a "$LOG"
