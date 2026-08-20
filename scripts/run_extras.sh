#!/usr/bin/env bash
# Extra robustness ensembles, queued to run AFTER the main parallel driver finishes
# (waits for run_phase3_parallel.py to exit, so the running job is never disturbed
# and the GPU is never double-scheduled). Resumable; solo. ens_1.5b first (cheap,
# ~1.5GB), then het_15_7b ([1.5B,7B] ~6.5GB, solo).
set -u
cd "$(dirname "$0")/.."
NVLIB=.venv/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH="$PWD/$NVLIB/cuda_runtime/lib:$PWD/$NVLIB/cublas/lib:$PWD/$NVLIB/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"
PY=./.venv/bin/python
LOG=results_phase3/_extras.log
mkdir -p results_phase3
echo "EXTRAS waiter START $(date '+%F %H:%M:%S')" >> "$LOG"

# Wait until the canonical parallel driver is finished AND no runner is active.
while pgrep -f "run_phase3_parallel.py" >/dev/null || pgrep -f "run_experiment.py" >/dev/null; do
  sleep 60
done
echo "EXTRAS starting $(date '+%F %H:%M:%S') — main queue clear" | tee -a "$LOG"

SEEDS=(42 123 456 789 1011 1213 1415 1617 1819 2021)
run_cfg(){  # name n
  local cfg=$1 n=$2 out="results_phase3/$1" miss=()
  for ((i=0;i<n;i++)); do s=${SEEDS[$i]}
    [ -f "$out/seed_$s/final_metrics.json" ] || { [ -d "$out/seed_$s" ] && rm -rf "$out/seed_$s"; miss+=("$s"); }
  done
  [ ${#miss[@]} -eq 0 ] && { echo "SKIP $cfg" | tee -a "$LOG"; return; }
  while pgrep -f "run_experiment.py" >/dev/null; do sleep 30; done   # stay solo
  echo "== $(date '+%H:%M:%S') START $cfg (missing ${miss[*]}) ==" | tee -a "$LOG"
  $PY scripts/run_experiment.py --config "config/$cfg.yaml" --seeds "${miss[@]}" \
      --output-dir "$out" >> "results_phase3/$cfg.log" 2>&1
  echo "== $(date '+%H:%M:%S') END $cfg (rc=$?) ==" | tee -a "$LOG"
}
run_cfg ens_3b_n3 10   # top up the 2 control seeds that aborted (rc=-6) during crash-recovery
run_cfg ens_1.5b 10
run_cfg het_15_7b 10
echo "EXTRAS ALL DONE $(date '+%F %H:%M:%S')" | tee -a "$LOG"
