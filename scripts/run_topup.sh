#!/usr/bin/env bash
# Top-up the confirmatory extras interrupted by the 2026-06-09 suspend crash.
# GPU-gated: waits for the running het_15_7b (run_experiment.py) to exit so the
# GPU is never double-scheduled, then finishes ens_1.5b's missing seeds and the
# rand_1.5b/123 control top-up. Resumable; skips any seed that already has
# final_metrics.json. NEITHER run is load-bearing (not referenced by any chapter
# or audit doc) — this is dataset completeness only.
set -u
cd "$(dirname "$0")/.."
NVLIB=.venv/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH="$PWD/$NVLIB/cuda_runtime/lib:$PWD/$NVLIB/cublas/lib:$PWD/$NVLIB/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"
PY=./.venv/bin/python
LOG=results_phase3/_topup.log
echo "TOPUP waiter START $(date '+%F %H:%M:%S')" >> "$LOG"

# Stay solo: wait until the het_15_7b run (and any other run_experiment) is done.
while pgrep -f "run_experiment.py" >/dev/null; do sleep 60; done
echo "TOPUP starting $(date '+%F %H:%M:%S') — GPU clear" | tee -a "$LOG"

run_cfg(){  # name "seed seed ..."
  local cfg=$1; shift; local out="results_phase3/$cfg" miss=()
  for s in "$@"; do
    [ -f "$out/seed_$s/final_metrics.json" ] || { [ -d "$out/seed_$s" ] && rm -rf "$out/seed_$s"; miss+=("$s"); }
  done
  [ ${#miss[@]} -eq 0 ] && { echo "SKIP $cfg (all complete)" | tee -a "$LOG"; return; }
  while pgrep -f "run_experiment.py" >/dev/null; do sleep 30; done   # stay solo
  echo "== $(date '+%H:%M:%S') START $cfg (missing ${miss[*]}) ==" | tee -a "$LOG"
  $PY scripts/run_experiment.py --config "config/$cfg.yaml" --seeds "${miss[@]}" \
      --output-dir "$out" >> "results_phase3/$cfg.log" 2>&1
  echo "== $(date '+%H:%M:%S') END $cfg (rc=$?) ==" | tee -a "$LOG"
}

run_cfg ens_1.5b 42 123 456 789 1011 1213 1415 1617 1819 2021
run_cfg rand_1.5b 42 123 456 789 1011 1213 1415 1617 1819 2021
echo "TOPUP ALL DONE $(date '+%F %H:%M:%S')" | tee -a "$LOG"
