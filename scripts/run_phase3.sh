#!/usr/bin/env bash
# Phase-3 orchestrator: all fan-out new runs, SEQUENTIAL (one model at a time →
# 7B/het never share the GPU). Gated/backward-compatible; resumable (skips done seeds).
#
# ⚠️ NON-CANONICAL FALLBACK. The driver actually in use is scripts/run_phase3_parallel.py
#    (memory-aware, concurrent). Keep this script in sync with that driver's JOBS list.
#    It refuses to start if the parallel driver is already running, to avoid divergent
#    scheduling / double GPU load.
#   order: RandomRouter controls → greedy → large-N → ENSEMBLES (the decisive runs) →
#          MBPP+ → context-shuffle causal test → Terminal-Bench (last).
set -u
cd "$(dirname "$0")/.."
if pgrep -f "run_phase3_parallel.py" >/dev/null; then
  echo "ABORT: run_phase3_parallel.py is active (the canonical driver). Not starting the"
  echo "       sequential fallback — it would double-schedule the GPU. Use the parallel one."
  exit 1
fi
NVLIB=.venv/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH="$PWD/$NVLIB/cuda_runtime/lib:$PWD/$NVLIB/cublas/lib:$PWD/$NVLIB/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"
PY=./.venv/bin/python
LOG=results_phase3/_orchestrator.log
mkdir -p results_phase3
echo "================ PHASE3 START $(date '+%F %H:%M:%S') ================" | tee -a "$LOG"

# Gate: do not start until rrpersona (and any other swarm runner) is finished.
while pgrep -f "run_experiment.py" >/dev/null; do
  echo "[gate] $(date '+%H:%M:%S') waiting for active runner (rrpersona) to finish..." | tee -a "$LOG"; sleep 30
done
echo "[gate] clear — GPU free, starting phase-3" | tee -a "$LOG"

SEEDS=(42 123 456 789 1011 1213 1415 1617 1819 2021)
run_cfg(){  # $1=config name  $2=n_seeds
  local cfg=$1 n=$2 out="results_phase3/$1" missing=()
  for ((i=0;i<n;i++)); do s=${SEEDS[$i]}
    [ -f "$out/seed_$s/final_metrics.json" ] || { [ -d "$out/seed_$s" ] && rm -rf "$out/seed_$s"; missing+=("$s"); }
  done
  [ ${#missing[@]} -eq 0 ] && { echo "SKIP $cfg (done)" | tee -a "$LOG"; return; }
  # safety: never run while another runner is alive
  while pgrep -f "run_experiment.py" >/dev/null; do sleep 15; done
  echo "== $(date '+%H:%M:%S') START $cfg (n=$n, missing=${missing[*]}) ==" | tee -a "$LOG"
  $PY scripts/run_experiment.py --config "config/$cfg.yaml" --seeds "${missing[@]}" \
      --output-dir "$out" >> "results_phase3/$cfg.log" 2>&1
  echo "== $(date '+%H:%M:%S') END $cfg (rc=$?) ==" | tee -a "$LOG"
}

# 1) cheap no-code: RandomRouter control, greedy, large-N
for c in rand_1.5b rand_3b rand_7b greedy_3b popN_8 popN_12 popN_16; do run_cfg "$c" 10; done
# 2) DECISIVE ENSEMBLES first (computable LR / GLMM). Order matches run_phase3_parallel.py.
#    Single-assignment het_swarm is DROPPED: it cannot compute the differentiation test
#    (each problem seen ≤once/agent → singular interaction) and crashed. het_swarm_ensemble
#    subsumes the interesting part. See audit/het_interaction_prereg.md.
run_cfg ens_3b_n3 10            # 3x3B homogeneous control (weight-only-different from het)
run_cfg ens_3b 10              # 4-agent homogeneous ensemble null
run_cfg het_swarm_ensemble 10  # THE functional-differentiation test (2x1.5B+1x3B, all-assign)
run_cfg ens_7b 5               # clean 7B ensemble null
# 4) MBPP+ headroom
for c in mbpp_1.5b_baseline mbpp_1.5b_lowtemp mbpp_3b_baseline mbpp_3b_lowtemp; do run_cfg "$c" 10; done
run_cfg mbpp_7b_baseline 5
run_cfg mbpp_7b_lowtemp 5

# 5) causal context-shuffle (reads completed multi-seed runs; solo)
echo "== $(date '+%H:%M:%S') context-shuffle ==" | tee -a "$LOG"
$PY scripts/run_context_shuffle.py --src results_multiseed/exp_3b_baseline  --model-size 3b --seeds 42 123 456 789 1011 --out results_context_shuffle >> results_phase3/context_shuffle.log 2>&1
$PY scripts/run_context_shuffle.py --src results_multiseed/exp_3b_low_temp  --model-size 3b --seeds 42 123 456 789 1011 --out results_context_shuffle >> results_phase3/context_shuffle.log 2>&1
$PY scripts/run_context_shuffle.py --src results_multiseed/exp_7b_model     --model-size 7b --seeds 42 123 456 789 1011 --out results_context_shuffle >> results_phase3/context_shuffle.log 2>&1
echo "== $(date '+%H:%M:%S') context-shuffle done ==" | tee -a "$LOG"

# 6) Terminal-Bench floor probe (last; own GPU server, solo)
echo "== $(date '+%H:%M:%S') Terminal-Bench ==" | tee -a "$LOG"
bash scripts/run_terminalbench_all.sh >> results_phase3/terminalbench.log 2>&1
echo "================ PHASE3 ALL DONE $(date '+%F %H:%M:%S') ================" | tee -a "$LOG"
