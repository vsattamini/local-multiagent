#!/usr/bin/env bash
# Multi-seed re-run of the core experiments for variance / power.
# RESUMABLE: skips any (config, seed) whose final_metrics.json already exists.
# n=20 on the 5 KEY conditions (central claims), n=10 on the 3 REST conditions.
# Generation is seeded (reproducible) at temperature 0.2. Output: results_multiseed/.
set -u
cd "$(dirname "$0")/.."

NVLIB=.venv/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH="$PWD/$NVLIB/cuda_runtime/lib:$PWD/$NVLIB/cublas/lib:$PWD/$NVLIB/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"
PY=./.venv/bin/python

# 20 fixed seeds (first 5 match the initial run).
ALL_SEEDS=(42 123 456 789 1011 1213 1415 1617 1819 2021 2223 2425 2627 2829 3031 3233 3435 3637 3839 4041)

# KEY conditions get all 20; REST get first 10. KEY ordered fast->slow so the
# central-claim power accrues early.
KEY="exp2.1_experimental exp_low_temp exp_3b_baseline exp_3b_low_temp exp_7b_model"
REST="exp_5_agents exp_3b_model exp_3b_5_agents"

mkdir -p results_multiseed
echo "START $(date '+%F %H:%M:%S') n_key=20 n_rest=10" >> results_multiseed/_driver.log

run_config () {
  local cfg="$1"; local n="$2"
  local out="results_multiseed/$cfg"
  # collect missing seeds
  local missing=()
  for ((i=0; i<n; i++)); do
    local s="${ALL_SEEDS[$i]}"
    if [ ! -f "$out/seed_$s/final_metrics.json" ]; then
      # drop any partial dir before re-running
      [ -d "$out/seed_$s" ] && rm -rf "$out/seed_$s"
      missing+=("$s")
    fi
  done
  if [ ${#missing[@]} -eq 0 ]; then
    echo "==== $(date '+%H:%M:%S') SKIP $cfg (all $n seeds done) ====" | tee -a results_multiseed/_driver.log
    return 0
  fi
  echo "==== $(date '+%H:%M:%S') START $cfg (n=$n, missing=${missing[*]}) ====" | tee -a results_multiseed/_driver.log
  $PY scripts/run_experiment.py \
      --config "config/$cfg.yaml" \
      --num-tasks 164 \
      --seeds "${missing[@]}" \
      --output-dir "$out" \
      >> "results_multiseed/$cfg.log" 2>&1
  echo "==== $(date '+%H:%M:%S') END   $cfg (rc=$?) ====" | tee -a results_multiseed/_driver.log
}

# --- Phase 2 (added 2026-06-06): full 7B coverage + differentiation mechanisms ---
# GPU note: only ~3.9GB free alongside the desktop. 3B (~2.5GB) + 1.5B (~1.5GB)
# co-reside safely; 7B (~5GB) must run SOLO. So this BIG stream runs the 3B
# mechanisms FIRST (overlapping the parallel 1.5B small stream), then 7B LAST
# (solo). The 1.5B mechanisms live in scripts/run_multiseed_small.sh.
MECH3B="mech_3b_typefilter mech_3b_persona"

for cfg in $KEY;    do run_config "$cfg" 20; done
for cfg in $REST;   do run_config "$cfg" 10; done
for cfg in $MECH3B; do run_config "$cfg" 10; done   # 3B mechanisms (safe to overlap 1.5B stream)

# 7B coverage is intentionally NOT here. The 7B models are large (~5GB) and must
# run SOLO (never in parallel with another run). They are handled by
# scripts/run_multiseed_7b.sh, which the orchestrator (scripts/run_all.sh) starts
# only AFTER both parallel streams (this 3B stream + the 1.5B stream) finish.
echo "ALL DONE (BIG/3B stream) $(date '+%F %H:%M:%S')" | tee -a results_multiseed/_driver.log
