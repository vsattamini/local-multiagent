#!/bin/bash
# Full symmetric sweep for 3B and 7B models
# Matches the existing 1.5B sweep design:
#   temp_sweep: 5 agents, vary temperature (0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5)
#   pop_sweep:  temp 0.1, vary agents (3, 4, 5, 6, 7, 8, 9, 10)

set -e
LOG_DIR="results/sweep_logs"
mkdir -p "$LOG_DIR"

run_exp() {
    local config=$1
    local outdir=$2
    local temp=$3
    local agents=$4
    local logname=$5

    if [ -f "${outdir}/seed_42/final_metrics.json" ]; then
        echo "[SKIP] ${logname} already done"
        return
    fi

    echo "[RUN ] ${logname} (temp=${temp}, agents=${agents})"
    python scripts/run_experiment.py \
        --config "$config" \
        --router-temp "$temp" \
        --n-agents "$agents" \
        --output-dir "$outdir" \
        --seeds 42 \
        > "${LOG_DIR}/${logname}.log" 2>&1
}

# ============== 7B TEMPERATURE SWEEP (5 agents) ==============
for t in 0.1 0.3 0.5 0.7 0.9 1.1 1.3 1.5; do
    safe_t=$(echo "$t" | tr '.' '_')
    run_exp config/exp_7b_model.yaml "results/sweep_7b_temp_${safe_t}" "$t" 5 "7b_temp_${safe_t}"
done

# ============== 7B POPULATION SWEEP (temp 0.1) ==============
for n in 3 4 5 6 7 8 9 10; do
    run_exp config/exp_7b_model.yaml "results/sweep_7b_pop_n${n}" 0.1 "$n" "7b_pop_n${n}"
done

# ============== 3B TEMPERATURE SWEEP (5 agents) ==============
for t in 0.1 0.3 0.5 0.7 0.9 1.1 1.3 1.5; do
    safe_t=$(echo "$t" | tr '.' '_')
    run_exp config/exp_3b_model.yaml "results/sweep_3b_temp_${safe_t}" "$t" 5 "3b_temp_${safe_t}"
done

# ============== 3B POPULATION SWEEP (temp 0.1) ==============
for n in 3 4 5 6 7 8 9 10; do
    run_exp config/exp_3b_model.yaml "results/sweep_3b_pop_n${n}" 0.1 "$n" "3b_pop_n${n}"
done

echo "ALL SWEEPS COMPLETE"
