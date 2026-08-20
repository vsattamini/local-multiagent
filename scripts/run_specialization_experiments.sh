#!/bin/bash
# Run specialization experiments
# Compares: low temp, 5 agents, random control, 7B model

set -e

echo "=============================================="
echo "SPECIALIZATION EXPERIMENTS"
echo "=============================================="
echo ""

# 1. Low Temperature (faster differentiation)
echo "[1/4] Running Low Temperature experiment..."
python scripts/run_experiment.py \
    --config config/exp_low_temp.yaml \
    --seeds 42 \
    --use-full-categories

# 2. More Agents (5 agents)
echo ""
echo "[2/4] Running 5 Agents experiment..."
python scripts/run_experiment.py \
    --config config/exp_5_agents.yaml \
    --seeds 42 \
    --use-full-categories

# 3. Random Control (baseline comparison)
echo ""
echo "[3/4] Running Random Control experiment..."
python scripts/run_experiment.py \
    --config config/exp_random_control.yaml \
    --seeds 42 \
    --use-full-categories

# 4. 7B Model (stronger capability)
echo ""
echo "[4/4] Running 7B Model experiment..."
python scripts/run_experiment.py \
    --config config/exp_7b_model.yaml \
    --seeds 42 \
    --use-full-categories

echo ""
echo "=============================================="
echo "ALL EXPERIMENTS COMPLETE"
echo "=============================================="
echo ""
echo "Results in:"
echo "  - results/exp_low_temp/"
echo "  - results/exp_5_agents/"
echo "  - results/exp_random_control/"
echo "  - results/exp_7b_model/"
echo ""
echo "Compare with baseline:"
echo "  - results/exp2.1_experimental/ (original: temp=0.5, N=3, 1.5B)"
