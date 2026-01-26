#!/bin/bash
# Temperature Sweep: Test specialization with temps 0.1 to 1.5
# Fixed: 5 agents, affinity router
# Based on finding: Low temp (0.1) gave S=0.196, high temp (0.5) gave S=0.009

set -e

echo "=============================================="
echo "TEMPERATURE SWEEP: 0.1 to 1.5 (5 agents)"
echo "=============================================="
echo "Goal: Find optimal temperature for specialization"
echo "Hypothesis: Lower temp = more exploitation = more emergence"
echo ""

BASE_CONFIG="config/exp_5_agents.yaml"

for temp in 0.1 0.3 0.5 0.7 0.9 1.1 1.3 1.5; do
    # Convert temp to underscore format for folder name (0.1 -> 0_1)
    temp_folder=$(echo $temp | tr '.' '_')
    
    echo "[temp=$temp] Running experiment..."
    
    python scripts/run_experiment.py \
        --config "$BASE_CONFIG" \
        --n-agents 5 \
        --router-temp $temp \
        --output-dir "results/temp_sweep_${temp_folder}" \
        --seeds 42 \
        --use-full-categories
    
    echo ""
done

echo "=============================================="
echo "TEMPERATURE SWEEP COMPLETE"
echo "=============================================="
echo ""
echo "Results in:"
for temp in 0.1 0.3 0.5 0.7 0.9 1.1 1.3 1.5; do
    temp_folder=$(echo $temp | tr '.' '_')
    echo "  - results/temp_sweep_${temp_folder}/"
done
echo ""
echo "Analyze with:"
echo "  python -c \"import json; from pathlib import Path"
echo "  for t in ['0_1','0_3','0_5','0_7','0_9','1_1','1_3','1_5']:"
echo "      f = Path(f'results/temp_sweep_{t}/seed_42/final_metrics.json')"
echo "      if f.exists():"
echo "          m = json.load(open(f))['metrics']"
echo "          print(f'{t}: S={m[\"specialization_index\"]:.3f}')\""
