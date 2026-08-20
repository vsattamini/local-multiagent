#!/bin/bash
# Population Sweep: Test specialization with 3-10 agents
# Answers RQ3: What's the optimal N for emergence?

set -e

echo "=============================================="
echo "POPULATION SWEEP: 3-10 AGENTS"
echo "=============================================="
echo ""

for n in 3 4 5 6 7 8 9 10; do
    echo "[N=$n] Running experiment with $n agents..."
    
    python scripts/run_experiment.py \
        --config config/exp_low_temp.yaml \
        --n-agents $n \
        --output-dir "results/pop_sweep_n${n}" \
        --seeds 42 \
        --use-full-categories
    
    echo ""
done

echo "=============================================="
echo "POPULATION SWEEP COMPLETE"
echo "=============================================="
echo ""
echo "Results in:"
for n in 3 4 5 6 7 8 9 10; do
    echo "  - results/pop_sweep_n${n}/"
done
echo ""
echo "To compare, check final_metrics.json in each folder"
