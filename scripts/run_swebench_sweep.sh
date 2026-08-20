#!/bin/bash
# SWE-bench Combination Sweep
# Tests combinations of temp (0.1-0.3) and agents (5-15, step 2) for both 1.5B and 7B models
#
# Usage:
#   ./scripts/run_swebench_sweep.sh           # Use default prompt (v1)
#   PROMPT_VERSION=v3 ./scripts/run_swebench_sweep.sh  # Use v3 prompt
#
# Prompt Versions:
#   v1 - Simple prompt (original, 6 unique issues solved)
#   v2 - Few-shot with bug (worse results)
#   v3 - Fixed few-shot
#
# Total experiments: 3 temps × 6 agent counts × 2 models = 36 experiments

set -e

# Configuration - customize these
PROMPT_VERSION="${PROMPT_VERSION:-v1}"  # Default to v1 (best performing)
OUTPUT_BASE="results/swebench_sweep_${PROMPT_VERSION}"

echo "=============================================="
echo "SWE-BENCH COMBINATION SWEEP"
echo "=============================================="
echo "Prompt Version: ${PROMPT_VERSION}"
echo "Temps: 0.1, 0.2, 0.3"
echo "Agents: 5, 7, 9, 11, 13, 15"
echo "Models: 1.5B, 7B"
echo "Output: ${OUTPUT_BASE}"
echo "=============================================="
echo ""


MODEL_1_5B="models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
MODEL_7B="models/qwen2.5-coder-7b-instruct-q4_k_m.gguf"

# Function to run a single combination
run_combination() {
    local model_path=$1
    local model_name=$2
    local temp=$3
    local n_agents=$4
    
    temp_str=$(echo $temp | tr '.' '_')
    output_dir="${OUTPUT_BASE}/${model_name}/temp${temp_str}_n${n_agents}"
    
    echo "[${model_name}] temp=${temp}, n_agents=${n_agents}, prompt=${PROMPT_VERSION} -> ${output_dir}"
    
    python scripts/run_swebench_experiment.py \
        --condition experimental \
        --model-path "$model_path" \
        --n-agents $n_agents \
        --router-temp $temp \
        --prompt-version "$PROMPT_VERSION" \
        --output-dir "$output_dir"
    
    echo ""
}

# Run 1.5B model combinations
echo "=========================================="
echo "MODEL: 1.5B (Qwen2.5-Coder-1.5B)"
echo "=========================================="

if [ -f "$MODEL_1_5B" ]; then
    for temp in 0.1 0.2 0.3; do
        for n_agents in 5 7 9 11 13 15; do
            run_combination "$MODEL_1_5B" "1.5b" $temp $n_agents
        done
    done
else
    echo "SKIPPING 1.5B - Model file not found: $MODEL_1_5B"
fi

echo ""
echo "=========================================="
echo "MODEL: 7B (Qwen2.5-Coder-7B)"
echo "=========================================="

if [ -f "$MODEL_7B" ]; then
    for temp in 0.1 0.2 0.3; do
        for n_agents in 5 7 9 11 13 15; do
            run_combination "$MODEL_7B" "7b" $temp $n_agents
        done
    done
else
    echo "SKIPPING 7B - Model file not found: $MODEL_7B"
fi

echo ""
echo "=============================================="
echo "SWEEP COMPLETE"
echo "=============================================="
echo ""
echo "Results organized as:"
echo "  results/swebench_sweep/"
echo "  ├── 1.5b/"
echo "  │   ├── temp0_1_n5/"
echo "  │   ├── temp0_1_n7/"
echo "  │   └── ..."
echo "  └── 7b/"
echo "      ├── temp0_1_n5/"
echo "      └── ..."
echo ""
echo "To analyze results:"
echo "  python scripts/analyze_swebench_sweep.py"
