#!/bin/bash
# Quick test: Compare V3 vs V4 prompts
# 20 issues each, n=5 agents, temp=0.1

set -e

echo "=============================================="
echo "PROMPT COMPARISON TEST: V3 vs V4"
echo "=============================================="
echo "Issues: 20"
echo "Agents: 5"
echo "Temperature: 0.1"
echo "Model: 1.5B"
echo "=============================================="

MODEL="models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"

echo ""
echo "[1/2] Running V3 (zero-shot, dense format)..."
python scripts/run_swebench_experiment.py \
    --condition experimental \
    --model-path "$MODEL" \
    --n-agents 5 \
    --n-issues 20 \
    --router-temp 0.1 \
    --prompt-version v3 \
    --output-dir "results/prompt_test/v3"

echo ""
echo "[2/2] Running V4 (zero-shot, step-by-step)..."
python scripts/run_swebench_experiment.py \
    --condition experimental \
    --model-path "$MODEL" \
    --n-agents 5 \
    --n-issues 20 \
    --router-temp 0.1 \
    --prompt-version v4 \
    --output-dir "results/prompt_test/v4"

echo ""
echo "=============================================="
echo "DONE! Compare results:"
echo "  results/prompt_test/v3/results.json"
echo "  results/prompt_test/v4/results.json"
echo "=============================================="
