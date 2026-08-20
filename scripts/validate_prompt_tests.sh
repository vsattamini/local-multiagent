#!/bin/bash
# Script to validate prompt test results (format + harness)
# Usage: ./scripts/validate_prompt_tests.sh

set -e

echo "=============================================="
echo "VALIDATING PROMPT TEST RESULTS (V3 & V4)"
echo "=============================================="

# This will run the validation script for both folders.
# It uses the existing validate_swebench_results.py which handles
# Docker-based harness validation.

echo "[1/2] Validating V3..."
python scripts/validate_swebench_results.py \
    --results-dir results/prompt_test/v3 \
    --full-validation \
    --workers 4

echo ""
echo "[2/2] Validating V4..."
python scripts/validate_swebench_results.py \
    --results-dir results/prompt_test/v4 \
    --full-validation \
    --workers 4

echo ""
echo "=============================================="
echo "COMPARISON SUMMARY"
echo "=============================================="

V3_FILE="results/prompt_test/v3/experimental/validation_results.json"
V4_FILE="results/prompt_test/v4/experimental/validation_results.json"

python3 -c "
import json
from pathlib import Path

def get_stats(file_path):
    if not Path(file_path).exists():
        return 'Not found'
    with open(file_path) as f:
        data = json.load(f)
    summary = data.get('summary', {})
    total = summary.get('total', 0)
    valid = summary.get('format_valid', 0)
    passed = summary.get('harness_passed', 0)
    return f'Total: {total}, Format Valid: {valid} ({valid/total*100:.1f}%), Harness Passed: {passed}'

print(f'V3 (Dense): {get_stats(\"$V3_FILE\")}')
print(f'V4 (Explicit): {get_stats(\"$V4_FILE\")}')
"
echo "=============================================="
