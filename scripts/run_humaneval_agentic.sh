#!/usr/bin/env bash
# Agentic self-debug HumanEval (single-model, iterate-with-test-feedback) for all
# three Qwen sizes. SEQUENTIAL — run only after the swarm sweep frees the GPU.
# Resumable: skips any (size, seed) whose summary.json already exists.
set -u
cd "$(dirname "$0")/.."

NVLIB=.venv/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH="$PWD/$NVLIB/cuda_runtime/lib:$PWD/$NVLIB/cublas/lib:$PWD/$NVLIB/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"
PY=./.venv/bin/python

SEEDS="42 123 456 789 1011"
MAXATT="${MAXATT:-5}"
LOG=results_humaneval_agentic/_driver.log
mkdir -p results_humaneval_agentic
echo "START $(date '+%F %H:%M:%S') seeds=[$SEEDS] max_attempts=$MAXATT" >> "$LOG"

for size in 1.5b 3b 7b; do
  echo "==== $(date '+%H:%M:%S') START $size ====" | tee -a "$LOG"
  $PY scripts/run_humaneval_agentic.py --model-size "$size" --seeds $SEEDS \
      --max-attempts "$MAXATT" --output-dir results_humaneval_agentic \
      >> "results_humaneval_agentic/$size.log" 2>&1
  echo "==== $(date '+%H:%M:%S') END   $size (rc=$?) ====" | tee -a "$LOG"
done
echo "ALL DONE $(date '+%F %H:%M:%S')" | tee -a "$LOG"
