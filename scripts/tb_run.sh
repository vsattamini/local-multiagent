#!/usr/bin/env bash
# Run Terminal-Bench against the local Qwen server (started by tb_serve_model.sh).
# Harness runs from the isolated .venv-tb; model inference is the separate server.
#
# Prereqs: Docker running; scripts/tb_serve_model.sh active on :8000.
#
# Usage:
#   scripts/tb_run.sh oracle hello-world          # harness sanity (no model)
#   scripts/tb_run.sh terminus hello-world        # single-task smoke w/ local model
#   scripts/tb_run.sh terminus                    # full core set (sequential)
set -u
cd "$(dirname "$0")/.."

AGENT="${1:-terminus}"
TASK="${2:-}"        # empty => full dataset
PORT="${PORT:-8000}"

export OPENAI_API_BASE="http://127.0.0.1:${PORT}/v1"
export OPENAI_BASE_URL="http://127.0.0.1:${PORT}/v1"
export OPENAI_API_KEY="sk-noauth"

ARGS=(run --dataset terminal-bench-core==0.1.1 --agent "$AGENT" --n-concurrent 1
      --output-path results_terminalbench)
# oracle/null agents need no model; others target the local server.
if [ "$AGENT" != "oracle" ] && [ "$AGENT" != "null" ]; then
  ARGS+=(--model "openai/qwen2.5-coder")
fi
[ -n "$TASK" ] && ARGS+=(--task-id "$TASK")

echo "tb ${ARGS[*]}"
exec ./.venv-tb/bin/tb "${ARGS[@]}"
