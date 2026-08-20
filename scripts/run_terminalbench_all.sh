#!/usr/bin/env bash
# Terminal-Bench floor probe for all three Qwen sizes, SOLO/sequential.
# For each size: start the local OpenAI model server (GPU), wait until ready,
# run `tb terminus` on terminal-bench-core, then stop the server before the next.
set -u
cd "$(dirname "$0")/.."
LOG=results_terminalbench/_driver.log
mkdir -p results_terminalbench
PORT="${PORT:-8000}"

# Guard: 7B server is ~5GB; must not coexist with a swarm run.
if pgrep -f "run_experiment.py" >/dev/null; then
  echo "ABORT: run_experiment.py alive — Terminal-Bench needs the GPU solo." | tee -a "$LOG"; exit 3
fi

echo "START (Terminal-Bench) $(date '+%F %H:%M:%S')" >> "$LOG"

# Optional harness sanity once (oracle, no model) — should resolve 1/1 when unloaded.
echo "== oracle sanity (hello-world) ==" | tee -a "$LOG"
./.venv-tb/bin/tb run --dataset terminal-bench-core==0.1.1 --agent oracle \
    --task-id hello-world --n-concurrent 1 \
    --output-path results_terminalbench/oracle >> "$LOG" 2>&1 || true

for size in 1.5b 3b 7b; do
  echo "== $(date '+%H:%M:%S') serve $size ==" | tee -a "$LOG"
  scripts/tb_serve_model.sh "$size" > "results_terminalbench/server_$size.log" 2>&1 &
  SERVER=$!
  # wait until the server answers /v1/models (up to ~3 min)
  ready=0
  for i in $(seq 1 90); do
    if curl -s "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then ready=1; break; fi
    sleep 2
  done
  if [ "$ready" -ne 1 ]; then
    echo "  server $size NOT ready — skipping" | tee -a "$LOG"; kill "$SERVER" 2>/dev/null; wait "$SERVER" 2>/dev/null; continue
  fi
  echo "  $(date '+%H:%M:%S') run terminus on core ($size)" | tee -a "$LOG"
  OPENAI_API_BASE="http://127.0.0.1:$PORT/v1" OPENAI_BASE_URL="http://127.0.0.1:$PORT/v1" \
  OPENAI_API_KEY="sk-noauth" \
  ./.venv-tb/bin/tb run --dataset terminal-bench-core==0.1.1 --agent terminus \
      --model "openai/qwen2.5-coder" --n-concurrent 1 \
      --global-agent-timeout-sec 600 \
      --output-path "results_terminalbench/$size" >> "results_terminalbench/$size.log" 2>&1
  echo "  $(date '+%H:%M:%S') done $size (rc=$?); stopping server" | tee -a "$LOG"
  kill "$SERVER" 2>/dev/null; wait "$SERVER" 2>/dev/null
  sleep 3
done
echo "ALL DONE (Terminal-Bench) $(date '+%F %H:%M:%S')" | tee -a "$LOG"
