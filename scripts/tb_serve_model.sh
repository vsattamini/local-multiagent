#!/usr/bin/env bash
# Serve a local Qwen2.5-Coder GGUF as an OpenAI-compatible endpoint for
# Terminal-Bench (LiteLLM -> openai/<alias> -> http://127.0.0.1:8000/v1).
# Uses the MAIN .venv (which has CUDA llama-cpp-python). The Terminal-Bench
# harness itself runs from the separate .venv-tb.
#
# Usage:
#   scripts/tb_serve_model.sh 1.5b            # GPU (default)
#   scripts/tb_serve_model.sh 7b
#   scripts/tb_serve_model.sh 3b 0            # 2nd arg = n_gpu_layers (0 = CPU, for
#                                             #   parallel-safe smoke while GPU is busy)
set -u
cd "$(dirname "$0")/.."

SIZE="${1:-1.5b}"
NGL="${2:--1}"     # -1 = full GPU offload; 0 = CPU only
PORT="${PORT:-8000}"
MODEL="models/qwen2.5-coder-${SIZE}-instruct-q4_k_m.gguf"
[ -f "$MODEL" ] || { echo "model not found: $MODEL"; exit 1; }

NVLIB=.venv/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH="$PWD/$NVLIB/cuda_runtime/lib:$PWD/$NVLIB/cublas/lib:$PWD/$NVLIB/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"

echo "Serving $MODEL  (n_gpu_layers=$NGL) on http://127.0.0.1:$PORT/v1  alias=qwen2.5-coder"
exec ./.venv/bin/python -m llama_cpp.server \
    --model "$MODEL" \
    --model_alias "qwen2.5-coder" \
    --host 127.0.0.1 --port "$PORT" \
    --n_ctx 8192 \
    --n_gpu_layers "$NGL"
