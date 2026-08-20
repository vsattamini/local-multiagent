# Terminal-Bench — Integration Plan & Status

**Goal:** probe whether the local Qwen2.5-Coder SLMs can do *realistic, multi-step
terminal tasks* (vs. single-shot HumanEval), to (a) establish a capability floor on
hard agentic work and (b) test the "more headroom → more room for differentiation"
hypothesis raised by the multi-seed audit.

## Framing (important for the thesis)
Terminal-Bench is **agentic & sequential** (one tmux session per task, multi-step
command loop, pytest-validated). It does NOT map onto the swarm's *task-type
routing* paradigm. So treat it as a **capability/headroom probe**, not a
specialization experiment:
- First: single-model baseline (built-in `terminus` agent + local Qwen) → floor.
- Later (optional): a custom 2-role agent (planner + executor) as a minimal
  multi-agent variant — see stub `src/tb_agent/local_swarm_agent.py`.

Expectation (from the official leaderboard): top ≈ 64.5%; smallest *open* models on
the board are 17–32B at ~15%; lowest published 5.7%. **No sub-10B model is on the
board** → Qwen 1.5B/3B/7B will almost certainly score **single digits / ~0** on
`terminal-bench-core`. That is itself a reportable result.

## Environment (isolated, no GPU contention with the sweep)
- Harness: **`.venv-tb`** (separate venv — TB pulls litellm/pydantic/docker; must NOT
  perturb the sweep's `.venv`). CLI: `./.venv-tb/bin/tb`. `uv` 0.9.26 is also present.
- Model server: **`.venv`** (CUDA llama-cpp-python) via `scripts/tb_serve_model.sh`,
  exposing OpenAI-compatible `http://127.0.0.1:8000/v1`, model alias `qwen2.5-coder`.
- Harness = CPU/Docker only (no GPU). Only the model server uses the GPU → run the
  server **after the sweep finishes** (or with `n_gpu_layers=0` for a CPU smoke).

## Commands
```bash
# 0) (one-time) confirm harness + Docker, NO model needed:
scripts/tb_run.sh oracle hello-world          # oracle runs the reference solution

# 1) start the model server (GPU) — after the sweep frees the GPU:
scripts/tb_serve_model.sh 1.5b                # or 3b / 7b ; add a 2nd arg 0 for CPU

# 2) single-task smoke with the local model:
scripts/tb_run.sh terminus hello-world        # falls back to `naive` if terminus over-prompts the SLM

# 3) full core set, sequential (single-digit pass rate expected):
scripts/tb_run.sh terminus                    # n-concurrent 1; budget disk for image builds
```
Outputs → `results_terminalbench/`.

## Open items to verify after install (flagged by research as unconfirmed)
- `tb run --help`: exact subset flag (`--n-tasks`?) and current dataset version.
- Whether `terminus` vs `naive` works better with a 1.5B (use `naive` if needed).
- Disk: each task builds/pulls its own Docker image (multi-GB total across core set).

## Decided scope
Floor probe, **single-model `terminus`**, **all three sizes (1.5B/3B/7B)**, full
`terminal-bench-core==0.1.1`, sequential (`--n-concurrent 1`), run AFTER the sweep.

## Status
- [x] Researched API/integration (LiteLLM `openai/<alias>` → local server; `BaseAgent`).
- [x] Isolated `.venv-tb` + `terminal-bench` installed; `tb` CLI verified. Flags confirmed:
      `--n-tasks`, `--task-id` (glob), `--model provider/name`, `--agent oracle|naive|terminus|...`,
      `--n-concurrent`, `--dataset name==version`.
- [x] `scripts/tb_serve_model.sh`, `scripts/tb_run.sh`, custom-agent stub written.
- [x] Oracle smoke: harness ran end-to-end (built image, ran container, wrote results.json) →
      **mechanically validated**. BUT hello-world **timed out at 60s** (0/1). Cause: CPU
      contention from the concurrent 2-stream sweep (permutation tests + embeddings). NOT a
      harness bug — re-run oracle when the machine is unloaded to confirm it resolves 1/1.
- [ ] Re-confirm oracle (unloaded) → then floor probe: `terminus` × {1.5b,3b,7b} on core, sequential.
      Run after sweep (~22:30) via the model server; expect single-digit/~0 pass rates.
