#!/usr/bin/env bash
# Master orchestrator — runs the whole remaining pipeline in sequence, with a
# hard guarantee that 7B (and every solo phase) NEVER runs in parallel with
# anything else. Resumable throughout (each driver skips completed work).
#
#   Phase A  (PARALLEL): 3B stream  ||  1.5B stream     -> wait for BOTH
#   Phase B  (SOLO):     7B coverage (n=20)
#   Phase C  (SOLO):     agentic self-debug HumanEval (1.5B/3B/7B)
#   Phase D  (SOLO):     Terminal-Bench floor probe (1.5B/3B/7B)
#
# Safe to relaunch after an interruption: completed phases/seeds are skipped.
set -u
cd "$(dirname "$0")/.."
LOG=results_multiseed/_orchestrator.log
mkdir -p results_multiseed
echo "================ ORCHESTRATOR START $(date '+%F %H:%M:%S') ================" | tee -a "$LOG"

# ---- Phase A: parallel 3B + 1.5B ----
echo "[A] $(date '+%H:%M:%S') launch parallel 3B + 1.5B streams" | tee -a "$LOG"
bash scripts/run_multiseed.sh &        BIG=$!
sleep 20                                # stagger model loads
bash scripts/run_multiseed_small.sh &  SMALL=$!
wait "$BIG";   echo "[A] $(date '+%H:%M:%S') 3B stream done (rc=$?)"  | tee -a "$LOG"
wait "$SMALL"; echo "[A] $(date '+%H:%M:%S') 1.5B stream done (rc=$?)" | tee -a "$LOG"

# Safety: ensure no swarm runner lingers before any solo phase.
while pgrep -f "run_experiment.py" >/dev/null; do echo "[wait] runner still alive..." | tee -a "$LOG"; sleep 10; done

# ---- Phase B: 7B coverage, SOLO ----
echo "[B] $(date '+%H:%M:%S') 7B coverage (solo)" | tee -a "$LOG"
bash scripts/run_multiseed_7b.sh; echo "[B] $(date '+%H:%M:%S') 7B done (rc=$?)" | tee -a "$LOG"
while pgrep -f "run_experiment.py" >/dev/null; do sleep 10; done

# ---- Phase C: agentic self-debug HumanEval, SOLO ----
echo "[C] $(date '+%H:%M:%S') agentic HumanEval (solo)" | tee -a "$LOG"
bash scripts/run_humaneval_agentic.sh; echo "[C] $(date '+%H:%M:%S') agentic HE done (rc=$?)" | tee -a "$LOG"
while pgrep -f "run_experiment.py" >/dev/null; do sleep 10; done
while pgrep -f "run_humaneval_agentic.py" >/dev/null; do sleep 10; done

# ---- Phase D: Terminal-Bench floor probe, SOLO ----
echo "[D] $(date '+%H:%M:%S') Terminal-Bench (solo)" | tee -a "$LOG"
bash scripts/run_terminalbench_all.sh; echo "[D] $(date '+%H:%M:%S') Terminal-Bench done (rc=$?)" | tee -a "$LOG"

echo "================ ORCHESTRATOR ALL DONE $(date '+%F %H:%M:%S') ================" | tee -a "$LOG"
