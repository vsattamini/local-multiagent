#!/usr/bin/env python3
"""
Memory-aware PARALLEL scheduler for the remaining phase-3 runs (8GB GPU).
Per-size concurrency caps (the user's grouping): up to 3x1.5B OR 2x3B OR 1x7B
co-resident; 7B and the heterogeneous (3-model) config run SOLO. A VRAM budget
(process MiB) is the hard backstop so we never oversubscribe the card.

Resumable: each config's missing seeds are computed; completed seeds are skipped;
partial seed dirs are removed before re-running.

Phases: (1) swarm configs in parallel  (2) context-shuffle in parallel
        (3) Terminal-Bench (sequential, manages its own GPU server).
"""
import os, sys, time, shutil, subprocess
from pathlib import Path

ROOT = Path(__file__).parent.parent
os.chdir(ROOT)
NV = ".venv/lib/python3.12/site-packages/nvidia"
ENV = os.environ.copy()
ENV["LD_LIBRARY_PATH"] = ":".join(
    f"{ROOT}/{NV}/{x}/lib" for x in ("cuda_runtime", "cublas", "cuda_nvrtc")
) + ":" + ENV.get("LD_LIBRARY_PATH", "")
PY = "./.venv/bin/python"
SEEDS = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]

COST = {"1.5b": 1700, "3b": 2600, "7b": 5000, "het": 5800}  # process MiB
CAP = {"1.5b": 3, "3b": 2, "7b": 1, "het": 1}
SOLO = {"7b", "het"}
BUDGET = 6600  # sum of process MiB (8188 - ~1350 desktop - headroom)

LOG = open("results_phase3/_parallel.log", "a")
def L(m):
    line = f"{time.strftime('%H:%M:%S')} {m}"
    print(line, flush=True); LOG.write(line + "\n"); LOG.flush()

def missing_seeds(cfg, n):
    out = Path(f"results_phase3/{cfg}"); miss = []
    for s in SEEDS[:n]:
        if not (out / f"seed_{s}" / "final_metrics.json").exists():
            d = out / f"seed_{s}"
            if d.exists():
                shutil.rmtree(d)
            miss.append(s)
    return miss

# (config, n_seeds, size)
# DECISIVE = the runs that produce computable, decision-relevant LR results
# (gap-review priority). het_swarm_ensemble is THE functional-differentiation test;
# ens_3b_n3 is its weight-only-different homogeneous control; ens_7b is the clean 7B
# null; ens_3b (4-agent) is the near-complete homogeneous ensemble null.
DECISIVE = [
    ("ens_3b_n3", 10, "3b"),            # homogeneous control (3x3B)
    ("ens_3b", 10, "3b"),               # 4-agent homogeneous ensemble null (finishing)
    ("het_swarm_ensemble", 10, "het"),  # THE decisive heterogeneous-ensemble run
    ("ens_7b", 5, "7b"),                # clean 7B ensemble null
]
# REST = single-assignment MBPP+ pass@1 numbers (largely predictable; lower priority).
REST = [
    ("mbpp_3b_baseline", 10, "3b"), ("mbpp_3b_lowtemp", 10, "3b"),
    ("mbpp_7b_baseline", 5, "7b"), ("mbpp_7b_lowtemp", 5, "7b"),
]
# NOTE: single-assignment het_swarm DROPPED (subsumed by het_swarm_ensemble; it crashed).
#       Terminal-Bench DROPPED for now (expected-floor probe → footnote).

def build_swarm_jobs(spec):
    jobs = []
    for cfg, n, size in spec:
        ms = missing_seeds(cfg, n)
        if not ms:
            L(f"SKIP {cfg} (done)"); continue
        cmd = [PY, "scripts/run_experiment.py", "--config", f"config/{cfg}.yaml",
               "--seeds", *map(str, ms), "--output-dir", f"results_phase3/{cfg}"]
        jobs.append({"name": cfg, "size": size, "cmd": cmd,
                     "log": f"results_phase3/{cfg}.log"})
    return jobs

def build_shuffle_jobs():
    specs = [("exp_3b_baseline", "3b"), ("exp_3b_low_temp", "3b"), ("exp_7b_model", "7b")]
    jobs = []
    for src, size in specs:
        name = f"shuffle_{src}"
        done = all((Path("results_context_shuffle")/src/f"seed_{s}"/"shuffle_result.json").exists()
                   for s in [42, 123, 456, 789, 1011])
        if done:
            L(f"SKIP {name} (done)"); continue
        cmd = [PY, "scripts/run_context_shuffle.py", "--src", f"results_multiseed/{src}",
               "--model-size", size, "--seeds", "42", "123", "456", "789", "1011",
               "--out", "results_context_shuffle"]
        jobs.append({"name": name, "size": size, "cmd": cmd,
                     "log": f"results_phase3/{name}.log"})
    return jobs

def run_pool(jobs, label):
    L(f"=== {label}: {len(jobs)} jobs ===")
    running = []  # (name, size, proc)
    def used(): return sum(COST[s] for _, s, _ in running)
    def cnt(sz): return sum(1 for _, s, _ in running if s == sz)
    def solo_active(): return any(s in SOLO for _, s, _ in running)
    pending = list(jobs)
    while pending or running:
        running = [(n, s, p) for (n, s, p) in running if p.poll() is None
                   or (L(f"DONE {n} (rc={p.returncode})") or False)]
        progressed = True
        while progressed:
            progressed = False
            for i, j in enumerate(pending):
                s = j["size"]
                if solo_active():
                    break
                if s in SOLO and running:
                    continue
                if cnt(s) >= CAP[s]:
                    continue
                if used() + COST[s] > BUDGET:
                    continue
                p = subprocess.Popen(j["cmd"], stdout=open(j["log"], "a"),
                                     stderr=subprocess.STDOUT, env=ENV)
                running.append((j["name"], s, p)); pending.pop(i)
                L(f"START {j['name']} ({s}) | running={[n for n,_,_ in running]} ~{used()}MiB")
                progressed = True
                break
        time.sleep(15)
    L(f"=== {label} COMPLETE ===")

def main():
    L("################ PHASE3-PARALLEL START ################")
    run_pool(build_swarm_jobs(DECISIVE), "DECISIVE")
    run_pool(build_swarm_jobs(REST), "REST (MBPP+ single-assignment)")
    run_pool(build_shuffle_jobs(), "CONTEXT-SHUFFLE")
    L("################ PHASE3-PARALLEL ALL DONE (Terminal-Bench deferred) ################")

if __name__ == "__main__":
    main()
