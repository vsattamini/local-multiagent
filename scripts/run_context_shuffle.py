#!/usr/bin/env python3
"""
Causal context-shuffle test (metrics agent's Tier-2 #1).

For a COMPLETED swarm run: reconstruct each agent's FINAL context buffer (its last
<=K successful (problem, solution) pairs, FIFO), then re-infer all 164 tasks with
routing FIXED to the observed assignment, under two conditions:
  - control:  each agent uses its OWN final buffer
  - shuffled: each agent uses ANOTHER agent's buffer (cyclic swap)
Compares Pass@1 (and per-type success). A large control-minus-shuffled drop ⇒
specialization is functionally load-bearing; ~0 ⇒ epiphenomenal (routing artifact).

Usage:
  python scripts/run_context_shuffle.py --src results_multiseed/exp_3b_low_temp \
         --model-size 3b --seeds 42 123 456 789 1011 --out results_context_shuffle
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from models.llama_cpp import LlamaCppModel
from swarm.humaneval import HumanEvalLoader
from swarm.executor import HumanEvalExecutor
from swarm.agent import SwarmAgent
from swarm.metrics import MetricsEngine, RobustnessMetrics
from swarm.types import TaskType

MODELS = {"1.5b": "models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
          "3b": "models/qwen2.5-coder-3b-instruct-q4_k_m.gguf",
          "7b": "models/qwen2.5-coder-7b-instruct-q4_k_m.gguf"}
SYS = ("You are a Python coding assistant. Write clean, correct code that solves the "
       "problem. Only output the function implementation, no explanations or extra text.")


def reconstruct_buffers(log, problems, K=5):
    """Final FIFO buffer per agent = last K successful (problem, solution, type)."""
    bufs = {}
    for t in log:
        if not t["success"]:
            continue
        a = t["agent_id"]
        bufs.setdefault(a, [])
        bufs[a].append((problems[t["task_id"]], t["solution"], t["task_type"]))
    return {a: v[-K:] for a, v in bufs.items()}


def render_prompt(buf, problem):
    p = SYS + "\n\n"
    for prob, sol, _ in buf:
        p += f"### Problem:\n{prob}\n\n### Solution:\n```python\n{sol}\n```\n\n"
    p += f"### Problem:\n{problem}\n\n### Solution:\n```python\n"
    return p


def run_condition(model, ex, log, problems, tasks_by_id, buffers, agent_ids, swap, seed):
    """Re-infer all tasks with routing fixed; swap=False→own buffer, True→cyclic-swapped."""
    n = len(agent_ids)
    order = {a: agent_ids[(i + 1) % n] for i, a in enumerate(agent_ids)} if swap else {a: a for a in agent_ids}
    succ = 0; per = {}  # (agent,type)->[s,t]
    rec = []
    for i, t in enumerate(log):
        a = t["agent_id"]; tid = t["task_id"]
        buf = buffers.get(order[a], [])
        prompt = render_prompt(buf, problems[tid])
        sol = model.generate(prompt, max_tokens=512, temperature=0.2, top_p=0.95,
                             seed=seed + i * 131 + a)
        r = ex.execute_humaneval(sol, tasks_by_id[tid].test_code, tasks_by_id[tid].entry_point)
        succ += int(r.success)
        ty = t["task_type"]; per.setdefault((a, ty), [0, 0]); per[(a, ty)][1] += 1; per[(a, ty)][0] += int(r.success)
        rec.append({"task_id": tid, "agent": a, "buffer_from": order[a], "type": ty, "success": r.success})
    return succ / len(log), rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="results_multiseed/<config> dir")
    ap.add_argument("--model-size", required=True, choices=list(MODELS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--out", default="results_context_shuffle")
    args = ap.parse_args()

    loader = HumanEvalLoader()
    tasks = loader.get_tasks_from_json(n_tasks=164)
    problems = {t.id: t.problem for t in tasks}
    tasks_by_id = {t.id: t for t in tasks}
    me = MetricsEngine.__new__(MetricsEngine)  # only need specialization_index

    print(f"Loading {args.model_size} ...")
    model = LlamaCppModel(model_name=f"qwen-{args.model_size}", model_path=MODELS[args.model_size],
                          n_ctx=4096, n_gpu_layers=-1); model.load()
    ex = HumanEvalExecutor(timeout=10, clean_mode="strict")
    name = Path(args.src).name

    for seed in args.seeds:
        sd = Path(args.src) / f"seed_{seed}" / "task_log.jsonl"
        if not sd.exists():
            print(f"skip seed {seed} (no {sd})"); continue
        outdir = Path(args.out) / name / f"seed_{seed}"
        if (outdir / "shuffle_result.json").exists():
            print(f"skip {outdir} (done)"); continue
        outdir.mkdir(parents=True, exist_ok=True)
        log = [json.loads(l) for l in open(sd) if l.strip()]
        buffers = reconstruct_buffers(log, problems)
        agent_ids = sorted(buffers.keys())
        S_obs = me.specialization_index(log)
        print(f"=== {name} seed {seed}: {len(agent_ids)} agents w/ buffers ===")
        ctrl_p1, _ = run_condition(model, ex, log, problems, tasks_by_id, buffers, agent_ids, False, seed)
        shuf_p1, _ = run_condition(model, ex, log, problems, tasks_by_id, buffers, agent_ids, True, seed)
        res = RobustnessMetrics.context_shuffle_sensitivity(S_obs, S_obs, ctrl_p1, shuf_p1)
        res.update({"control_pass1": ctrl_p1, "shuffled_pass1": shuf_p1,
                    "pass1_drop": ctrl_p1 - shuf_p1, "S_observed": S_obs, "seed": seed, "src": name})
        json.dump(res, open(outdir / "shuffle_result.json", "w"), indent=2)
        print(f"  control={ctrl_p1:.3f}  shuffled={shuf_p1:.3f}  drop={ctrl_p1-shuf_p1:+.3f}")


if __name__ == "__main__":
    main()
