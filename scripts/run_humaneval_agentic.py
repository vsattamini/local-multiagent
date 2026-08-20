#!/usr/bin/env python3
"""
Agentic self-debug HumanEval — a Terminal-Bench-style iterate-with-verifier loop,
but in the code-generation domain and with a SINGLE model (no swarm).

For each of the 164 HumanEval tasks the model gets up to K attempts:
  attempt 1: write a solution from the problem;
  on failure: it is shown its previous code + the unit-test error and asked to fix;
the real HumanEval tests (verified executor) are the verifier.

Reports per model size:
  - pass@1            : solved on the FIRST attempt (clean single-shot baseline,
                        directly comparable to published HumanEval Pass@1)
  - solve@K_feedback  : solved within K attempts using test feedback (the agentic number)
  - mean attempts, and both metrics broken down by task type.

Single-model, seeded, sequential. Run AFTER the swarm sweep frees the GPU.

Usage:
  python scripts/run_humaneval_agentic.py --model-size 7b --seeds 42 123 --max-attempts 5 \
         --output-dir results_humaneval_agentic
"""
import argparse, json, sys, time
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import numpy as np
from models.llama_cpp import LlamaCppModel
from swarm.humaneval import HumanEvalLoader

MODELS = {
    "1.5b": "models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
    "3b":   "models/qwen2.5-coder-3b-instruct-q4_k_m.gguf",
    "7b":   "models/qwen2.5-coder-7b-instruct-q4_k_m.gguf",
}
SYSTEM = ("You are a Python coding assistant. Write clean, correct code that solves "
          "the problem. Only output the function implementation, no explanations.")


def first_prompt(problem: str) -> str:
    return f"{SYSTEM}\n\n### Problem:\n{problem}\n\n### Solution:\n```python\n"


def retry_prompt(problem: str, prev_code: str, error: str) -> str:
    err = (error or "").strip()[:600]
    return (f"{SYSTEM}\n\n### Problem:\n{problem}\n\n"
            f"### Previous attempt (FAILED):\n```python\n{prev_code.strip()}\n```\n"
            f"### Test error:\n{err}\n\n"
            f"Provide a corrected, complete implementation.\n### Solution:\n```python\n")


def run_seed(model, loader, executor, seed, max_attempts):
    np.random.seed(seed)
    tasks = loader.get_tasks_from_json(n_tasks=164)
    records = []
    for i, task in enumerate(tasks):
        prev_code, err = None, None
        solved_at = None
        for attempt in range(1, max_attempts + 1):
            prompt = (first_prompt(task.problem) if attempt == 1
                      else retry_prompt(task.problem, prev_code, err))
            sol = model.generate(prompt, max_tokens=512, temperature=0.2, top_p=0.95, seed=seed)
            res = executor.execute_humaneval(sol, task.test_code, task.entry_point)
            if res.success:
                solved_at = attempt
                break
            prev_code, err = sol, res.error_message
        records.append({
            "task_id": task.id, "task_type": task.task_type.value,
            "solved_at": solved_at,
            "pass1": solved_at == 1,
            "passK": solved_at is not None,
            "attempts": solved_at if solved_at else max_attempts,
        })
        if (i + 1) % 20 == 0:
            p1 = np.mean([r["pass1"] for r in records])
            pk = np.mean([r["passK"] for r in records])
            print(f"  [{i+1}/164] running pass@1={p1:.3f} solve@K={pk:.3f}", flush=True)
    return records


def summarize(records, max_attempts):
    n = len(records)
    by_type = defaultdict(lambda: [0, 0, 0])  # type -> [n, pass1, passK]
    for r in records:
        t = by_type[r["task_type"]]
        t[0] += 1; t[1] += int(r["pass1"]); t[2] += int(r["passK"])
    return {
        "n": n,
        "pass_at_1": float(np.mean([r["pass1"] for r in records])),
        "solve_at_K_feedback": float(np.mean([r["passK"] for r in records])),
        "mean_attempts_to_solve": float(np.mean([r["solved_at"] for r in records if r["solved_at"]]) or 0),
        "max_attempts": max_attempts,
        "by_type": {k: {"n": v[0], "pass@1": v[1]/v[0], "solve@K": v[2]/v[0]} for k, v in by_type.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-size", choices=list(MODELS), required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--max-attempts", type=int, default=5)
    ap.add_argument("--output-dir", default="results_humaneval_agentic")
    args = ap.parse_args()

    from swarm.executor import HumanEvalExecutor
    model_path = MODELS[args.model_size]
    print(f"Loading {model_path} ...")
    model = LlamaCppModel(model_name=f"qwen-{args.model_size}", model_path=model_path,
                          n_ctx=4096, n_gpu_layers=-1)
    model.load()
    loader = HumanEvalLoader()
    executor = HumanEvalExecutor(timeout=10)

    for seed in args.seeds:
        out = Path(args.output_dir) / args.model_size / f"seed_{seed}"
        if (out / "summary.json").exists():
            print(f"skip {out} (done)"); continue
        out.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        print(f"\n=== {args.model_size} seed {seed} (max_attempts={args.max_attempts}) ===")
        records = run_seed(model, loader, executor, seed, args.max_attempts)
        summ = summarize(records, args.max_attempts)
        summ["elapsed_sec"] = time.time() - t0
        summ["model_size"] = args.model_size
        summ["seed"] = seed
        json.dump(records, open(out / "records.json", "w"), indent=2)
        json.dump(summ, open(out / "summary.json", "w"), indent=2)
        print(f"  DONE seed {seed}: pass@1={summ['pass_at_1']:.3f} "
              f"solve@{args.max_attempts}={summ['solve_at_K_feedback']:.3f} "
              f"({summ['elapsed_sec']:.0f}s)")


if __name__ == "__main__":
    main()
