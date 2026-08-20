"""Shared loader for hostile re-audit. Read-only."""
import json, glob, os, ast
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P3 = os.path.join(ROOT, "results_phase3")

def load_run(name):
    """Return DataFrame of all task_log rows across valid seeds (final_metrics present)."""
    rows = []
    for sd in sorted(glob.glob(os.path.join(P3, name, "seed_*"))):
        if not os.path.exists(os.path.join(sd, "final_metrics.json")):
            continue
        seed = os.path.basename(sd).replace("seed_", "")
        f = os.path.join(sd, "task_log.jsonl")
        if not os.path.exists(f):
            continue
        for l in open(f):
            d = json.loads(l)
            d["seed"] = seed
            rows.append(d)
    df = pd.DataFrame(rows)
    if len(df):
        df["success"] = df["success"].astype(int)
    return df

def het_model_of(agent_id):
    # het_swarm_ensemble: agents 0,1 = 1.5B, agent 2 = 3B
    return "3b" if agent_id == 2 else "1.5b"

# ---- problem feature extraction from the HumanEval prompts (latent axes) ----
_PROMPT_CACHE = None
def humaneval_prompts():
    global _PROMPT_CACHE
    if _PROMPT_CACHE is not None:
        return _PROMPT_CACHE
    import gzip
    # locate dataset
    cands = glob.glob(os.path.join(ROOT, "**", "HumanEval*.jsonl*"), recursive=True)
    cands += glob.glob(os.path.join(ROOT, "**", "*human*eval*.json*"), recursive=True)
    data = {}
    for c in cands:
        try:
            op = gzip.open if c.endswith(".gz") else open
            for l in op(c, "rt"):
                d = json.loads(l)
                if "task_id" in d and "prompt" in d:
                    data[d["task_id"]] = d
        except Exception:
            pass
    _PROMPT_CACHE = data
    return data

def ast_size(code):
    try:
        t = ast.parse(code)
        return sum(1 for _ in ast.walk(t))
    except Exception:
        return np.nan

if __name__ == "__main__":
    df = load_run("het_swarm_ensemble")
    print("het rows", len(df), "seeds", df.seed.nunique(), "agents", sorted(df.agent_id.unique()))
    print(df.groupby("task_type").success.agg(["mean","count"]))
    hp = humaneval_prompts()
    print("prompts loaded:", len(hp))
