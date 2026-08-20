#!/usr/bin/env python3
"""
Label-free difficulty-axis interaction test (de-hostages the 4-way task taxonomy).

The thesis's functional-differentiation null is measured on a hand-built 4-way
type taxonomy (string/math/list/logic) where `logic` is a 6-10 problem keyword
*fallback* carrying every marginal signal. cap5 §5.5.2 pre-registers a LABEL-FREE
latent axis (length / AST size / complexity) as the PRIMARY robustness check, so the
null does not hinge on that fragile taxonomy. This script delivers it.

We build a HumanEval difficulty axis with the SAME features pre-registered for the
MBPP axis (data/mbpp_difficulty_axis.json): prompt_chars, prompt_tokens, ast_nodes,
cyclomatic, canon_lines -> difficulty_z (composite z-score) + difficulty_tercile.
Then, on the every-agent-every-task ENSEMBLE runs (the only design where the
interaction is computable, no perfect separation), we fit

    success ~ C(agent_id) * difficulty_z      (GEE, cov=exchangeable, groups=task_id)

The agent×difficulty_z INTERACTION is the label-free analogue of the agent×type
interaction = does an agent's success vary with problem difficulty DIFFERENTLY from
its peers? Among IDENTICAL agents this must be null if the type-null is not a taxonomy
artifact. We cluster on task_id (the correct unit: seeds reuse the same 164 problems),
exactly as in het_interaction_result.md v3.

READ-ONLY w.r.t. experiments: consumes existing logs only. No GPU. No generation.

Usage: .venv/bin/python audit/difficulty_axis_interaction.py
"""
import ast as _ast
import glob
import json
import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HE = os.path.join(ROOT, "data", "HumanEval.jsonl")
AXIS_OUT = os.path.join(ROOT, "data", "humaneval_difficulty_axis.json")

# Ensemble runs where the interaction is computable (every-agent-every-task, HumanEval).
ENSEMBLE_RUNS = {
    "het_swarm_ensemble": "results_phase3/het_swarm_ensemble",   # 2x1.5B + 1x3B (heterogeneous)
    "ens_3b_n3":          "results_phase3/ens_3b_n3",             # 3x3B homogeneous control
    "ens_3b":             "results_phase3/ens_3b",                # 4x3B homogeneous
    "ens_7b":             "results_phase3/ens_7b",                # 4x7B homogeneous
}


# ---------- 1. build the HumanEval difficulty axis (same features as the MBPP axis) ----------
def _cyclomatic(tree):
    """Branch-point count + 1 (McCabe-style, AST-based)."""
    branches = (_ast.If, _ast.For, _ast.While, _ast.And, _ast.Or,
                _ast.ExceptHandler, _ast.With, _ast.Assert, _ast.comprehension,
                getattr(_ast, "IfExp", _ast.If))
    return 1 + sum(isinstance(n, branches) for n in _ast.walk(tree))


def build_axis():
    rows = []
    for line in open(HE):
        p = json.loads(line)
        prompt, canon = p["prompt"], p["canonical_solution"]
        # canonical_solution is the body that completes the prompt signature; parse the
        # full function so AST is well-formed.
        full = prompt + canon
        try:
            tree = _ast.parse(full)
            ast_nodes = sum(1 for _ in _ast.walk(tree))
            cyc = _cyclomatic(tree)
        except SyntaxError:
            # fall back to parsing the canonical body alone, indented under a def
            try:
                tree = _ast.parse(canon)
                ast_nodes = sum(1 for _ in _ast.walk(tree))
                cyc = _cyclomatic(tree)
            except SyntaxError:
                ast_nodes, cyc = np.nan, np.nan
        rows.append({
            "task_id": p["task_id"],
            "prompt_chars": len(prompt),
            "prompt_tokens": len(prompt.split()),
            "ast_nodes": ast_nodes,
            "cyclomatic": cyc,
            "canon_lines": sum(1 for ln in canon.splitlines() if ln.strip()),
        })
    df = pd.DataFrame(rows)
    # composite difficulty = mean of z-scored structural features (label-free)
    feats = ["prompt_chars", "prompt_tokens", "ast_nodes", "cyclomatic", "canon_lines"]
    z = (df[feats] - df[feats].mean()) / df[feats].std(ddof=0)
    df["difficulty_z"] = z.mean(axis=1)
    # proper 3-way terciles (the MBPP axis has a medium=0 bug; do it correctly here)
    df["difficulty_tercile"] = pd.qcut(df["difficulty_z"], 3,
                                       labels=["easy", "medium", "hard"]).astype(str)
    df.to_json(AXIS_OUT, orient="records", indent=2)
    return df


# ---------- 2. load ensemble outcomes, gating on final_metrics ----------
def load_run(run_dir):
    frames = []
    for seed_dir in sorted(glob.glob(os.path.join(ROOT, run_dir, "seed_*"))):
        if not os.path.exists(os.path.join(seed_dir, "final_metrics.json")):
            continue  # skip partial/aborted seeds (same gate as the rest of the audit)
        tl = os.path.join(seed_dir, "task_log.jsonl")
        if not os.path.exists(tl):
            continue
        seed = os.path.basename(seed_dir).replace("seed_", "")
        for line in open(tl):
            r = json.loads(line)
            frames.append({"task_id": r["task_id"], "agent_id": int(r["agent_id"]),
                           "success": int(bool(r["success"])), "seed": seed})
    return pd.DataFrame(frames)


# ---------- 3. agent x difficulty interaction, clustered on task_id ----------
def interaction_test(df):
    """GEE logit success ~ C(agent) * difficulty_z, exchangeable, groups=task_id.
    Returns joint Wald p for the interaction terms + the main-effect (agent) joint p."""
    df = df.sort_values("task_id").reset_index(drop=True)
    groups = df["task_id"]
    res = {}
    try:
        m = smf.gee("success ~ C(agent_id) * difficulty_z", "task_id", data=df,
                    family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()
                    ).fit()
        inter = [p for p in m.params.index if ":difficulty_z" in p]
        agent = [p for p in m.params.index if p.startswith("C(agent_id)") and ":" not in p]
        res["interaction_p"] = float(m.wald_test(_constraints(inter, m), scalar=True).pvalue) if inter else None
        res["agent_main_p"]  = float(m.wald_test(_constraints(agent, m), scalar=True).pvalue) if agent else None
        res["n_obs"] = int(len(df)); res["n_tasks"] = int(df["task_id"].nunique())
        res["n_agents"] = int(df["agent_id"].nunique())
    except Exception as e:
        res["error"] = str(e)
    return res


def _constraints(names, model):
    """Build an R matrix selecting the named params for a joint Wald test."""
    idx = {n: i for i, n in enumerate(model.params.index)}
    R = np.zeros((len(names), len(model.params)))
    for r, n in enumerate(names):
        R[r, idx[n]] = 1.0
    return R


def main():
    print("Building HumanEval difficulty axis (label-free; same features as the MBPP pre-reg)...")
    axis = build_axis()
    print(f"  saved {AXIS_OUT}: {len(axis)} problems")
    print(f"  ast_nodes median={axis['ast_nodes'].median():.0f}  "
          f"terciles={axis['difficulty_tercile'].value_counts().to_dict()}")
    print()
    print("Agent x DIFFICULTY interaction (label-free), GEE clustered on task_id:")
    print(f"{'config':<22} {'agents':>6} {'n_obs':>7} {'agent main p':>13} {'INTERACTION p':>14}  verdict")
    axis_small = axis[["task_id", "difficulty_z", "difficulty_tercile"]]
    for name, d in ENSEMBLE_RUNS.items():
        df = load_run(d)
        if df.empty:
            print(f"{name:<22} (no complete seeds)")
            continue
        df = df.merge(axis_small, on="task_id", how="inner")
        r = interaction_test(df)
        if "error" in r:
            print(f"{name:<22} ERROR: {r['error'][:60]}")
            continue
        ip, ap = r["interaction_p"], r["agent_main_p"]
        verdict = "NS (no diff-interaction)" if (ip is None or ip >= 0.05) else "** interaction sig **"
        print(f"{name:<22} {r['n_agents']:>6} {r['n_obs']:>7} "
              f"{(f'{ap:.4f}' if ap is not None else 'n/a'):>13} "
              f"{(f'{ip:.4f}' if ip is not None else 'n/a'):>14}  {verdict}")
    print()
    print("Interpretation: among IDENTICAL agents (ens_3b, ens_7b, ens_3b_n3) a null")
    print("agent×difficulty interaction confirms the functional-differentiation null is")
    print("NOT an artifact of the 4-way type taxonomy. For het, any difficulty interaction")
    print("is a model-SCALE competence effect (bigger model relatively better), not roles.")


if __name__ == "__main__":
    main()
