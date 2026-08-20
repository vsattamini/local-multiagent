"""
ATTACK 2b: Probe the mbpp_3b_lowtemp 'hit' (p_cluster=0.0000). Is it real per-type SUCCESS
specialization, or an artifact of single-assignment + 4 seeds + degenerate task clusters?
Key threat: under single-assignment, a given task_id is usually attempted by only ONE agent,
so cluster=task_id can't separate agent from task -> cluster-robust Wald is unreliable.
The honest test: does an agent's per-type success rate beat peers BEYOND what the routing
(which types it got) + overall skill explain? Use a permutation test that shuffles the
agent->task assignment WITHIN type, preserving per-type difficulty and per-agent load.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _hostile_loader import load_run
import numpy as np, pandas as pd
from scipy import stats

for run in ["mbpp_3b_lowtemp","mbpp_1.5b_lowtemp","popN_8","popN_12"]:
    df = load_run(run)
    print(f"\n===== {run}  (rows={len(df)}, seeds={df.seed.nunique()}, agents={df.agent_id.nunique()}) =====")
    # how many distinct agents attempt each task_id within a seed?
    perseed = df.groupby(["seed","task_id"]).agent_id.nunique()
    print(f"  agents-per-(seed,task): mean={perseed.mean():.2f} max={perseed.max()} -> single-assignment={perseed.mean()<1.5}")
    # per-agent per-type success table
    tab = df.groupby(["agent_id","task_type"]).success.agg(["mean","count"])
    print("  per-agent x per-type success (mean[n]):")
    piv_m = df.groupby(["agent_id","task_type"]).success.mean().unstack()
    piv_n = df.groupby(["agent_id","task_type"]).success.count().unstack()
    print(piv_m.round(2).to_string())
    print("  (n per cell:)"); print(piv_n.to_string())

    # ---- HONEST permutation: within each (seed, task_type), shuffle which agent got which task,
    # keeping each agent's attempt-count per type fixed and each task's outcome tied to difficulty.
    # Statistic = max over agents of (agent overall success-rate spread across types), i.e.
    # the agent x type interaction measured as variance of residual after removing agent & type main effects.
    d = df.copy()
    # observed interaction statistic: sum of squared (cell_rate - agent_marg - type_marg + grand)
    def interaction_stat(data):
        g = data.groupby(["agent_id","task_type"]).success.mean().unstack()
        am = data.groupby("agent_id").success.mean()
        tm = data.groupby("task_type").success.mean()
        gm = data.success.mean()
        resid = g.sub(am, axis=0).sub(tm, axis=1) + gm
        w = data.groupby(["agent_id","task_type"]).success.count().unstack().fillna(0)
        return float(np.nansum(w.values * (resid.values**2)))
    obs = interaction_stat(d)
    rng = np.random.default_rng(0)
    null = []
    arr = d[["seed","task_type","agent_id","success"]].copy()
    for _ in range(2000):
        s = arr.copy()
        # shuffle success within (seed, task_type) -> breaks agent-success link, keeps type difficulty & loads
        s["success"] = s.groupby(["seed","task_type"])["success"].transform(lambda x: rng.permutation(x.values))
        null.append(interaction_stat(s))
    null = np.array(null)
    p = (np.sum(null >= obs)+1)/(len(null)+1)
    print(f"  PERMUTATION (shuffle success within seed x type): obs={obs:.2f}, null mean={null.mean():.2f}, p={p:.4f}")
