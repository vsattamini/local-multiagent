"""
ATTACK 3: Temporal divergence. Even if END-STATE is null, does any agent's per-type
performance DIVERGE over the run (late-run specialization)? Split each seed's run by
timestamp into early/late halves; measure the agent x type interaction stat in each;
test whether LATE > EARLY (one-sided, across seeds). Also track S over the run from snapshots.
"""
import sys, os, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _hostile_loader import load_run
import numpy as np, pandas as pd
from scipy import stats

def inter_stat(data):
    if data.agent_id.nunique()<2 or data.task_type.nunique()<2: return np.nan
    g = data.groupby(["agent_id","task_type"]).success.mean().unstack()
    am=data.groupby("agent_id").success.mean(); tm=data.groupby("task_type").success.mean()
    gm=data.success.mean()
    resid=g.sub(am,axis=0).sub(tm,axis=1)+gm
    w=data.groupby(["agent_id","task_type"]).success.count().unstack().fillna(0)
    n=w.values.sum()
    return float(np.nansum(w.values*(resid.values**2))/n)  # normalized per-obs

for run in ["popN_8","popN_12","popN_16","mbpp_1.5b_lowtemp","mbpp_3b_lowtemp","het_swarm_ensemble"]:
    df = load_run(run)
    early_s, late_s, deltas = [], [], []
    for sd,g in df.groupby("seed"):
        g = g.sort_values("timestamp")
        # within seed, split each agent's stream so loads stay balanced across halves
        g["half"] = g.groupby("agent_id").cumcount()
        g["half"] = g.groupby("agent_id")["half"].transform(lambda x: (x >= x.median()).astype(int))
        e = inter_stat(g[g.half==0]); l = inter_stat(g[g.half==1])
        if not (np.isnan(e) or np.isnan(l)):
            early_s.append(e); late_s.append(l); deltas.append(l-e)
    deltas=np.array(deltas)
    if len(deltas)>1:
        t,p = stats.ttest_1samp(deltas,0)
        p_one = p/2 if t>0 else 1-p/2  # one-sided late>early
        print(f"{run:20s}: early={np.mean(early_s):.4f} late={np.mean(late_s):.4f} "
              f"deltaLate-Early={deltas.mean():+.4f}  one-sided p(late>early)={p_one:.4f}  (n={len(deltas)} seeds)")
    else:
        print(f"{run:20s}: insufficient")

# S trajectory from snapshots: does S rise monotonically (specialization building)?
print("\n=== S trajectory (snapshots): slope of S vs task_count, per seed ===")
for run in ["popN_8","popN_16","mbpp_1.5b_lowtemp"]:
    slopes=[]
    for sd in sorted(glob.glob(f"results_phase3/{run}/seed_*")):
        if not os.path.exists(sd+"/final_metrics.json"): continue
        f=sd+"/snapshots.jsonl"
        if not os.path.exists(f): continue
        tc,S=[],[]
        for l in open(f):
            d=json.loads(l)
            if d.get("task_count",0)>0:
                tc.append(d["task_count"]); S.append(d["specialization_index"])
        if len(tc)>3:
            sl=np.polyfit(tc,S,1)[0]; slopes.append(sl)
    slopes=np.array(slopes)
    if len(slopes):
        t,p=stats.ttest_1samp(slopes,0)
        print(f"  {run:18s}: mean S-slope={slopes.mean():+.2e}/task  p={p:.3f}  (rising specialization if >0)")
