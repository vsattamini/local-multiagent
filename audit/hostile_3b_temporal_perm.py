"""
ATTACK 3b (vectorized): Is late>early interaction-growth REAL or a sparsity/difficulty artifact?
Honest test: per seed-half, compute the agent x type interaction SS and subtract the mean of a
null built by permuting success WITHIN task_type (preserves per-half loads & per-type difficulty).
Compare late-half excess vs early-half excess across seeds. Vectorized permutation.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _hostile_loader import load_run
import numpy as np, pandas as pd
from scipy import stats

def excess(sub, nperm=500, rng=None):
    sub=sub[["agent_id","task_type","success"]].copy()
    if sub.agent_id.nunique()<2 or sub.task_type.nunique()<2: return np.nan
    a=pd.Categorical(sub.agent_id); t=pd.Categorical(sub.task_type)
    ai=a.codes; ti=t.codes; y=sub.success.values.astype(float)
    na=ai.max()+1; nt=ti.max()+1
    flat=ai*nt+ti
    def stat(yv):
        cell_sum=np.bincount(flat, yv, minlength=na*nt).reshape(na,nt)
        cell_n=np.bincount(flat, minlength=na*nt).reshape(na,nt).astype(float)
        with np.errstate(invalid="ignore",divide="ignore"):
            g=cell_sum/cell_n
        am=np.nansum(cell_sum,1)/np.nansum(cell_n,1)
        tm=np.nansum(cell_sum,0)/np.nansum(cell_n,0)
        gm=yv.mean()
        resid=g-am[:,None]-tm[None,:]+gm
        return np.nansum(cell_n*np.where(cell_n>0,resid,0)**2)/cell_n.sum()
    obs=stat(y)
    # permute y within each task_type block
    order=np.argsort(ti, kind="stable")
    ti_sorted=ti[order]; bounds=np.searchsorted(ti_sorted, np.arange(nt+1))
    nulls=np.empty(nperm)
    for k in range(nperm):
        yp=y.copy()
        for j in range(nt):
            idx=order[bounds[j]:bounds[j+1]]
            yp[idx]=rng.permutation(y[idx])
        nulls[k]=stat(yp)
    return obs-nulls.mean()

rng=np.random.default_rng(0)
for run in ["popN_8","popN_12","popN_16","het_swarm_ensemble"]:
    df=load_run(run); diffs=[]
    for sd,g in df.groupby("seed"):
        g=g.sort_values("timestamp").copy()
        g["h"]=g.groupby("agent_id").cumcount()
        g["h"]=g.groupby("agent_id")["h"].transform(lambda x:(x>=x.median()).astype(int))
        e=excess(g[g.h==0],rng=rng); l=excess(g[g.h==1],rng=rng)
        if not(np.isnan(e) or np.isnan(l)): diffs.append(l-e)
    diffs=np.array(diffs)
    t,p=stats.ttest_1samp(diffs,0); p_one=p/2 if t>0 else 1-p/2
    print(f"{run:20s}: mean(late_excess-early_excess)={diffs.mean():+.5f} one-sided p={p_one:.4f} (n={len(diffs)})  late_excess mean alone test below")
    # also raw late_excess > 0 ?
    le=[]
    for sd,g in df.groupby("seed"):
        g=g.sort_values("timestamp").copy()
        g["h"]=g.groupby("agent_id").cumcount()
        g["h"]=g.groupby("agent_id")["h"].transform(lambda x:(x>=x.median()).astype(int))
        v=excess(g[g.h==1],rng=rng)
        if not np.isnan(v): le.append(v)
    le=np.array(le); tl,pl=stats.ttest_1samp(le,0); pl1=pl/2 if tl>0 else 1-pl/2
    print(f"    late-half excess-over-null: mean={le.mean():+.5f} one-sided p={pl1:.4f}")
