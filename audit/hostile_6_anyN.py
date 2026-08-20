"""
ATTACK 6: Is the het +20pp any@N gain emergent multi-agent coordination, or just best-of-N
over two model scales? Strongest pro-emergence case: any@N should EXCEED what you'd get from
N draws of the SINGLE BEST agent (the 3B alone, best-of-N). If het any@N > best-single-agent
any@N, the diversity (the 1.5B agents solving things the 3B misses) adds real coverage.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _hostile_loader import load_run, het_model_of
import numpy as np, pandas as pd
from scipy import stats

df=load_run("het_swarm_ensemble"); df["model"]=df.agent_id.apply(het_model_of)
# per seed x task: success vector over the 3 agents
rows=[]
for (sd,tid),g in df.groupby(["seed","task_id"]):
    s={int(r.agent_id):int(r.success) for _,r in g.iterrows()}
    rows.append({"seed":sd,"task_id":tid,"a0":s.get(0,0),"a1":s.get(1,0),"a2":s.get(2,0)})
M=pd.DataFrame(rows)
# per seed: pass@1 (avg agent), any@3 (any of 3), 3B-alone, best-of-the-two-1.5B (a0,a1 = 2 draws same model)
res=[]
for sd,g in M.groupby("seed"):
    pass1=g[["a0","a1","a2"]].values.mean()
    anyN=(g[["a0","a1","a2"]].max(axis=1)).mean()
    only3b=g["a2"].mean()
    # the KEY contrast: does the het ensemble beat the 3B ALONE? (the best single agent)
    # and does it beat 3B + best-of its OWN self-consistency? approx by 3B alone here.
    anyN_minus_3b = anyN - only3b
    # contribution of the 1.5B agents: tasks the 3B FAILS but a 1.5B solves
    rescue = ((g["a2"]==0)&((g["a0"]==1)|(g["a1"]==1))).mean()
    res.append((sd,pass1,anyN,only3b,anyN_minus_3b,rescue))
R=pd.DataFrame(res,columns=["seed","pass1","anyN","only3b","anyN_minus_3b","rescue_by_small"])
print("Per-seed (het 2x1.5B + 1x3B):")
print(R.round(3).to_string(index=False))
print(f"\nmean pass@1={R.pass1.mean():.3f}  any@N={R.anyN.mean():.3f}  any@N - pass@1 = {R.anyN.mean()-R.pass1.mean():+.3f}")
print(f"3B-alone={R.only3b.mean():.3f}")
t,p=stats.ttest_1samp(R.anyN_minus_3b,0)
print(f"\nKEY: any@N - (3B alone) = {R.anyN_minus_3b.mean():+.3f}  t={t:.2f} p={p:.4f}")
print(f"  => the het ensemble beats the single best agent (3B) by this much via 1.5B coverage")
print(f"fraction of tasks where 3B FAILS but a 1.5B agent SUCCEEDS (rescue): {R.rescue_by_small.mean():.3f}")
print("\nHOSTILE READ: this is best-of-N with a diverse pool (test-oracle selection). It is")
print("real coverage from substrate heterogeneity, NOT coordination/communication between agents")
print("(agents never see each other; selection is by the hidden test-suite oracle).")

# Contrast: homogeneous best-of-N (ens_3b) to show the gain is mostly self-consistency there
h=load_run("ens_3b")
rr=[]
for sd,g in h.groupby("seed"):
    piv=g.pivot_table(index="task_id",columns="agent_id",values="success",aggfunc="max").fillna(0)
    rr.append((piv.values.mean(), piv.max(axis=1).mean()))
H=pd.DataFrame(rr,columns=["pass1","anyN"])
print(f"\nHomogeneous ens_3b: pass@1={H.pass1.mean():.3f} any@N={H.anyN.mean():.3f} gain={H.anyN.mean()-H.pass1.mean():+.3f}")
print("  (homogeneous gain is small = identical agents give little coverage diversity)")
