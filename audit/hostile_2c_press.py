"""
ATTACK 2c: Press the mbpp_3b_lowtemp p=0.0195. Threats:
 (1) driven by agent0's n=5 string @ 0% (tiny degenerate cell)
 (2) only 4 seeds -> per-seed reproducibility?
 (3) a per-type ATTEMPT(routing) signal masquerading as a SUCCESS niche?
 (4) multiple-comparisons: this is 1 of ~10 runs tested.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _hostile_loader import load_run
import numpy as np, pandas as pd

df = load_run("mbpp_3b_lowtemp")

def perm_stat(d, nperm=4000, seed=0):
    rng = np.random.default_rng(seed)
    def stat(data):
        g = data.groupby(["agent_id","task_type"]).success.mean().unstack()
        am = data.groupby("agent_id").success.mean()
        tm = data.groupby("task_type").success.mean()
        gm = data.success.mean()
        resid = g.sub(am,axis=0).sub(tm,axis=1)+gm
        w = data.groupby(["agent_id","task_type"]).success.count().unstack().fillna(0)
        return float(np.nansum(w.values*(resid.values**2)))
    obs = stat(d); null=[]
    for _ in range(nperm):
        s=d.copy()
        s["success"]=s.groupby(["seed","task_type"])["success"].transform(lambda x: rng.permutation(x.values))
        null.append(stat(s))
    null=np.array(null); return obs,(np.sum(null>=obs)+1)/(len(null)+1)

# (1) drop cells with n<15
cnt = df.groupby(["agent_id","task_type"]).success.transform("count")
df_big = df[cnt>=15]
o0,p0 = perm_stat(df); o1,p1 = perm_stat(df_big)
print(f"(1) full: obs={o0:.2f} p={p0:.4f}  |  drop cells n<15 (removes agent0 string n=5): obs={o1:.2f} p={p1:.4f}")
print("    remaining cells:"); print(df_big.groupby(["agent_id","task_type"]).success.count().unstack().to_string())

# (2) per-seed: is the interaction consistent or driven by 1 seed?
print("\n(2) per-seed interaction stat (each seed alone):")
for sd,g in df.groupby("seed"):
    o,p = perm_stat(g, nperm=1500, seed=1)
    print(f"    seed {sd}: obs={o:.2f} p={p:.4f}  agent-type cells n: {g.groupby('agent_id').size().to_dict()}")

# (3) is it routing not skill? Look at which agent OWNS which type by attempts, and whether
#     the 'niche' agent's success on its owned type beats OTHER agents on that SAME type.
print("\n(3) per-type: best vs worst agent success rate (only cells n>=20):")
for t,g in df.groupby("task_type"):
    rr = g.groupby("agent_id").success.agg(["mean","count"])
    rr = rr[rr["count"]>=20]
    if len(rr)>=2:
        print(f"    {t:7s}: {[(int(a),round(r['mean'],2),int(r['count'])) for a,r in rr.iterrows()]}  spread={rr['mean'].max()-rr['mean'].min():.2f}")
