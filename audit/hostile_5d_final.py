"""
ATTACK 5d: The decisive within-agent diagonal test. For a REAL positive specialization, a
persona-k agent must be better at type k than at the OTHER types IT ITSELF attempts
(within-agent contrast: own-type rate - other-type rate). This removes the 'agents differ in
overall skill' confound entirely. Test across persona runs, per seed, with sign + t test.
Also: aggregate diagonal-vs-offdiagonal across all rrpersona runs.
"""
import sys, os, json, glob
import numpy as np, pandas as pd
from scipy import stats
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def load(name):
    rows=[]
    for sd in sorted(glob.glob(os.path.join(ROOT,"results_multiseed",name,"seed_*"))):
        if not os.path.exists(os.path.join(sd,"final_metrics.json")): continue
        for l in open(os.path.join(sd,"task_log.jsonl")):
            d=json.loads(l); d["seed"]=os.path.basename(sd)[5:]; rows.append(d)
    df=pd.DataFrame(rows); df["success"]=df["success"].astype(int); return df
PT={0:"string",1:"math",2:"list",3:"logic"}

print("WITHIN-AGENT diagonal: (own persona-type rate) - (mean of other-types rate), per agent.")
print("Positive => agent really IS better at its assigned type (true specialization).\n")
for r in ["mech_1_5b_persona","mech_3b_persona","mech_1_5b_rrpersona","mech_3b_rrpersona","mech_7b_rrpersona"]:
    df=load(r)
    print(f"== {r} ==")
    perseed_diag=[]
    for sd,g in df.groupby("seed"):
        ds=[]
        for a,t in PT.items():
            own=g[(g.agent_id==a)&(g.task_type==t)].success
            oth=g[(g.agent_id==a)&(g.task_type!=t)].success
            if len(own)>=2 and len(oth)>=2: ds.append(own.mean()-oth.mean())
        if ds: perseed_diag.append(np.mean(ds))
    perseed_diag=np.array(perseed_diag)
    t,p=stats.ttest_1samp(perseed_diag,0)
    print(f"   within-agent own-minus-other (avg over 4 personas), per seed: mean={perseed_diag.mean():+.3f} t={t:.2f} p={p:.4f} signs +:{(perseed_diag>0).sum()}/{len(perseed_diag)}")
    # per-persona pooled
    for a,t in PT.items():
        own=df[(df.agent_id==a)&(df.task_type==t)].success
        oth=df[(df.agent_id==a)&(df.task_type!=t)].success
        if len(own)>=10:
            print(f"      agent{a}={t:7s}: own={own.mean():.2f}(n{len(own)}) other={oth.mean():.2f}(n{len(oth)}) within-diff={own.mean()-oth.mean():+.2f}")
