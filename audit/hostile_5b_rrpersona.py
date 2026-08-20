"""
ATTACK 5b: The rrpersona interaction (p=0.0003) is significant but NEGATIVE-diagonal
(logic-persona agent tanks on logic). Threats:
 - is it again the tiny logic fallback cell? -> drop logic, recompute interaction
 - does it GENERALIZE (string/math/list personas helping their own type) or is it ONE bad agent?
 - per-type: does the persona agent beat OR lose to peers on its own type, with adequate n?
This determines whether personas produced REAL (even if harmful) type-conditioned behavior
= a genuine agent x type effect the success-rate test would otherwise miss.
"""
import sys, os, json, glob
import numpy as np, pandas as pd
from scipy import stats
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def load_ms(name):
    rows=[]
    for sd in sorted(glob.glob(os.path.join(ROOT,"results_multiseed",name,"seed_*"))):
        if not os.path.exists(os.path.join(sd,"final_metrics.json")): continue
        for l in open(os.path.join(sd,"task_log.jsonl")):
            d=json.loads(l); d["seed"]=os.path.basename(sd)[5:]; rows.append(d)
    df=pd.DataFrame(rows); df["success"]=df["success"].astype(int); return df
PT={0:"string",1:"math",2:"list",3:"logic"}

def interaction_perm(df, types, nperm=3000, rng=None):
    sub=df[df.task_type.isin(types)][["agent_id","task_type","success"]].copy()
    a=pd.Categorical(sub.agent_id); t=pd.Categorical(sub.task_type)
    ai=a.codes; ti=t.codes; y=sub.success.values.astype(float)
    na=ai.max()+1; nt=ti.max()+1; flat=ai*nt+ti
    def stat(yv):
        cs=np.bincount(flat,yv,minlength=na*nt).reshape(na,nt)
        cn=np.bincount(flat,minlength=na*nt).reshape(na,nt).astype(float)
        with np.errstate(invalid="ignore",divide="ignore"): g=cs/cn
        am=np.nansum(cs,1)/np.nansum(cn,1); tm=np.nansum(cs,0)/np.nansum(cn,0); gm=yv.mean()
        return np.nansum(cn*np.where(cn>0,(g-am[:,None]-tm[None,:]+gm),0)**2)/cn.sum()
    obs=stat(y); order=np.argsort(ti,kind="stable"); ts=ti[order]; b=np.searchsorted(ts,np.arange(nt+1))
    null=np.empty(nperm)
    for k in range(nperm):
        yp=y.copy()
        for j in range(nt):
            idx=order[b[j]:b[j+1]]; yp[idx]=rng.permutation(y[idx])
        null[k]=stat(yp)
    return obs,(np.sum(null>=obs)+1)/(nperm+1)

rng=np.random.default_rng(0)
for r in ["mech_3b_rrpersona","mech_7b_rrpersona","mech_1_5b_rrpersona"]:
    df=load_ms(r)
    print(f"\n===== {r} =====")
    # cell sizes per (persona-agent, own type)
    piv_n=df.groupby(["agent_id","task_type"]).success.count().unstack()
    piv_m=df.groupby(["agent_id","task_type"]).success.mean().unstack()
    print("n per cell:\n", piv_n.to_string())
    # logic-persona agent (3) on logic vs other agents on logic
    if "logic" in piv_m.columns:
        lg=df[df.task_type=="logic"].groupby("agent_id").success.agg(["mean","count"])
        print("logic cell by agent (mean,n):", {int(a):(round(r2['mean'],2),int(r2['count'])) for a,r2 in lg.iterrows()})
    # full interaction with all types vs DROPPING logic
    o_all,p_all=interaction_perm(df,["string","math","list","logic"],rng=rng)
    o_nl,p_nl =interaction_perm(df,["string","math","list"],rng=rng)
    print(f"interaction WITH logic p={p_all:.4f}  |  DROP logic (string/math/list only) p={p_nl:.4f}")
    # generalization: does each NON-logic persona agent beat peers on its OWN type? (n>=15)
    print("own-type vs peers (excl logic):")
    for a in [0,1,2]:
        t=PT[a]; own=df[(df.task_type==t)&(df.agent_id==a)].success; peer=df[(df.task_type==t)&(df.agent_id!=a)].success
        if len(own)>=15 and len(peer)>=15:
            diff=own.mean()-peer.mean()
            print(f"   agent{a} ({t}): own={own.mean():.2f}(n{len(own)}) peer={peer.mean():.2f}(n{len(peer)}) diff={diff:+.2f}")
