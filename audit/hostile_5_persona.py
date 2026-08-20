"""
ATTACK 5: PERSONA runs (mech_*persona) explicitly prompt agent k as a {string,math,list,logic}
specialist. This is the BEST shot at type-based success specialization. Question: does the
persona-matched type get a SUCCESS boost for its assigned agent, beyond chance, clustered by task?
For affinity-persona, also: does the persona agent OUTPERFORM peers on its assigned type?
Honest permutation: shuffle success within task_type (preserves per-type difficulty & per-agent loads).
Statistic: DIAGONAL boost = mean over agents of (agent k success on persona-type k) minus
(agent k success on other types) -- the specialization the personas were designed to create.
"""
import sys, os, json, glob
import numpy as np, pandas as pd
from scipy import stats

ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def load_ms(name, base="results_multiseed"):
    rows=[]
    for sd in sorted(glob.glob(os.path.join(ROOT,base,name,"seed_*"))):
        if not os.path.exists(os.path.join(sd,"final_metrics.json")): continue
        seed=os.path.basename(sd).replace("seed_","")
        f=os.path.join(sd,"task_log.jsonl")
        if not os.path.exists(f): continue
        for l in open(f):
            d=json.loads(l); d["seed"]=seed; rows.append(d)
    df=pd.DataFrame(rows)
    if len(df): df["success"]=df["success"].astype(int)
    return df

PERSONA_TYPE={0:"string",1:"math",2:"list",3:"logic"}

def interaction_perm(df, nperm=3000, rng=None):
    """generic agent x type interaction SS, permuting success within task_type."""
    sub=df[["agent_id","task_type","success"]].copy()
    a=pd.Categorical(sub.agent_id); t=pd.Categorical(sub.task_type)
    ai=a.codes; ti=t.codes; y=sub.success.values.astype(float)
    na=ai.max()+1; nt=ti.max()+1; flat=ai*nt+ti
    def stat(yv):
        cs=np.bincount(flat,yv,minlength=na*nt).reshape(na,nt)
        cn=np.bincount(flat,minlength=na*nt).reshape(na,nt).astype(float)
        with np.errstate(invalid="ignore",divide="ignore"): g=cs/cn
        am=np.nansum(cs,1)/np.nansum(cn,1); tm=np.nansum(cs,0)/np.nansum(cn,0); gm=yv.mean()
        resid=g-am[:,None]-tm[None,:]+gm
        return np.nansum(cn*np.where(cn>0,resid,0)**2)/cn.sum()
    obs=stat(y)
    order=np.argsort(ti,kind="stable"); ts=ti[order]; b=np.searchsorted(ts,np.arange(nt+1))
    null=np.empty(nperm)
    for k in range(nperm):
        yp=y.copy()
        for j in range(nt):
            idx=order[b[j]:b[j+1]]; yp[idx]=rng.permutation(y[idx])
        null[k]=stat(yp)
    return obs,(np.sum(null>=obs)+1)/(nperm+1)

def diagonal_boost(df, nperm=3000, rng=None):
    """The designed effect: agent k does better on persona-type k than on other types,
    relative to how OTHER agents do on type k. = mean diagonal residual of the agentxtype rate matrix."""
    cats=["string","math","list","logic"]
    sub=df[df.agent_id.isin(PERSONA_TYPE)].copy()
    ti=pd.Categorical(sub.task_type, categories=cats).codes
    ai=sub.agent_id.values; y=sub.success.values.astype(float)
    nt=4; na=4; flat=ai*nt+ti
    def diag(yv):
        cs=np.bincount(flat,yv,minlength=na*nt).reshape(na,nt)
        cn=np.bincount(flat,minlength=na*nt).reshape(na,nt).astype(float)
        with np.errstate(invalid="ignore",divide="ignore"): g=cs/cn
        am=np.nansum(cs,1)/np.nansum(cn,1); tm=np.nansum(cs,0)/np.nansum(cn,0); gm=yv.mean()
        resid=g-am[:,None]-tm[None,:]+gm
        return np.nanmean(np.diag(resid))   # avg specialization on own persona-type
    obs=diag(y)
    order=np.argsort(ti,kind="stable"); ts=ti[order]; b=np.searchsorted(ts,np.arange(nt+1))
    null=np.empty(nperm)
    for k in range(nperm):
        yp=y.copy()
        for j in range(nt):
            idx=order[b[j]:b[j+1]]; yp[idx]=rng.permutation(y[idx])
        null[k]=diag(yp)
    p=(np.sum(null>=obs)+1)/(nperm+1)
    return obs, null.mean(), p

rng=np.random.default_rng(0)
print("=== PERSONA runs: designed diagonal (own-type) success boost, perm-null within type ===")
for r in ["mech_1_5b_persona","mech_3b_persona","mech_1_5b_rrpersona","mech_3b_rrpersona","mech_7b_rrpersona"]:
    df=load_ms(r)
    if not len(df): print(f"{r}: empty"); continue
    o,nm,p=diagonal_boost(df, rng=rng)
    oi,pi=interaction_perm(df, rng=rng)
    print(f"{r:20s} seeds={df.seed.nunique()} | diag_boost={o:+.4f} (null {nm:+.4f}) one-sided p={p:.4f} | full-interaction p={pi:.4f}")
    # show diagonal cells
    piv=df[df.agent_id.isin(PERSONA_TYPE)].groupby(["agent_id","task_type"]).success.mean().unstack()
    diag=[(PERSONA_TYPE[a], round(piv.loc[a,PERSONA_TYPE[a]],2) if PERSONA_TYPE[a] in piv.columns and a in piv.index else None) for a in PERSONA_TYPE]
    print(f"     own-type success per persona-agent: {diag}")
