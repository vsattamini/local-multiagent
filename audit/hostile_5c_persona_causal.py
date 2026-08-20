"""
ATTACK 5c: Is the rrpersona interaction (p=0.0003, survives dropping logic) a REAL
persona-induced agent x type effect, or a generic round-robin routing artifact?
Test 1: same interaction stat on NON-persona round-robin controls (ens_3b_n3, ens_7b, het):
        these are round-robin/ensemble, no personas. If THEY show the same interaction, it's
        routing, not personas.
Test 2: persona DIAGONAL contrast across model sizes -- if personas cause it, the pattern
        (own-type penalty/bonus) should be consistent and persona-aligned.
Test 3: the mech_1_5b_rrpersona string agent +0.23 -- per-seed sign test, is it robust?
Test 4: cross-run: aggregate the 3 rrpersona runs and test the persona-diagonal directly
        (does prompting agent k as type-k specialist change its type-k success vs a no-persona baseline?)
"""
import sys, os, json, glob
import numpy as np, pandas as pd
from scipy import stats
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def load(name, base):
    rows=[]
    for sd in sorted(glob.glob(os.path.join(ROOT,base,name,"seed_*"))):
        if not os.path.exists(os.path.join(sd,"final_metrics.json")): continue
        for l in open(os.path.join(sd,"task_log.jsonl")):
            d=json.loads(l); d["seed"]=os.path.basename(sd)[5:]; rows.append(d)
    df=pd.DataFrame(rows)
    if len(df): df["success"]=df["success"].astype(int)
    return df
PT={0:"string",1:"math",2:"list",3:"logic"}

def interaction_perm(df, nperm=3000, rng=None):
    sub=df[["agent_id","task_type","success"]].copy()
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
    null=np.array([_perm(stat,y,order,b,nt,rng) for _ in range(nperm)])
    return obs,(np.sum(null>=obs)+1)/(nperm+1)
def _perm(stat,y,order,b,nt,rng):
    yp=y.copy()
    for j in range(nt):
        idx=order[b[j]:b[j+1]]; yp[idx]=rng.permutation(y[idx])
    return stat(yp)

rng=np.random.default_rng(0)
print("=== TEST 1: interaction on NON-persona round-robin/ensemble controls (should be NULL) ===")
for name,base in [("ens_3b_n3","results_phase3"),("ens_7b","results_phase3"),("het_swarm_ensemble","results_phase3")]:
    df=load(name,base)
    o,p=interaction_perm(df,rng=rng)
    print(f"  {name:20s} interaction perm p={p:.4f}")

print("\n=== TEST 4: persona diagonal vs matched NON-persona baseline (same model, affinity) ===")
# compare own-type success of persona-agent vs the population's own-type success in a baseline run
pairs=[("mech_3b_rrpersona","ens_3b_n3"),("mech_7b_rrpersona","ens_7b")]
for pr,base in pairs:
    dp=load(pr,"results_multiseed"); db=load(base,"results_phase3")
    print(f"\n  {pr} vs baseline {base}:")
    for a in [0,1,2,3]:
        t=PT[a]
        own=dp[(dp.agent_id==a)&(dp.task_type==t)].success
        # baseline: ALL agents on type t (no persona) = the no-persona reference rate for type t
        ref=db[db.task_type==t].success
        if len(own)>=10 and len(ref)>=10:
            # 2-proportion z
            z,pz=stats.ttest_ind(own, ref, equal_var=False)
            print(f"    persona-agent{a}={t:7s}: own={own.mean():.2f}(n{len(own)}) vs no-persona ref={ref.mean():.2f}(n{len(ref)})  diff={own.mean()-ref.mean():+.2f} p={pz:.3f}")

print("\n=== TEST 3: mech_1_5b_rrpersona string-persona agent +0.23, per-seed robustness ===")
df=load("mech_1_5b_rrpersona","results_multiseed")
diffs=[]
for sd,g in df.groupby("seed"):
    own=g[(g.agent_id==0)&(g.task_type=="string")].success
    peer=g[(g.agent_id!=0)&(g.task_type=="string")].success
    if len(own)>0 and len(peer)>0: diffs.append(own.mean()-peer.mean())
diffs=np.array(diffs)
t,p=stats.ttest_1samp(diffs,0);
print(f"  per-seed (own-peer) string diff: mean={diffs.mean():+.3f} t={t:.2f} two-sided p={p:.4f} (n={len(diffs)} seeds), signs +:{(diffs>0).sum()}/-:{(diffs<0).sum()}")
# but is it specialization or just agent0 being a generally-better draw? check agent0 on OTHER types
o_other=df[(df.agent_id==0)&(df.task_type!="string")].success
o_str=df[(df.agent_id==0)&(df.task_type=="string")].success
print(f"  agent0: string={o_str.mean():.2f}(n{len(o_str)}) vs its-own-other-types={o_other.mean():.2f}(n{len(o_other)})  -> if string>>other, real own-type boost")
