"""
ATTACK 2: Single-assignment affinity runs (popN_*, mbpp_*). Under affinity routing each
task goes to ONE agent, so roles COULD form. Two questions:
 (a) ATTEMPT niches: do agents specialize in WHICH types they attempt (vs RandomRouter)? [routing]
 (b) SUCCESS niches: does an agent's per-type SUCCESS rate exceed peers beyond chance,
     clustered by task? [genuine competence differentiation]
For (b), under single-assignment we can't compare agents on the SAME task. Test instead:
  is there an agent x type interaction in success, using a logit with task_id cluster-robust SE?
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _hostile_loader import load_run
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

def attempt_specialization(df):
    """S-like: per agent, distribution over types attempted. Report agent x type chi2 on ATTEMPTS."""
    ct = pd.crosstab(df.agent_id, df.task_type)
    chi2,p,dof,_ = stats.chi2_contingency(ct)
    n = ct.values.sum(); v = np.sqrt(chi2/(n*(min(ct.shape)-1)))
    return chi2,p,v,ct

def success_interaction_clustered(df):
    """agent x type interaction in SUCCESS, cluster-robust by task_id."""
    d = df.copy()
    d["agent_id"] = d["agent_id"].astype(str)
    d["task_type"] = d["task_type"].astype(str)
    # need both factors with >1 level and reasonable cells
    try:
        m_full = smf.logit("success ~ C(agent_id)*C(task_type)", data=d).fit(disp=0)
        m_add  = smf.logit("success ~ C(agent_id)+C(task_type)", data=d).fit(disp=0)
        lr = 2*(m_full.llf - m_add.llf)
        ddof = m_full.df_model - m_add.df_model
        p_lr = stats.chi2.sf(lr, ddof)
    except Exception as e:
        return None
    # cluster-robust Wald on interaction terms via GEE-like: use cluster cov on the full OLS-LPM
    d2 = d.copy()
    mlpm = smf.ols("success ~ C(agent_id)*C(task_type)", data=d2).fit(
        cov_type="cluster", cov_kwds={"groups": d2["task_id"]})
    inter = [t for t in mlpm.params.index if ":" in t]
    if inter:
        R = np.zeros((len(inter), len(mlpm.params)))
        idx = {n:i for i,n in enumerate(mlpm.params.index)}
        for i,t in enumerate(inter): R[i, idx[t]] = 1
        wald = mlpm.f_test(R)
        p_clu = float(wald.pvalue)
    else:
        p_clu = np.nan
    return p_lr, p_clu, len(inter)

runs = ["popN_8","popN_12","popN_16","mbpp_1.5b_baseline","mbpp_1.5b_lowtemp",
        "mbpp_3b_baseline","mbpp_3b_lowtemp","rand_1.5b","rand_3b","rand_7b","greedy_3b"]
print(f"{'run':22s} {'router/temp':10s} | ATTEMPT chi2  p        V  | SUCCESS-interaction  p_LR(pooled)  p_cluster(task)")
for r in runs:
    df = load_run(r)
    if not len(df):
        print(f"{r:22s} EMPTY"); continue
    import json,glob
    cfg = json.load(open(sorted(glob.glob(f"results_phase3/{r}/seed_*/config.json"))[0]))
    rt = f"{cfg.get('router_type')[:6]}/{cfg.get('router_temperature')}"
    chi2,pa,v,ct = attempt_specialization(df)
    si = success_interaction_clustered(df)
    if si:
        p_lr,p_clu,ni = si
        print(f"{r:22s} {rt:10s} | {chi2:8.1f} {pa:7.4f} {v:.3f} | p_LR={p_lr:.4f}  p_cluster={p_clu:.4f} ({ni} terms)")
    else:
        print(f"{r:22s} {rt:10s} | {chi2:8.1f} {pa:7.4f} {v:.3f} | success-interaction not computable")
