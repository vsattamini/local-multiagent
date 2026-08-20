"""
ATTACK 1b: De-circularize the difficulty finding and probe whether it is
(a) an artifact of defining difficulty from the same rates, and
(b) just per-model competence (a SCALING fact) vs a genuine interaction the type labels miss.
Hostile self-check.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _hostile_loader import load_run, het_model_of
import numpy as np, pandas as pd
from scipy import stats

df = load_run("het_swarm_ensemble")
df["model"] = df.agent_id.apply(het_model_of)

# Per-task per-model rate (over seeds + over the 2 small agents)
rate = df.groupby(["task_id","model"]).success.mean().unstack()
small, big = rate["1.5b"], rate["3b"]
adv = big - small

# ---- DE-CIRCULARIZE difficulty: define hardness ONLY from the 1.5B (predictor),
#      then test whether the 3B's rate (outcome) rises faster -> this is just
#      a learning-curve / competence-by-difficulty relation. Use small-only hardness.
small_hard = 1 - small
r,p = stats.spearmanr(small_hard, adv)
print(f"[de-circular] adv vs (1.5B-only hardness): spearman r={r:+.3f} p={p:.4g}")
print("  Interpretation: positive => 3B gains most where the 1.5B struggles. Pure competence gap, not roles.")

# Is the 3B advantage just monotone competence? Check: where 1.5B already solves it,
# does the 3B add anything? (no role to play). Where 1.5B fails, 3B rescues.
band = pd.cut(small, [-.01,.25,.75,1.01], labels=["1.5B-fails","mid","1.5B-solves"])
print("\n3B advantage by 1.5B-competence band:")
for b in band.cat.categories:
    a = adv[band==b]
    print(f"  {b:12s} n={len(a):3d}  mean 3B adv={a.mean():+.3f}")

# ---- KEY HOSTILE TEST: is the difficulty-scaling ITSELF emergent (multi-agent) or
#      just single-model competence? Compare to the homogeneous control ens_3b_n3:
#      there is no 'big vs small', so build a pseudo-advantage = best agent - worst agent
#      per task and see if IT scales with difficulty too. If yes, the het difficulty
#      pattern is generic competence-spread, present even among identical agents.
print("\n=== Does difficulty-scaling appear among IDENTICAL agents (control)? ===")
for ctl in ["ens_3b_n3","ens_3b"]:
    c = load_run(ctl)
    rr = c.groupby(["task_id","agent_id"]).success.mean().unstack()
    mean_rate = rr.mean(axis=1)
    spread = rr.max(axis=1) - rr.min(axis=1)   # identical-agent spread (noise)
    hard = 1 - mean_rate
    r2,p2 = stats.spearmanr(hard, spread)
    print(f"  {ctl}: identical-agent max-min spread vs hardness  r={r2:+.3f} p={p2:.4g}  (mean spread {spread.mean():.3f})")

# ---- Net-of-difficulty type test: does TYPE still matter once difficulty is controlled? ----
print("\n=== Type effect on adv, NET of difficulty (OLS, task as unit) ===")
import statsmodels.formula.api as smf
ttype = df.groupby("task_id").task_type.first()
dat = pd.DataFrame({"adv":adv, "type":ttype, "hard": (1-small)}).dropna()
m0 = smf.ols("adv ~ C(type)", data=dat).fit()
m1 = smf.ols("adv ~ hard + C(type)", data=dat).fit()
print(f"  type-only R2={m0.rsquared:.3f}  F-pval(type)={m0.f_pvalue:.4f}")
# partial F for type after hard
from statsmodels.stats.anova import anova_lm
m_hardonly = smf.ols("adv ~ hard", data=dat).fit()
ff = anova_lm(m_hardonly, m1)
print(f"  adding type after hardness: partial F p = {ff['Pr(>F)'].iloc[1]:.4f}  (does type survive difficulty control?)")
print(f"  hardness alone R2={m_hardonly.rsquared:.3f}")
