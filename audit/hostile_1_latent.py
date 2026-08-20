"""
ATTACK 1: Latent-axis specialization in the heterogeneous swarm.
The 4 type labels are coarse/leaky (math=keyword 'number/sum/digit', logic=residual).
Build per-problem 3B-vs-1.5B advantage and hunt for ANY partition of the 164 problems on
which the 3B's edge varies in a way the type labels miss -- tested with TASK as the unit
(cluster-robust), since seeds reuse the same 164 problems (pseudo-replication otherwise).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _hostile_loader import load_run, humaneval_prompts, ast_size, het_model_of
import numpy as np, pandas as pd
from scipy import stats

df = load_run("het_swarm_ensemble")
df["model"] = df.agent_id.apply(het_model_of)

# Per (task_id, model) success rate, averaged over the 2x1.5B agents and over seeds.
# This collapses to ONE row per task = the correct unit.
g = df.groupby(["task_id", "model"]).success.mean().unstack()  # cols: 1.5b, 3b
g["adv"] = g["3b"] - g["1.5b"]          # 3B advantage per problem
g["mean_rate"] = (g["3b"] + g["1.5b"]) / 2
ttype = df.groupby("task_id").task_type.first()
g["type"] = ttype
print("=== n problems:", len(g), "  mean 3B adv:", round(g.adv.mean(),3))

# ---------- baseline: reproduce the SETTLED type-interaction null (task as unit) ----------
groups = [g.adv[g.type==t].values for t in ["string","math","list","logic"]]
F,p = stats.f_oneway(*groups)
print(f"\n[REPRO] adv ~ type, task-as-unit ANOVA: F={F:.3f} p={p:.4f}  (settled ~0.056)")
print("  per-type mean 3B adv:", {t: round(g.adv[g.type==t].mean(),3) for t in ["string","math","list","logic"]})

# ---------- LATENT AXES ----------
hp = humaneval_prompts()
feat = {}
for tid in g.index:
    item = hp.get(tid, {})
    prompt = item.get("prompt","")
    sol = item.get("canonical_solution","") or ""
    code = prompt + sol
    feat[tid] = {
        "prompt_len": len(prompt),
        "prompt_lines": prompt.count("\n"),
        "ast_size": ast_size(prompt + "\n" + sol) if sol else ast_size(prompt),
        "n_examples": prompt.count(">>>") + prompt.lower().count("example"),
        "has_loop": int(("for " in sol) or ("while " in sol)),
        "has_import": int("import " in prompt or "import " in sol),
        "sol_len": len(sol),
        "difficulty": 1 - g.loc[tid,"mean_rate"],  # empirical hardness
    }
F = pd.DataFrame(feat).T
g2 = g.join(F)

print("\n=== LATENT-AXIS PARTITIONS: does 3B advantage vary across the axis? (task as unit) ===")
def split_test(col, label):
    x = g2[col].astype(float)
    valid = x.notna() & g2.adv.notna()
    x, adv = x[valid], g2.adv[valid]
    if x.nunique() < 3:
        med = x.median(); hi = adv[x>med]; lo = adv[x<=med]
        t,p = stats.ttest_ind(hi,lo,equal_var=False)
        r,pr = stats.pointbiserialr((x>med).astype(int), adv)
        print(f"  {label:14s} median-split  hi_adv={hi.mean():.3f} lo_adv={lo.mean():.3f}  t={t:.2f} p={p:.4f}  r={r:.2f}")
        return p
    # continuous: correlation of adv with axis  + tercile ANOVA
    r,pr = stats.spearmanr(x, adv)
    terc = pd.qcut(x, 3, labels=False, duplicates="drop")
    groups = [adv[terc==k] for k in sorted(pd.unique(terc.dropna()))]
    if len(groups) >= 2:
        Fa,pa = stats.f_oneway(*groups)
    else:
        Fa,pa = np.nan,np.nan
    print(f"  {label:14s} spearman r={r:+.3f} p={pr:.4f} | tercile ANOVA F={Fa:.2f} p={pa:.4f}")
    return min(pr, pa if not np.isnan(pa) else 1)

ps = {}
for col,lab in [("difficulty","difficulty"),("prompt_len","prompt_len"),
                ("ast_size","ast_size"),("sol_len","sol_len"),
                ("n_examples","n_examples"),("has_loop","has_loop"),
                ("has_import","stdlib_import")]:
    ps[lab] = split_test(col, lab)

# ---------- k-means on solve-pattern (per-task agent success vector) ----------
print("\n=== K-MEANS on per-agent solve pattern, then test 3B adv across clusters ===")
from sklearn.cluster import KMeans
# feature = per-agent (3 agents) mean success over seeds for each task
piv = df.groupby(["task_id","agent_id"]).success.mean().unstack().fillna(0)
for k in [2,3,4]:
    km = KMeans(n_clusters=k, random_state=0, n_init=10).fit(piv.values)
    lab = pd.Series(km.labels_, index=piv.index)
    adv = g.adv.reindex(piv.index)
    groups = [adv[lab==c].values for c in range(k)]
    groups = [x for x in groups if len(x)>1]
    if len(groups)>=2:
        Fa,pa = stats.f_oneway(*groups)
        print(f"  k={k}: cluster sizes {[int((lab==c).sum()) for c in range(k)]}  3B-adv-by-cluster ANOVA F={Fa:.2f} p={pa:.4f}")

# Bonferroni note
import numpy as np
m = len([x for x in ps.values() if x is not None])
best = min(ps.values())
print(f"\n[SUMMARY] {m} latent axes tested. Smallest raw p = {best:.4f}; Bonferroni x{m} = {min(1,best*m):.4f}")
print("axis raw-p:", {k: round(v,4) for k,v in sorted(ps.items(), key=lambda kv: kv[1])})
