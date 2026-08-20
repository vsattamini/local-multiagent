"""
ATTACK 4: Push the het logic dip (p~0.056-0.078). Threats to the null:
 (a) logic is a fallback residual -> maybe a CLEANER taxonomy makes a real interaction appear.
 (b) maybe dropping the fallback-logic problems sharpens string/math/list into significance.
We test several alternative problem partitions and run the task-as-unit ANOVA on 3B advantage,
applying multiplicity correction (we are explicitly fishing here).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _hostile_loader import load_run, humaneval_prompts, het_model_of
import numpy as np, pandas as pd
from scipy import stats
import re

df=load_run("het_swarm_ensemble"); df["model"]=df.agent_id.apply(het_model_of)
rate=df.groupby(["task_id","model"]).success.mean().unstack()
adv=(rate["3b"]-rate["1.5b"]);
orig_type=df.groupby("task_id").task_type.first()
hp=humaneval_prompts()

def anova(labels, adv, name):
    s=pd.Series(labels); valid=s.notna()&adv.notna()
    grp=[adv[valid][s[valid]==c].values for c in s[valid].unique()]
    grp=[x for x in grp if len(x)>=3]
    if len(grp)<2: return None
    F,p=stats.f_oneway(*grp)
    sizes={c:int((s[valid]==c).sum()) for c in s[valid].unique()}
    return F,p,sizes

tests={}

# (a) original 4-way (baseline)
tests["orig_4way"]=anova(orig_type, adv, "orig")

# (b) drop fallback-logic, 3-way
m=orig_type!="logic"
tests["drop_logic_3way"]=anova(orig_type[m], adv[m], "drop_logic")

# (c) re-derive cleaner taxonomy from canonical solution operations
def op_label(tid):
    item=hp.get(tid,{}); sol=(item.get("canonical_solution") or "")+ " "+item.get("prompt","")
    s=sol.lower()
    if re.search(r'\bre\.|regex|\.split|\.join|\.replace|str\(|\.lower|\.upper|char', s): return "stringops"
    if re.search(r'%|//|\*\*|math\.|sqrt|prime|divis|factor|sum\(|abs\(', s): return "arith"
    if re.search(r'sorted|\.sort|\bfor .* in |\.append|list\(|\[.*for', s): return "iter"
    return "control"
oplab=pd.Series({tid:op_label(tid) for tid in adv.index})
tests["solution_ops_4way"]=anova(oplab, adv, "ops")

# (d) binary: needs-loop vs no-loop (control-flow axis)
loop=pd.Series({tid:("loop" if (("for " in (hp.get(tid,{}).get("canonical_solution") or "")) or ("while " in (hp.get(tid,{}).get("canonical_solution") or ""))) else "noloop") for tid in adv.index})
tests["loop_vs_noloop"]=anova(loop, adv, "loop")

# (e) merge math+logic (both 'reasoning') vs string+list ('manipulation')
merge=orig_type.map({"math":"reason","logic":"reason","string":"manip","list":"manip"})
tests["reason_vs_manip"]=anova(merge, adv, "merge2")

# (f) kmeans on prompt embeddings? skip (no GPU/model); use bag-of-words clustering instead
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
texts=[hp.get(tid,{}).get("prompt","") for tid in adv.index]
X=TfidfVectorizer(max_features=200, stop_words="english").fit_transform(texts)
for k in [3,4,5]:
    km=KMeans(n_clusters=k,random_state=0,n_init=10).fit(X)
    tests[f"tfidf_kmeans_k{k}"]=anova(pd.Series(km.labels_,index=adv.index), adv, f"tfidf{k}")

print(f"{'partition':22s} {'F':>7s} {'raw_p':>8s} {'sig?':>5s}  sizes")
raw=[]
for k,v in tests.items():
    if v is None: print(f"{k:22s}  not computable"); continue
    F,p,sz=v; raw.append(p)
    print(f"{k:22s} {F:7.2f} {p:8.4f} {'*' if p<0.05 else '':>5s}  {sz}")
from statsmodels.stats.multitest import multipletests
rej,padj,_,_=multipletests([v[1] for v in tests.values() if v], method="holm")
print(f"\nHolm-corrected across {len(padj)} partitions: min adj-p = {padj.min():.4f}  any significant: {rej.any()}")
