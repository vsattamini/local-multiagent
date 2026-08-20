"""
ATTACK 4b: taxonomy robustness, extended with the tie-break rule of the labeller.

Supersedes audit/hostile_4_taxonomy.py: it reproduces the 8 partitions of that script
and adds 2 more that attack the ONE undisclosed degree of freedom in
scripts/categorize_tasks.py -- the tie-break.

  classify_task() scores each of the 5 keyword buckets and returns
  `max(scores, key=scores.get)`. On a tie, Python's max returns the FIRST key in
  dict insertion order: string > math > list > logic > search. That rule is not
  documented anywhere, and it fires on 26 of the 114 auto-labelled HumanEval
  problems. In those 26 ties, `logic` was tied for first 10 times and lost all 10;
  `search` was tied twice and lost both -- which is why `logic` has n=10 and
  `search` is empty.

Added partitions:
  tiebreak_reversed : same scores, ties resolved by the LAST co-winner instead of
                      the first (logic 10 -> 20, search 0 -> 2, 26 labels move).
  drop_ties         : the 26 tied problems are dropped entirely (n = 138).

Same estimand as hostile_4: task-as-unit one-way ANOVA on the per-problem
advantage of the 3B agent over the 1.5B agents, run `het_swarm_ensemble`
(2x1.5B + 1x3B, HumanEval, ensemble mode) = line L1 of thesis table tab:ec4b.
Holm correction over the full family, since we are explicitly fishing.

Read-only. Run from the repo root:
    .venv/bin/python audit/hostile_4b_taxonomy_ties.py
"""
import sys, os, json, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _hostile_loader import load_run, humaneval_prompts, het_model_of
import numpy as np, pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

df = load_run("het_swarm_ensemble")
df["model"] = df.agent_id.apply(het_model_of)
rate = df.groupby(["task_id", "model"]).success.mean().unstack()
adv = rate["3b"] - rate["1.5b"]
orig_type = df.groupby("task_id").task_type.first()
hp = humaneval_prompts()


def anova(labels, adv):
    s = pd.Series(labels)
    valid = s.notna() & adv.notna()
    grp = [adv[valid][s[valid] == c].values for c in s[valid].unique()]
    grp = [x for x in grp if len(x) >= 3]
    if len(grp) < 2:
        return None
    F, p = stats.f_oneway(*grp)
    sizes = {str(c): int((s[valid] == c).sum()) for c in s[valid].unique()}
    return F, p, sizes


tests = {}

# --- the 8 partitions of hostile_4 -------------------------------------------
tests["orig_4way"] = anova(orig_type, adv)
m = orig_type != "logic"
tests["drop_logic_3way"] = anova(orig_type[m], adv[m])


def op_label(tid):
    item = hp.get(tid, {})
    s = ((item.get("canonical_solution") or "") + " " + item.get("prompt", "")).lower()
    if re.search(r'\bre\.|regex|\.split|\.join|\.replace|str\(|\.lower|\.upper|char', s):
        return "stringops"
    if re.search(r'%|//|\*\*|math\.|sqrt|prime|divis|factor|sum\(|abs\(', s):
        return "arith"
    if re.search(r'sorted|\.sort|\bfor .* in |\.append|list\(|\[.*for', s):
        return "iter"
    return "control"


tests["solution_ops_4way"] = anova(pd.Series({t: op_label(t) for t in adv.index}), adv)
loop = pd.Series({t: ("loop" if (("for " in (hp.get(t, {}).get("canonical_solution") or ""))
                                 or ("while " in (hp.get(t, {}).get("canonical_solution") or "")))
                      else "noloop") for t in adv.index})
tests["loop_vs_noloop"] = anova(loop, adv)
tests["reason_vs_manip"] = anova(
    orig_type.map({"math": "reason", "logic": "reason", "string": "manip", "list": "manip"}), adv)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

texts = [hp.get(t, {}).get("prompt", "") for t in adv.index]
X = TfidfVectorizer(max_features=200, stop_words="english").fit_transform(texts)
for k in [3, 4, 5]:
    km = KMeans(n_clusters=k, random_state=0, n_init=10).fit(X)
    tests[f"tfidf_kmeans_k{k}"] = anova(pd.Series(km.labels_, index=adv.index), adv)

# --- the 2 new tie-break partitions ------------------------------------------
# re-derive the labeller's scores without importing datasets (offline)
_src = open(os.path.join(ROOT, "scripts", "categorize_tasks.py")).read()
_ns = {"__file__": os.path.join(ROOT, "scripts", "categorize_tasks.py")}
exec(_src.split("def main(")[0].replace("from datasets import load_dataset", ""), _ns)
MANUAL, KW = _ns["MANUAL_CATEGORIZATION"], _ns["CATEGORY_KEYWORDS"]

rows = [json.loads(l) for l in open(os.path.join(ROOT, "data", "HumanEval.jsonl")) if l.strip()]
first, last, tied, fallback = {}, {}, set(), []
for it in rows:
    tid = it["task_id"]
    if tid in MANUAL:
        first[tid] = last[tid] = MANUAL[tid]
        continue
    text = (it["prompt"] + " " + it["entry_point"]).lower()
    sc = {c: sum(1 for k in kws if k in text) for c, kws in KW.items()}
    mx = max(sc.values())
    if mx == 0:
        first[tid] = last[tid] = "logic"
        fallback.append(tid)
        continue
    w = [c for c in KW if sc[c] == mx]          # dict order == declaration order
    first[tid], last[tid] = w[0], w[-1]
    if len(w) > 1:
        tied.add(tid)

stored = json.load(open(os.path.join(ROOT, "data", "humaneval_categories_full.json")))
assert all(stored[t] == first[t] for t in stored), "re-derivation does not reproduce the stored labels"

tests["tiebreak_reversed"] = anova(pd.Series(last).reindex(adv.index), adv)
tests["drop_ties"] = anova(
    pd.Series({t: (first[t] if t not in tied else np.nan) for t in adv.index}), adv)

# --- report ------------------------------------------------------------------
print(f"re-derivation reproduces data/humaneval_categories_full.json exactly: 164/164")
print(f"fallback clause fired on {len(fallback)} of 114 auto-labelled problems")
print(f"ties broken by declaration order: {len(tied)} of 114")
lost_logic = sum(1 for t in tied if last[t] == "logic" or "logic" in (first[t], last[t]) and first[t] != "logic")
print(f"  ties in which logic was a co-winner and lost: "
      f"{sum(1 for t in tied if first[t] != 'logic' and last[t] == 'logic')}")
print()
names = list(tests)
ps = [tests[n][1] for n in names]
rej, padj, _, _ = multipletests(ps, method="holm")
print(f"{'partition':20s} {'F':>7s} {'raw_p':>8s} {'adj_p':>8s}  sizes")
for n, a in zip(names, padj):
    F, p, sz = tests[n]
    print(f"{n:20s} {F:7.2f} {p:8.4f} {a:8.4f}  {sz}")
print(f"\nHolm over all {len(ps)} partitions: min adj-p = {padj.min():.4f}   any significant: {bool(rej.any())}")
rej8, padj8, _, _ = multipletests(ps[:8], method="holm")
print(f"Holm over the 8 partitions of hostile_4: min adj-p = {padj8.min():.4f}  any significant: {bool(rej8.any())}")
