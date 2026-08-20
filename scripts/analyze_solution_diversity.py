#!/usr/bin/env python3
"""
GAP 2 — Do agents differentiate in UNMEASURED ways (style, approach, diversity),
even though per-type success rate (the LR axis) shows no differentiation?

Uses the ensemble runs (ens_3b: 4 agents x all 164 tasks x 9 seeds, every solution
logged). Three probes, all EXPLORATORY, no GPU:

  A. any@N voting benefit — mean per-agent pass@1 vs any@4 vs majority@4.
     If any@4 >> pass@1, solution DIVERSITY yields an ensemble gain even with no
     competence differentiation. Speaks to the "so what / contribution" gap.
  B. Solution diversity — per task, pairwise token-Jaccard distance between the 4
     agents' solutions. High = agents take genuinely different approaches.
  C. Agent-identity predictability — can a classifier recover agent_id from style
     features (length, #lines, loops, recursion, comprehensions, defs)? Accuracy
     vs 25% chance via a label-permutation null. If agents are stylistically
     identical too, accuracy ~= chance -> the null extends beyond success rate.

Writes audit/solution_diversity.md.
"""
import json, glob, re
from collections import defaultdict
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import warnings; warnings.filterwarnings("ignore")

rng = np.random.RandomState(20260608)
SEEDS = sorted(glob.glob("results_phase3/ens_3b/seed_*/task_log.jsonl"))


def code_only(s):
    """Strip markdown fences / stray backticks; keep the code body."""
    s = s or ""
    s = re.sub(r"```[a-zA-Z]*", "", s).replace("```", "")
    return s.strip()


def tokset(s):
    return set(re.findall(r"[A-Za-z_][A-Za-z_0-9]*|[^\sA-Za-z_0-9]", code_only(s)))


def jaccard_dist(a, b):
    A, B = tokset(a), tokset(b)
    if not A and not B:
        return 0.0
    return 1 - len(A & B) / len(A | B)


def style_feats(s):
    c = code_only(s)
    lines = [l for l in c.splitlines() if l.strip()]
    return [
        len(c),
        len(lines),
        c.count("for ") + c.count("while "),
        len(re.findall(r"\bdef\b", c)),
        1 if re.search(r"\[.*for .*in .*\]", c) else 0,   # comprehension
        c.count("return"),
        c.count("if "),
        c.count("lambda"),
    ]


def main():
    out = ["# GAP 2 — Unmeasured differentiation: diversity, voting, style\n",
           "Ensemble ens_3b (4 agents x 164 tasks x %d seeds). EXPLORATORY.\n" % len(SEEDS)]

    # ---- A. voting benefit ----
    p1, anyN, majN = [], [], []
    for sp in SEEDS:
        rows = [json.loads(l) for l in open(sp) if l.strip()]
        by_task = defaultdict(list)
        for r in rows:
            by_task[r["task_id"]].append(int(r["success"]))
        n_ag = max(len(v) for v in by_task.values())
        p1.append(np.mean([s for v in by_task.values() for s in v]))
        anyN.append(np.mean([1 if any(v) else 0 for v in by_task.values()]))
        majN.append(np.mean([1 if sum(v) > len(v) / 2 else 0 for v in by_task.values()]))
    p1, anyN, majN = np.array(p1), np.array(anyN), np.array(majN)

    def paired_ci(diff, n=10000):  # bootstrap CI on a per-seed paired difference
        d = rng.choice(diff, (n, len(diff)), replace=True).mean(1)
        return np.percentile(d, [2.5, 97.5])

    any_ci = paired_ci(anyN - p1); maj_ci = paired_ci(majN - p1)
    out.append("## A. Voting benefit (diversity payoff)\n")
    out.append(f"- mean per-agent pass@1: {p1.mean():.3f} ± {p1.std(ddof=1):.3f}")
    out.append(f"- any@N (≥1 of {n_ag} correct): {anyN.mean():.3f} ± {anyN.std(ddof=1):.3f}")
    out.append(f"- majority@N (strict, >½ of {n_ag}; 2–2 ties count as fail): "
               f"{majN.mean():.3f} ± {majN.std(ddof=1):.3f}")
    out.append(f"- **any@N − pass@1 = {(anyN - p1).mean():+.3f}** "
               f"[95% CI {any_ci[0]:+.3f},{any_ci[1]:+.3f}, paired over {len(p1)} seeds] "
               f"— headroom unlocked purely by solution diversity / coverage")
    out.append(f"- **majority@N − pass@1 = {(majN - p1).mean():+.3f}** "
               f"[95% CI {maj_ci[0]:+.3f},{maj_ci[1]:+.3f}] — net effect of strict voting; "
               f"CI spanning/below 0 ⇒ majority voting does NOT beat a single agent (it can "
               f"outvote correct minorities). The coverage gain needs a better selector than vote.")

    # ---- B. solution diversity ----
    dists, dists_corr = [], []
    for sp in SEEDS:
        rows = [json.loads(l) for l in open(sp) if l.strip()]
        by_task = defaultdict(list)
        for r in rows:
            by_task[r["task_id"]].append((r.get("solution", ""), int(r["success"])))
        for sols in by_task.values():
            for i in range(len(sols)):
                for j in range(i + 1, len(sols)):
                    d = jaccard_dist(sols[i][0], sols[j][0])
                    dists.append(d)
                    if sols[i][1] and sols[j][1]:
                        dists_corr.append(d)
    out.append("\n## B. Solution diversity (pairwise token-Jaccard distance)\n")
    out.append(f"- all agent pairs, all tasks: mean={np.mean(dists):.3f}, median={np.median(dists):.3f}")
    out.append(f"- among pairs where BOTH agents were correct: mean={np.mean(dists_corr):.3f} "
               f"(n={len(dists_corr)}) — diversity persists even on the same solved problem ⇒ "
               f"agents reach correctness by different code, not one canonical answer.")

    # ---- C. agent-identity predictability ----
    X, y = [], []
    for sp in SEEDS:
        for r in (json.loads(l) for l in open(sp) if l.strip()):
            X.append(style_feats(r.get("solution", "")))
            y.append(int(r["agent_id"]))
    X, y = np.array(X, float), np.array(y)
    clf = RandomForestClassifier(n_estimators=200, random_state=0)
    acc = cross_val_score(clf, X, y, cv=5).mean()
    chance = max(np.bincount(y)) / len(y)
    # permutation null on the SAME pipeline
    perm = []
    for _ in range(50):
        yp = rng.permutation(y)
        perm.append(cross_val_score(clf, X, yp, cv=5).mean())
    perm = np.array(perm)
    p_perm = (perm >= acc).mean()
    out.append("\n## C. Can agent identity be recovered from solution style?\n")
    out.append(f"- RF 5-fold accuracy: {acc:.3f} | majority-class chance: {chance:.3f} | "
               f"permutation-null mean: {perm.mean():.3f} (95th={np.percentile(perm, 95):.3f})")
    out.append(f"- permutation p (acc ≥ observed under shuffled labels): {p_perm:.3f}")
    verdict = ("indistinguishable from chance ⇒ **no stylistic agent signature**: "
               "the null extends beyond success rate to coding style"
               if p_perm > 0.05 else
               "ABOVE chance ⇒ a stylistic signature exists; differentiation may be "
               "present on an axis the LR success-rate test cannot see (revisit the null)")
    out.append(f"- **Verdict:** {verdict}.")

    open("audit/solution_diversity.md", "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
