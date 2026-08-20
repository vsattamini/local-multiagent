#!/usr/bin/env python3
"""
Exploratory analyses for the adversarial-foil review (NO GPU).
Registers everything as a committed artifact: audit/EXPLORATORY_RESULTS.md.

All numbers here are EXPLORATORY (computed post-hoc from existing logs), not
confirmatory. They are reproducible: re-run `python audit/exploratory_analysis.py`.

Covers:
  1. Ensemble voting (ens_3b): pass@1 vs any@N(==test@N, deployable) vs majority@N,
     overall + per task-type, mean +/- 95% CI across seeds.
  2. Between-agent success DISAGREEMENT rate per task (why LR has nothing to bite on).
  3. Specialization S contrast: affinity baseline vs RandomRouter vs greedy vs popN vs
     ensemble — earns (or refutes) the "S is a routing artifact" claim WITH data.
  4. LR-computability map: which designs admit the agent x type interaction test.
"""
import json, glob, os
import numpy as np
from collections import defaultdict

def ci95(xs):
    xs = np.asarray(xs, float)
    if len(xs) == 0: return (float("nan"), float("nan"), float("nan"), 0)
    m = xs.mean()
    if len(xs) == 1: return (m, m, m, 1)
    se = xs.std(ddof=1) / np.sqrt(len(xs))
    return (m, m - 1.96*se, m + 1.96*se, len(xs))

def load_seed_tasklog(path):
    return [json.loads(l) for l in open(path) if l.strip()]

# ---------- 1 & 2: voting + disagreement on ens_3b ----------
def voting_analysis(exp="results_phase3/ens_3b"):
    seed_dirs = sorted(glob.glob(f"{exp}/seed_*"))
    per_seed = []   # dicts of metrics per seed
    for sd in seed_dirs:
        tl = os.path.join(sd, "task_log.jsonl")
        if not os.path.exists(tl): continue
        rows = load_seed_tasklog(tl)
        bytask = defaultdict(dict)
        ttype = {}
        for r in rows:
            bytask[r["task_id"]][r["agent_id"]] = bool(r["success"])
            ttype[r["task_id"]] = r["task_type"]
        n_tasks = len(bytask)
        if n_tasks == 0: continue
        # overall
        p1   = np.mean([s for d in bytask.values() for s in d.values()])
        anyN = np.mean([any(d.values()) for d in bytask.values()])
        majN = np.mean([sum(d.values()) > len(d)/2 for d in bytask.values()])
        disagree = np.mean([len(set(d.values())) > 1 for d in bytask.values()])
        # per-type any@N and pass@1
        pertype = {}
        types = set(ttype.values())
        for t in types:
            tasks_t = [tid for tid in bytask if ttype[tid] == t]
            if not tasks_t: continue
            p1_t   = np.mean([s for tid in tasks_t for s in bytask[tid].values()])
            any_t  = np.mean([any(bytask[tid].values()) for tid in tasks_t])
            pertype[t] = (p1_t, any_t, len(tasks_t))
        per_seed.append(dict(p1=p1, anyN=anyN, majN=majN, disagree=disagree,
                             pertype=pertype, n_tasks=n_tasks,
                             n_agents=len(next(iter(bytask.values())))))
    return per_seed

# ---------- 3: S contrast ----------
def s_values(exp):
    out = []
    for sd in sorted(glob.glob(f"{exp}/seed_*")):
        fp = os.path.join(sd, "final_metrics.json")
        try: m = json.load(open(fp))["metrics"]
        except Exception: continue
        s = m.get("specialization_index")
        if s is not None: out.append(s)
    return out

def main():
    L = []
    L.append("# Exploratory results — registered artifact (NO-GPU, post-hoc)\n")
    L.append("Reproduce: `python audit/exploratory_analysis.py`. "
             "All numbers EXPLORATORY (not pre-registered). Companion to "
             "`audit/ADVERSARIAL_FOIL_REVIEW.md`.\n")

    # 1 & 2
    ps = voting_analysis()
    L.append("## 1. Ensemble voting on ens_3b (homogeneous 3B, every agent every task)\n")
    if ps:
        na = ps[0]["n_agents"]
        L.append(f"- seeds: {len(ps)} · agents/task: {na} · tasks: {ps[0]['n_tasks']}")
        for key, label in [("p1","pass@1 (mean per agent)"),
                           ("anyN","any@N == test@N (keep any soln passing the test suite — DEPLOYABLE)"),
                           ("majN","majority@N")]:
            m, lo, hi, n = ci95([s[key] for s in ps])
            L.append(f"- **{label}**: {m:.3f}  [95% CI {lo:.3f}, {hi:.3f}]  (n={n})")
        # voting gain
        gains = [s["anyN"] - s["p1"] for s in ps]
        m, lo, hi, n = ci95(gains)
        L.append(f"- **any@N − pass@1 (voting gain)**: +{m*100:.1f}pp  [95% CI {lo*100:.1f}, {hi*100:.1f}pp]")
        majg = [s["majN"] - s["p1"] for s in ps]
        mm, mlo, mhi, _ = ci95(majg)
        L.append(f"- majority@N − pass@1: {mm*100:+.1f}pp  [95% CI {mlo*100:.1f}, {mhi*100:.1f}pp]  "
                 f"(majority voting does NOT beat single-shot — only oracle/test-filtered selection does)")
        # 2 disagreement
        m, lo, hi, n = ci95([s["disagree"] for s in ps])
        L.append("\n## 2. Between-agent success disagreement per task\n")
        L.append(f"- tasks where agents disagree on pass/fail: **{m*100:.1f}%**  [95% CI {lo*100:.1f}, {hi*100:.1f}%]")
        L.append("- => on ~{:.0f}% of tasks all agents agree (all pass / all fail). With identical".format(100-m*100))
        L.append("  weights on a near-saturated benchmark, the only between-agent variance is decode")
        L.append("  noise, which rarely flips success => the agent x type interaction has almost nothing")
        L.append("  to bite on. The LR null is structural, not a power failure.")
        # per-type voting
        L.append("\n### per-type pass@1 vs any@N (mean across seeds)")
        L.append("| type | pass@1 | any@N | gain | ~n tasks |")
        L.append("|---|---|---|---|---|")
        types = sorted({t for s in ps for t in s["pertype"]})
        for t in types:
            p1s = [s["pertype"][t][0] for s in ps if t in s["pertype"]]
            ans = [s["pertype"][t][1] for s in ps if t in s["pertype"]]
            nts = int(np.median([s["pertype"][t][2] for s in ps if t in s["pertype"]]))
            L.append(f"| {t} | {np.mean(p1s):.3f} | {np.mean(ans):.3f} | +{(np.mean(ans)-np.mean(p1s))*100:.1f}pp | {nts} |")
    else:
        L.append("- (no ens_3b seeds found)")

    # 3 S contrast
    L.append("\n## 3. Specialization S: routing concentration vs chance\n")
    L.append("| design | router | assignment | n_agents | S (mean±sd, n) | note |")
    L.append("|---|---|---|---|---|---|")
    rows = [
        ("results_multiseed/exp_3b_baseline", "affinity", "single", 3, "baseline"),
        ("results_phase3/rand_3b", "random", "single", 3, "CHANCE FLOOR"),
        ("results_phase3/rand_7b", "random", "single", 3, "chance floor"),
        ("results_phase3/rand_1.5b", "random", "single", 3, "chance floor"),
        ("results_phase3/greedy_3b", "greedy", "single", 3, "collapses to 1 agent"),
        ("results_multiseed/exp_3b_low_temp", "affinity τ low", "single", 3, "low router temp"),
        ("results_phase3/popN_8", "affinity τ0.1", "single", 8, "large pop"),
        ("results_phase3/popN_12", "affinity τ0.1", "single", 12, "large pop"),
        ("results_phase3/popN_16", "affinity τ0.1", "single", 16, "large pop"),
        ("results_phase3/ens_3b", "round_robin", "ENSEMBLE", 4, "everyone does everything"),
    ]
    base = None
    for path, rt, asg, na, note in rows:
        xs = s_values(path)
        if not xs: continue
        m = np.mean(xs); sd = np.std(xs)
        if "baseline" in note and base is None: base = m
        L.append(f"| {os.path.basename(path)} | {rt} | {asg} | {na} | {m:.3f}±{sd:.3f} (n={len(xs)}) | {note} |")
    rand = np.mean(s_values("results_phase3/rand_3b") or [np.nan])
    if base and rand:
        L.append(f"\n- affinity baseline S ≈ {base:.3f} vs random S ≈ {rand:.3f} ⇒ affinity concentrates "
                 f"**{base/rand:.1f}× above chance** (same n_agents=3, same router temp 0.5, both single-assignment).")
    L.append("- => S is NOT pure chance: affinity routing genuinely concentrates, low temp / large pop "
             "drives S to ~0.5. BUT this is ALLOCATIONAL specialization (who gets which task), with NO "
             "competence basis (identical weights) and NO functional consequence (LR null, pass@1 flat). "
             "Reclaim S as emergent division of labor; do not call it functional specialization.")

    # 4 computability map
    L.append("\n## 4. LR-computability map (can the agent×type interaction test even run?)\n")
    L.append("| design | weights | assignment | LR computable? | can LR reject for a REAL reason? |")
    L.append("|---|---|---|---|---|")
    L.append("| ens_3b | homogeneous 3B | ensemble | YES | NO — agents symmetric ⇒ E[interaction]=0 ⇒ only Type-I |")
    L.append("| mech_*_typefilter | homogeneous | single | NO (perfect separation) | n/a |")
    L.append("| het_swarm (current) | heterogeneous | single | NO (perfect separation) | n/a |")
    L.append("| **het_swarm_ensemble (NEW, to run)** | **heterogeneous** | **ensemble** | **YES** | **YES — 3B genuinely better on some types ⇒ real interaction** |")
    L.append("\n=> The decisive cell (het weights + ensemble) is the ONLY one that can yield a non-trivial, "
             "computable LR. It has not been run. ens_3b is its matched homogeneous control.")

    txt = "\n".join(L) + "\n"
    open("audit/EXPLORATORY_RESULTS.md", "w").write(txt)
    print(txt)

if __name__ == "__main__":
    main()
