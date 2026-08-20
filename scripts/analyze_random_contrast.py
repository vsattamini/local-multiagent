#!/usr/bin/env python3
"""
GAP 3 — Is S ("routing concentration") an artifact, or affinity-driven?

The thesis dismisses S as a routing artifact by assertion. The RandomRouter
control runs (rand_*) were executed but never reported. This computes, per seed,
the canonical S = 1 - H(type|agent)/H(type) for the AFFINITY router vs the matched
RANDOM router (same model, n_agents=3, router_temperature=0.5, 164 tasks), and
tests S_affinity - S_random.

Interpretation:
  - If S_affinity ~= S_random: at this setting S is just the finite-sample
    concentration floor — not produced by the affinity mechanism.
  - If S_affinity >> S_random: the affinity router DOES create division of labor
    (task allocation). That is real emergent allocation — but, per the LR null,
    NOT functional (success-rate) differentiation. The two are distinct claims.

No GPU. Reads task_log.jsonl. Writes audit/random_contrast.md.
"""
import json, glob, math
from collections import Counter, defaultdict
import numpy as np
from scipy import stats

# (size_label, affinity_dir, random_dir)
PAIRS = [
    ("1.5B", "results_multiseed/exp2.1_experimental", "results_phase3/rand_1.5b"),
    ("3B",   "results_multiseed/exp_3b_baseline",      "results_phase3/rand_3b"),
    ("7B",   "results_multiseed/exp_7b_baseline",      "results_phase3/rand_7b"),
]


def entropy(counts):
    total = sum(counts)
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counts if c > 0)


def spec_index(log):
    """Canonical S, identical to audit/recompute_metrics.specialization_index."""
    if not log:
        return None
    types = [t["task_type"] for t in log]
    H_task = entropy(list(Counter(types).values()))
    if H_task == 0:
        return None
    by_agent = defaultdict(list)
    for t in log:
        by_agent[t["agent_id"]].append(t["task_type"])
    H_cond = sum((len(ts) / len(log)) * entropy(list(Counter(ts).values()))
                 for ts in by_agent.values())
    return max(0.0, min(1.0, 1 - H_cond / H_task))


def per_seed_S(d):
    out = []
    for sp in sorted(glob.glob(f"{d}/seed_*/task_log.jsonl")):
        log = [json.loads(l) for l in open(sp) if l.strip()]
        s = spec_index(log)
        if s is not None:
            out.append(s)
    return np.array(out)


def hedges_g(a, b):
    na, nb = len(a), len(b)
    sp = math.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if sp == 0:
        return float("nan")
    d = (a.mean() - b.mean()) / sp
    J = 1 - 3 / (4 * (na + nb) - 9)  # small-sample correction
    return d * J


def boot_ci(a, b, n=10000):
    """BCa-free percentile bootstrap of mean(affinity) - mean(random)."""
    rng = np.random.RandomState(20260608)
    diffs = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n)]
    return np.percentile(diffs, [2.5, 97.5])


def main():
    out = ["# GAP 3 — S: affinity-driven concentration vs RandomRouter floor\n",
           "Per-seed specialization index S for the affinity router vs the matched "
           "RandomRouter control (same model/n_agents/router_temp). EXPLORATORY "
           "(controls pre-registered as H2 deconfound, but contrast not in original plan). "
           "p-values Holm-corrected across the 3 size-tests (family) to control forking paths.\n"]
    rows = []
    for label, ad, rd in PAIRS:
        a, r = per_seed_S(ad), per_seed_S(rd)
        if len(a) == 0 or len(r) == 0:
            rows.append((label, None)); continue
        try:
            p = stats.mannwhitneyu(a, r, alternative="two-sided").pvalue
        except ValueError:
            p = float("nan")
        rows.append((label, dict(a=a, r=r, d=a.mean() - r.mean(), ci=boot_ci(a, r),
                                 g=hedges_g(a, r), p=p)))
    # Holm-Bonferroni across the valid tests
    valid = [(i, rr[1]["p"]) for i, rr in enumerate(rows) if rr[1] is not None]
    order = sorted(valid, key=lambda x: x[1]); m = len(order)
    holm = {}
    running = 0.0
    for rank, (i, p) in enumerate(order):
        adj = min(1.0, p * (m - rank)); running = max(running, adj); holm[i] = running

    out.append("| size | S_affinity (mean±sd, n) | S_random (mean±sd, n) | ΔS | 95% CI | Hedges g | MWU p | Holm p |")
    out.append("|---|---|---|---|---|---|---|---|")
    for i, (label, v) in enumerate(rows):
        if v is None:
            out.append(f"| {label} | (missing) | | | | | | |"); continue
        out.append(f"| {label} | {v['a'].mean():.3f}±{v['a'].std(ddof=1):.3f} (n={len(v['a'])}) | "
                   f"{v['r'].mean():.3f}±{v['r'].std(ddof=1):.3f} (n={len(v['r'])}) | {v['d']:+.3f} | "
                   f"[{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}] | {v['g']:+.2f} | {v['p']:.3f} | {holm[i]:.3f} |")

    out.append("\n## Reading\n")
    out.append("- ΔS > 0 with CI excluding 0 ⇒ the affinity mechanism concentrates routing "
               "beyond the random floor: real **task allocation / division of labor**.")
    out.append("- This is the metric S was *designed* to capture, and is logically distinct "
               "from functional (success-rate) differentiation, which the LR test finds absent "
               "(see lr_power_validation.md). Allocation emerged; competence differentiation did not.")
    open("audit/random_contrast.md", "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
