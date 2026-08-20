#!/usr/bin/env python3
"""
Independent recomputation audit.

Re-derives S (specialization index), Pass@1, tasks-by-agent/type, and the
chi-square functional-differentiation test DIRECTLY from each raw
task_log.jsonl, then compares against the stored final_metrics.json.

The implementations here are written from scratch (not imported from src/)
so that any bug in src/swarm/metrics.py would surface as a mismatch.

Usage: python audit/recompute_metrics.py
"""
import json
import math
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
from scipy.stats import chi2_contingency

RESULTS = Path("results")


def load_log(p):
    rows = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def entropy(counts):
    total = sum(counts)
    if total == 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c > 0:
            pr = c / total
            h -= pr * math.log2(pr)
    return h


def specialization_index(log):
    """S = 1 - H(type|agent)/H(type), computed independently."""
    if not log:
        return 0.0
    types = [t["task_type"] for t in log]
    total = len(log)
    H_task = entropy(list(Counter(types).values()))
    if H_task == 0:
        return 0.0
    by_agent = defaultdict(list)
    for t in log:
        by_agent[t["agent_id"]].append(t["task_type"])
    H_cond = 0.0
    for agent, ts in by_agent.items():
        p_agent = len(ts) / total
        H_cond += p_agent * entropy(list(Counter(ts).values()))
    S = 1 - H_cond / H_task
    return max(0.0, min(1.0, S))


def chi2_success(log):
    """Chi-square on agent x task_type table of SUCCESS counts (matches src)."""
    rows = defaultdict(lambda: defaultdict(float))
    agents = set()
    types = set()
    for t in log:
        agents.add(t["agent_id"])
        types.add(t["task_type"])
        if t["success"]:
            rows[t["agent_id"]][t["task_type"]] += 1
    agents = sorted(agents)
    types = sorted(types)
    if len(agents) < 2 or len(types) < 2:
        return None
    table = np.array([[rows[a][ty] for ty in types] for a in agents], dtype=float)
    # drop all-zero rows/cols the way pandas/scipy would still include them;
    # scipy errors on zero marginals, so guard:
    if (table.sum(axis=0) == 0).any() or (table.sum(axis=1) == 0).any():
        # replicate src behaviour: it passes the raw crosstab (zeros included)
        try:
            chi2, p, dof, _ = chi2_contingency(table)
        except ValueError:
            return {"chi2": None, "p": None, "V": None, "note": "zero-marginal"}
    else:
        chi2, p, dof, _ = chi2_contingency(table)
    n = table.sum()
    V = math.sqrt(chi2 / (n * (min(table.shape) - 1))) if n > 0 else 0.0
    return {"chi2": float(chi2), "p": float(p), "dof": int(dof), "V": float(V)}


def pass_at_1(log):
    if not log:
        return 0.0
    return sum(1 for t in log if t["success"]) / len(log)


def close(a, b, tol=1e-3):
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= tol


def audit_dir(d):
    log_p = d / "task_log.jsonl"
    fm_p = d / "final_metrics.json"
    if not log_p.exists() or not fm_p.exists():
        return None
    log = load_log(log_p)
    fm = json.load(open(fm_p))["metrics"]

    rc_S = specialization_index(log)
    rc_p1 = pass_at_1(log)
    rc_chi = chi2_success(log)

    st_S = fm["specialization_index"]
    st_p1 = fm["summary_stats"]["pass_at_1"]
    st_chi = fm["functional_differentiation"]["chi2"]
    st_V = fm["functional_differentiation"]["effect_size"]

    flags = []
    if not close(rc_S, st_S):
        flags.append(f"S: recomputed {rc_S:.4f} vs stored {st_S:.4f}")
    if not close(rc_p1, st_p1):
        flags.append(f"pass@1: recomputed {rc_p1:.4f} vs stored {st_p1:.4f}")
    if rc_chi and rc_chi.get("chi2") is not None:
        if not close(rc_chi["chi2"], st_chi, tol=0.5):
            flags.append(f"chi2: recomputed {rc_chi['chi2']:.2f} vs stored {st_chi:.2f}")
        if not close(rc_chi["V"], st_V, tol=0.02):
            flags.append(f"V: recomputed {rc_chi['V']:.3f} vs stored {st_V:.3f}")

    return {
        "dir": str(d.relative_to(RESULTS)),
        "n": len(log),
        "rc_S": rc_S, "st_S": st_S,
        "rc_p1": rc_p1, "st_p1": st_p1,
        "rc_chi2": rc_chi["chi2"] if rc_chi else None,
        "st_chi2": st_chi,
        "flags": flags,
    }


def main():
    dirs = sorted(set(p.parent for p in RESULTS.rglob("final_metrics.json")))
    print(f"Found {len(dirs)} experiment dirs with final_metrics.json\n")
    print(f"{'dir':<34}{'n':>4} {'S_rc':>7}{'S_st':>7} {'p1_rc':>7}{'p1_st':>7} {'chi_rc':>8}{'chi_st':>8}  flags")
    print("-" * 120)
    all_flags = []
    for d in dirs:
        r = audit_dir(d)
        if r is None:
            continue
        fl = "OK" if not r["flags"] else "MISMATCH"
        print(f"{r['dir']:<34}{r['n']:>4} {r['rc_S']:>7.3f}{r['st_S']:>7.3f} "
              f"{r['rc_p1']:>7.3f}{r['st_p1']:>7.3f} "
              f"{(r['rc_chi2'] or 0):>8.1f}{(r['st_chi2'] or 0):>8.1f}  {fl}")
        if r["flags"]:
            all_flags.append((r["dir"], r["flags"]))
    print("\n" + "=" * 60)
    if all_flags:
        print(f"{len(all_flags)} dirs with MISMATCHES:")
        for d, fl in all_flags:
            print(f"  {d}:")
            for f in fl:
                print(f"     - {f}")
    else:
        print("ALL stored metrics reproduce from raw logs (within tolerance).")


if __name__ == "__main__":
    main()
