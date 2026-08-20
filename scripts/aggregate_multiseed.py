#!/usr/bin/env python3
"""
Aggregate multi-seed results: mean +/- 95% CI for S, D, Pass@1, chi2, Cramer's V,
plus the logistic-regression LR test (functional differentiation) per seed and a
seed-level significance count. Emits results_multiseed/aggregate.json and a
markdown table to results_multiseed/aggregate.md.

Run:  ./.venv/bin/python scripts/aggregate_multiseed.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings("ignore")

ROOT = Path("results_multiseed")
CONFIGS = ["exp2.1_experimental", "exp_low_temp", "exp_5_agents",
           "exp_3b_baseline", "exp_3b_model", "exp_3b_low_temp",
           "exp_3b_5_agents", "exp_7b_model"]
LABELS = {
    "exp2.1_experimental": "1.5B baseline (T=0.5, n=3)",
    "exp_low_temp":        "1.5B low-temp (T=0.1, n=3)",
    "exp_5_agents":        "1.5B 5-agents (T=0.3, n=5)",
    "exp_3b_baseline":     "3B baseline (T=0.5, n=3)",
    "exp_3b_model":        "3B mid (T=0.3, n=3)",
    "exp_3b_low_temp":     "3B low-temp (T=0.1, n=3)",
    "exp_3b_5_agents":     "3B 5-agents (T=0.3, n=5)",
    "exp_7b_model":        "7B (T=0.3, n=3)",
}


def ci(vals):
    vals = [v for v in vals if v is not None]
    n = len(vals)
    if n == 0:
        return {"mean": None, "std": None, "lo": None, "hi": None, "n": 0, "vals": []}
    m = float(np.mean(vals)); sd = float(np.std(vals, ddof=1)) if n > 1 else 0.0
    se = sd / np.sqrt(n) if n else 0.0
    t = stats.t.ppf(0.975, df=max(n - 1, 1))
    return {"mean": m, "std": sd, "lo": m - t * se, "hi": m + t * se, "n": n,
            "vals": [round(float(v), 4) for v in vals]}


def lr_test(log):
    df = pd.DataFrame(log)
    df["y"] = df["success"].astype(int)
    df["agent_id"] = df["agent_id"].astype(str)
    if len(df) < 20 or df["agent_id"].nunique() < 2:
        return None
    try:
        m0 = smf.logit("y ~ C(task_type) + C(agent_id)", df).fit(disp=False)
        m1 = smf.logit("y ~ C(task_type) * C(agent_id)", df).fit(disp=False)
        chi2 = 2 * (m1.llf - m0.llf)
        ddf = int(m1.df_model - m0.df_model)
        p = float(1 - stats.chi2.cdf(chi2, df=ddf))
        return {"chi2": float(chi2), "df": ddf, "p": p, "sig": bool(p < 0.05)}
    except Exception as e:
        return {"error": str(e)[:80]}


def main():
    out = {}
    md = ["# Multi-seed aggregate (mean ± 95% CI over seeds)\n"]
    md.append("| Experiment | seeds | Pass@1 | S | S sig (perm, /seed) | D | χ² (counts) | LR sig /seed |")
    md.append("|---|---|---|---|---|---|---|---|")
    for cfg in CONFIGS:
        base = ROOT / cfg
        seed_dirs = sorted(base.glob("seed_*"))
        S, D, P, CHI, V = [], [], [], [], []
        perm_sig = 0; lr_sig = 0; lr_done = 0; nseed = 0
        for sd in seed_dirs:
            fm = sd / "final_metrics.json"
            tl = sd / "task_log.jsonl"
            if not fm.exists():
                continue
            nseed += 1
            m = json.load(open(fm))["metrics"]
            S.append(m["specialization_index"])
            D.append(m["context_divergence"])
            P.append(m["summary_stats"]["pass_at_1"])
            CHI.append(m["functional_differentiation"]["chi2"])
            V.append(m["functional_differentiation"]["effect_size"])
            if m["specialization_significance"]["significant"]:
                perm_sig += 1
            if tl.exists():
                log = [json.loads(l) for l in open(tl) if l.strip()]
                lr = lr_test(log)
                if lr and "chi2" in lr:
                    lr_done += 1
                    if lr["sig"]:
                        lr_sig += 1
        rec = {"label": LABELS[cfg], "n_seeds": nseed,
               "pass_at_1": ci(P), "S": ci(S), "D": ci(D),
               "chi2": ci(CHI), "cramers_v": ci(V),
               "perm_significant_seeds": f"{perm_sig}/{nseed}",
               "lr_significant_seeds": f"{lr_sig}/{lr_done}"}
        out[cfg] = rec
        if nseed:
            md.append(
                f"| {LABELS[cfg]} | {nseed} | "
                f"{rec['pass_at_1']['mean']:.3f} [{rec['pass_at_1']['lo']:.3f},{rec['pass_at_1']['hi']:.3f}] | "
                f"{rec['S']['mean']:.3f} [{rec['S']['lo']:.3f},{rec['S']['hi']:.3f}] | "
                f"{perm_sig}/{nseed} | "
                f"{rec['D']['mean']:.3f} | "
                f"{rec['chi2']['mean']:.1f} | "
                f"{lr_sig}/{lr_done} |")
        else:
            md.append(f"| {LABELS[cfg]} | 0 | (no data yet) | | | | | |")

    ROOT.mkdir(exist_ok=True)
    json.dump(out, open(ROOT / "aggregate.json", "w"), indent=2)
    (ROOT / "aggregate.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nWrote {ROOT/'aggregate.json'} and {ROOT/'aggregate.md'}")


if __name__ == "__main__":
    main()
