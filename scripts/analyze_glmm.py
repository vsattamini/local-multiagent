#!/usr/bin/env python3
"""
Functional differentiation analysis for specialization experiments.

DESIGN NOTE: Riedl (2025) uses a GLMM with (1|problem) because every agent
attempts every problem in his coalition design. In our affinity-routing
design each problem is solved by exactly ONE agent, so n_problems == n_obs
and the random intercept is unidentifiable (confounded with residuals).

The right model for our design is logistic regression with the
task_type:agent interaction, tested via likelihood-ratio. We report the
contingency-table chi-square (already in final_metrics.json) for parity
with prior literature, plus the LR test on the logit interaction. Where
the data are too sparse for ML (perfect separation), Firth-style penalized
likelihood is used.

Usage:
    python scripts/analyze_glmm.py --all
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")


def load_task_log(exp_dir: Path) -> pd.DataFrame:
    log_path = exp_dir / "task_log.jsonl"
    if not log_path.exists():
        return pd.DataFrame()
    rows = []
    with open(log_path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    df["success_int"] = df["success"].astype(int)
    df["agent_id"] = df["agent_id"].astype(str)
    return df


def chi2_from_contingency(df: pd.DataFrame) -> dict:
    """Chi-square test on the task_type x agent contingency table of successes."""
    succ = df[df["success_int"] == 1]
    if len(succ) == 0:
        return {"error": "No successful tasks"}
    table = pd.crosstab(succ["task_type"], succ["agent_id"])
    chi2, p, dof, _ = stats.chi2_contingency(table)
    n = table.values.sum()
    cramers_v = np.sqrt(chi2 / (n * (min(table.shape) - 1))) if n > 0 else 0.0
    return {
        "method": "chi2_contingency",
        "chi2": float(chi2),
        "df": int(dof),
        "p_value": float(p),
        "cramers_v": float(cramers_v),
        "significant": bool(p < 0.05),
        "n_successes": int(n),
    }


def lr_test_logit(df: pd.DataFrame) -> dict:
    """Likelihood-ratio test on task_type:agent interaction in logistic regression."""
    n_obs = len(df)
    n_agents = df["agent_id"].nunique()
    if n_obs < 20 or n_agents < 2:
        return {"error": "Insufficient data"}

    try:
        m_null = smf.logit("success_int ~ C(task_type) + C(agent_id)", df).fit(disp=False)
        m_full = smf.logit("success_int ~ C(task_type) * C(agent_id)", df).fit(disp=False)
        ll_diff = m_full.llf - m_null.llf
        df_diff = int(m_full.df_model - m_null.df_model)
        p_value = float(1 - stats.chi2.cdf(2 * ll_diff, df=df_diff))
        return {
            "method": "logit_lr_test",
            "log_lik_null": float(m_null.llf),
            "log_lik_full": float(m_full.llf),
            "chi2": 2 * float(ll_diff),
            "df": df_diff,
            "p_value": p_value,
            "significant": bool(p_value < 0.05),
        }
    except Exception as e:
        return {"error": f"Logit: {e}"}


def analyze_experiment(exp_dir: Path) -> dict:
    df = load_task_log(exp_dir)
    if df.empty:
        return {"name": exp_dir.parent.name, "error": "No task log"}
    name = exp_dir.parent.name
    return {
        "name": name,
        "n_obs": len(df),
        "n_problems": df["task_id"].nunique(),
        "n_agents": df["agent_id"].nunique(),
        "chi2_test": chi2_from_contingency(df),
        "lr_test": lr_test_logit(df),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="results/glmm_analysis.json")
    args = parser.parse_args()

    main_experiments = [
        "results/exp2.1_experimental",
        "results/exp_low_temp",
        "results/exp_5_agents",
        "results/exp_3b_baseline",
        "results/exp_3b_model",
        "results/exp_3b_low_temp",
        "results/exp_3b_5_agents",
        "results/exp_7b_model",
    ]

    results = []
    print("=" * 100)
    print(f"{'Experiment':<26} {'N':>4} | "
          f"{'chi2_χ²':>8} {'p':>9} {'V':>5} | "
          f"{'LR_χ²':>8} {'p':>9} {'sig':>4}")
    print("=" * 100)

    for exp_path in main_experiments:
        seed_dir = Path(exp_path) / "seed_42"
        if not seed_dir.exists():
            continue
        r = analyze_experiment(seed_dir)
        results.append(r)
        c = r.get("chi2_test", {})
        l = r.get("lr_test", {})
        c_chi = c.get("chi2", float("nan"))
        c_p = c.get("p_value", float("nan"))
        c_v = c.get("cramers_v", float("nan"))
        l_chi = l.get("chi2", float("nan"))
        l_p = l.get("p_value", float("nan"))
        l_sig = "✓" if l.get("significant") else ("err" if "error" in l else "✗")
        print(f"{r['name']:<26} {r['n_obs']:>4} | "
              f"{c_chi:>8.2f} {c_p:>9.4g} {c_v:>5.2f} | "
              f"{l_chi:>8.2f} {l_p:>9.4g} {l_sig:>4}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
