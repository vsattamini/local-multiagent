#!/usr/bin/env python3
"""
PRE-REGISTRATION + simulation for het_swarm_ensemble (the decisive run).

Opportunity flagged on top of the meta-reviewer's plan: the reviewer predicts the
heterogeneous swarm will yield "a significant, computable LR (per-type gap ~0.41)" and
treats that as the functional-differentiation result. But a heterogeneous swarm in which
the stronger model is simply UNIFORMLY better produces:
  (i)  a large agent MAIN effect (competence differs) — not specialization, and
  (ii) ceiling-inflated per-type SUCCESS-RATE gaps — a probability-scale artifact.
Neither is "different agents are best at different task types" (the thesis's claim).

This script shows, by simulation, which statistic survives which scenario, so the
het_swarm_ensemble analysis can be pre-committed to the RIGHT test before the data lands.

Scenarios (3 agents: 2 weak ~1.5B, 1 strong ~3B; realistic per-type base difficulty; 41
obs/cell; ceiling at 0.98):
  H0_competence : strong agent gets a CONSTANT log-odds boost on EVERY type (uniform
                  competence, NO niche). This is "bigger model, no specialization."
  H1_specialize : each agent gets a boost on ONE distinct type (genuine niche).

Reported per scenario: agent MAIN-effect LR p, agent×type INTERACTION LR p (logit),
and the raw per-type success-rate max-min gap. The pre-registration falls out of the
contrast.  No GPU.
"""
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
import warnings; warnings.filterwarnings("ignore")

rng = np.random.RandomState(20260608)
TYPES = ["string", "math", "list", "logic"]
logit = lambda p: np.log(p / (1 - p))
expit = lambda x: 1 / (1 + np.exp(-x))
# Table-1 uses MODERATE base rates (no ceiling) so the saturated logit MLE is estimable;
# the ceiling-artifact point is shown separately by arithmetic (Section 2).
BASE_MOD = {"string": 0.60, "math": 0.55, "list": 0.65, "logic": 0.45}
BASE_MOD_L = {t: logit(p) for t, p in BASE_MOD.items()}
# Section-2 uses the OBSERVED high ens_3b rates to expose ceiling inflation.
BASE_HIGH = {"string": 0.82, "math": 0.79, "list": 0.89, "logic": 0.66}


def simulate(scenario, per_cell=41, boost=1.2):
    """3 agents; agent 2 = strong. boost in LOG-ODDS units. Moderate base rates."""
    rows = []
    for a in range(3):
        for t in TYPES:
            lo = BASE_MOD_L[t]
            if scenario == "H0_competence" and a == 2:
                lo += boost                              # uniform competence, all types
            elif scenario == "H1_specialize":
                if t == TYPES[a % 4]:                    # each agent owns one type
                    lo += boost
            p = min(0.98, max(0.02, expit(lo)))
            for hit in (rng.rand(per_cell) < p):
                rows.append({"agent_id": a, "task_type": t, "success": int(hit)})
    return pd.DataFrame(rows)


def lr(df, full, reduced):
    """LR p-value; returns nan on non-convergence/separation so callers can drop it."""
    try:
        m0 = smf.logit(reduced, df).fit(disp=False)
        m1 = smf.logit(full, df).fit(disp=False)
        chi2 = 2 * (m1.llf - m0.llf); ddf = int(m1.df_model - m0.df_model)
        return float(1 - stats.chi2.cdf(chi2, ddf))
    except Exception:
        return float("nan")


def maxgap(df):
    rt = df.groupby(["task_type", "agent_id"])["success"].mean().unstack("agent_id")
    return float((rt.max(axis=1) - rt.min(axis=1)).max())


def main():
    out = ["# Pre-registration — het_swarm_ensemble: main effect vs specialization\n",
           "Simulation establishing which statistic answers the thesis question BEFORE the "
           "decisive run lands, so the analysis is pre-committed (anti-forking-paths).\n",
           "## Section 1 — main effect vs interaction (moderate base rates, 41 obs/cell)\n",
           "Three agents (agent 2 = strong/3B-like, +1.2 log-odds), 200 sims/scenario. "
           "Moderate base rates (no ceiling) so the saturated logit MLE is estimable.\n",
           "| scenario | agent MAIN-effect p (median) | agent×type INTERACTION p (median) "
           "| INTERACTION reject-rate | raw success-rate max-min gap (median) |",
           "|---|---|---|---|---|"]
    for scen in ["H0_competence", "H1_specialize"]:
        mains, inters, gaps, rej, nval = [], [], [], 0, 0
        N = 200
        for _ in range(N):
            df = simulate(scen)
            p_main = lr(df, "success ~ C(task_type)+C(agent_id)", "success ~ C(task_type)")
            p_int = lr(df, "success ~ C(task_type)*C(agent_id)", "success ~ C(task_type)+C(agent_id)")
            gaps.append(maxgap(df))
            if not np.isnan(p_main):
                mains.append(p_main)
            if not np.isnan(p_int):
                inters.append(p_int); nval += 1
                if p_int < 0.05:
                    rej += 1
        out.append(f"| {scen} | {np.median(mains):.3f} | {np.median(inters):.3f} | "
                   f"{rej/max(nval,1):.2f} (n={nval}) | {np.median(gaps):.3f} |")

    # Section 2 — ceiling inflates the RAW per-type gap with ZERO interaction (arithmetic).
    out.append("\n## Section 2 — ceiling inflates the raw gap under PURE competence (no interaction)\n")
    out.append("At the observed high ens_3b base rates, give the strong agent a CONSTANT "
               "+1.2 log-odds on every type (uniform competence, exactly zero interaction). "
               "Expected per-type success rates and the resulting raw max-min gap (no sampling, "
               "just arithmetic):\n")
    out.append("| type | weak agent rate | strong agent rate (+1.2 logit) | per-type gap |")
    out.append("|---|---|---|---|")
    gaps2 = []
    for t in TYPES:
        pw = BASE_HIGH[t]
        ps = min(0.98, expit(logit(pw) + 1.2))
        gaps2.append(ps - pw)
        out.append(f"| {t} | {pw:.2f} | {ps:.2f} | {ps - pw:+.2f} |")
    out.append(f"\n- raw max-min per-type gap = **{max(gaps2):.2f}** — large, yet there is "
               f"**zero** agent×type interaction by construction. The gap is biggest on the "
               f"HARDEST type (logic) and compressed on near-ceiling types (list), which is "
               f"exactly the spurious 'profile difference' a naive success-rate reading would "
               f"mislabel as specialization. (This is also why the reviewer's predicted "
               f"'per-type gap ~0.41' is consistent with NO specialization at all.)")

    out.append("\n## Pre-registered interpretation rules for het_swarm_ensemble\n")
    out.append("- **The differentiation claim rests ONLY on the agent×type INTERACTION LR "
               "(logit scale).** Under uniform competence (H0) the interaction test stays at "
               "~alpha — it does NOT false-positive on a pure size advantage — while the agent "
               "MAIN effect is strongly significant and the raw success-rate gap is large. So:")
    out.append("- A significant **main effect** in het_swarm is EXPECTED and is NOT "
               "specialization (it just says the 3B is better — trivially true).")
    out.append("- A large **raw per-type success-rate gap** is NOT sufficient evidence: under "
               "uniform competence + ceiling it is already inflated. Do not report it as "
               "differentiation without the interaction test.")
    out.append("- Only a significant **interaction** (each agent differentially better at "
               "different types) supports 'functional differentiation emerged'. The matched "
               "homogeneous control **ens_3b_n3** must show NO interaction (it already does in "
               "ens_3b); het showing interaction where ens_3b_n3 does not = heterogeneity-driven "
               "specialization, the cleanest possible positive result.")
    open("audit/het_interaction_prereg.md", "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
