# LR-test Power & Calibration Validation

The LR test detects per-type skill differences of MEANINGFUL size (>=0.15 gap) and does not false-positive — so the IDENTICAL-AGENT (ens_3b) null is a true negative for meaningful differentiation, not low power. Scope: this validates the HOMOGENEOUS-ensemble null only; the heterogeneous-weight interaction (het_swarm_ensemble) is the decisive test and is NOT settled here (see audit/het_interaction_prereg.md). A trivially small <0.10 gap cannot be excluded.

## A. Power vs injected effect size — matched to ens_3b design (4 agents, ~41 obs/cell, base success ~0.8 as observed)

NB: the x-axis is the *induced agent-to-agent gap* on the owned type (synth boosts the owned type by +d and damps the other three by -d/3, so the max-min spread across agents = 4d/3). This is the SAME statistic as the observed gap in section C — so the two are directly comparable.

| induced max-min per-type gap | reject-rate (power) over 400 sims |
|---|---|
| 0.00 | 0.080  <- null (should ~= 0.05) |
| 0.05 | 0.102 |
| 0.10 | 0.465 |
| 0.15 | 0.905  <- POWER |
| 0.20 | 0.998  <- POWER |

## B. Applied to REAL ensemble data (ens_3b — every problem x every agent)

- seeds with computable LR: 10
- omnibus LR p-values: min=0.961, median=0.993, frac<0.05 = 0.00

## C. Observed per-type skill gap vs its seed-matched null

Per seed, the largest spread in per-type success rate across the 4 agents (strongest *observed* differentiation), and where it falls in a parametric bootstrap of *identical* agents drawing from that seed's own per-type rates.

- observed worst-case per-type gap: median=0.100, mean=0.121, max=0.200
- percentile within the **seed-matched identical-agent null**: median=2th, max=38th (seeds above 95th = 0/10)
- the observed differentiation is **statistically indistinguishable from the spread identical agents produce by sampling noise alone** — no seed exceeds the null's 95th percentile.
- **One-sided non-superiority (SESOI=+0.10 excess over seed-matched noise floor): mean excess=-0.177 (n=10), p=0.0000 → observed differentiation does NOT exceed the noise floor and is in fact negative (agents MORE alike than independent draws). NB the independent-sampling null OVER-states the floor (real agents are correlated), so this is a conservative bound.**

## Verdict

- **Calibration**: under the null the test is mildly anti-conservative (rejects 0.08 at alpha=0.05, ~1.6x), i.e. biased *toward* detecting differentiation.
- **Power** (4 agents, ~41 obs/cell, base ~0.8): ≥0.80 by a 0.15 max-min per-type gap; at the *observed* median gap (0.10) power is only ~0.47, so an effect THIS small would be caught only about half the time — the omnibus p-value alone cannot carry the null at this effect size.
- **Omnibus**: the LR never rejects on real data (median p=0.99, 0/10 seeds).
- **Effect size (the load-bearing argument)**: the observed worst-case gap (median 0.10) falls at the 2th percentile of a *seed-matched identical-agent null* — it is at or BELOW the spread identical agents produce by sampling noise, an assumption-free comparison that does not depend on the power curve.
- **Conclusion:** for differentiation of meaningful size (≥0.15 gap) the test is well-powered and the omnibus is silent; for the trivially small effect actually observed, the seed-matched bootstrap shows it is indistinguishable from (indeed below) identical-agent noise. The null is a **true negative for meaningful differentiation**, not an artifact of low power. (We cannot exclude a trivially small <0.10 gap, but that is not the functional specialization the thesis sets out to detect.)
