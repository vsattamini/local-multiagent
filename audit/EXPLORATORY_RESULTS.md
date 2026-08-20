# Exploratory results — registered artifact (NO-GPU, post-hoc)

Reproduce: `python audit/exploratory_analysis.py`. All numbers EXPLORATORY (not pre-registered). Companion to `audit/ADVERSARIAL_FOIL_REVIEW.md`.

## 1. Ensemble voting on ens_3b (homogeneous 3B, every agent every task)

- seeds: 9 · agents/task: 4 · tasks: 164
- **pass@1 (mean per agent)**: 0.823  [95% CI 0.797, 0.848]  (n=9)
- **any@N == test@N (keep any soln passing the test suite — DEPLOYABLE)**: 0.885  [95% CI 0.868, 0.902]  (n=9)
- **majority@N**: 0.808  [95% CI 0.777, 0.839]  (n=9)
- **any@N − pass@1 (voting gain)**: +6.2pp  [95% CI 5.0, 7.4pp]
- majority@N − pass@1: -1.5pp  [95% CI -2.3, -0.7pp]  (majority voting does NOT beat single-shot — only oracle/test-filtered selection does)

## 2. Between-agent success disagreement per task

- tasks where agents disagree on pass/fail: **13.7%**  [95% CI 11.4, 15.9%]
- => on ~86% of tasks all agents agree (all pass / all fail). With identical
  weights on a near-saturated benchmark, the only between-agent variance is decode
  noise, which rarely flips success => the agent x type interaction has almost nothing
  to bite on. The LR null is structural, not a power failure.

### per-type pass@1 vs any@N (mean across seeds)
| type | pass@1 | any@N | gain | ~n tasks |
|---|---|---|---|---|
| list | 0.890 | 0.934 | +4.4pp | 37 |
| logic | 0.722 | 0.756 | +3.3pp | 10 |
| math | 0.795 | 0.869 | +7.3pp | 76 |
| string | 0.837 | 0.902 | +6.5pp | 41 |

## 3. Specialization S: routing concentration vs chance

| design | router | assignment | n_agents | S (mean±sd, n) | note |
|---|---|---|---|---|---|
| exp_3b_baseline | affinity | single | 3 | 0.042±0.021 (n=20) | baseline |
| rand_3b | random | single | 3 | 0.019±0.011 (n=10) | CHANCE FLOOR |
| rand_7b | random | single | 3 | 0.019±0.011 (n=10) | chance floor |
| rand_1.5b | random | single | 3 | 0.017±0.011 (n=9) | chance floor |
| greedy_3b | greedy | single | 3 | 0.000±0.000 (n=10) | collapses to 1 agent |
| exp_3b_low_temp | affinity τ low | single | 3 | 0.334±0.109 (n=20) | low router temp |
| popN_8 | affinity τ0.1 | single | 8 | 0.421±0.105 (n=10) | large pop |
| popN_12 | affinity τ0.1 | single | 12 | 0.494±0.086 (n=10) | large pop |
| popN_16 | affinity τ0.1 | single | 16 | 0.513±0.102 (n=10) | large pop |
| ens_3b | round_robin | ENSEMBLE | 4 | 0.000±0.000 (n=8) | everyone does everything |

- affinity baseline S ≈ 0.042 vs random S ≈ 0.019 ⇒ affinity concentrates **2.2× above chance** (same n_agents=3, same router temp 0.5, both single-assignment).
- => S is NOT pure chance: affinity routing genuinely concentrates, low temp / large pop drives S to ~0.5. BUT this is ALLOCATIONAL specialization (who gets which task), with NO competence basis (identical weights) and NO functional consequence (LR null, pass@1 flat). Reclaim S as emergent division of labor; do not call it functional specialization.

## 4. LR-computability map (can the agent×type interaction test even run?)

| design | weights | assignment | LR computable? | can LR reject for a REAL reason? |
|---|---|---|---|---|
| ens_3b | homogeneous 3B | ensemble | YES | NO — agents symmetric ⇒ E[interaction]=0 ⇒ only Type-I |
| mech_*_typefilter | homogeneous | single | NO (perfect separation) | n/a |
| het_swarm (current) | heterogeneous | single | NO (perfect separation) | n/a |
| **het_swarm_ensemble (NEW, to run)** | **heterogeneous** | **ensemble** | **YES** | **YES — 3B genuinely better on some types ⇒ real interaction** |

=> The decisive cell (het weights + ensemble) is the ONLY one that can yield a non-trivial, computable LR. It has not been run. ens_3b is its matched homogeneous control.
