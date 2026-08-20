# GAP 3 — S: affinity-driven concentration vs RandomRouter floor

Per-seed specialization index S for the affinity router vs the matched RandomRouter control (same model/n_agents/router_temp). EXPLORATORY (controls pre-registered as H2 deconfound, but contrast not in original plan). p-values Holm-corrected across the 3 size-tests (family) to control forking paths.

| size | S_affinity (mean±sd, n) | S_random (mean±sd, n) | ΔS | 95% CI | Hedges g | MWU p | Holm p |
|---|---|---|---|---|---|---|---|
| 1.5B | 0.048±0.022 (n=20) | 0.020±0.013 (n=10) | +0.028 | [+0.017,+0.040] | +1.40 | 0.001 | 0.004 |
| 3B | 0.042±0.022 (n=20) | 0.019±0.011 (n=10) | +0.023 | [+0.012,+0.035] | +1.15 | 0.006 | 0.012 |
| 7B | 0.037±0.019 (n=20) | 0.019±0.011 (n=10) | +0.019 | [+0.009,+0.029] | +1.08 | 0.015 | 0.015 |

## Reading

- ΔS > 0 with CI excluding 0 ⇒ the affinity mechanism concentrates routing beyond the random floor: real **task allocation / division of labor**.
- This is the metric S was *designed* to capture, and is logically distinct from functional (success-rate) differentiation, which the LR test finds absent (see lr_power_validation.md). Allocation emerged; competence differentiation did not.
