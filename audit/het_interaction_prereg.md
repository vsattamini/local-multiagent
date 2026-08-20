# Pre-registration — het_swarm_ensemble: main effect vs specialization

Simulation establishing which statistic answers the thesis question BEFORE the decisive run lands, so the analysis is pre-committed (anti-forking-paths).

## Section 1 — main effect vs interaction (moderate base rates, 41 obs/cell)

Three agents (agent 2 = strong/3B-like, +1.2 log-odds), 200 sims/scenario. Moderate base rates (no ceiling) so the saturated logit MLE is estimable.

| scenario | agent MAIN-effect p (median) | agent×type INTERACTION p (median) | INTERACTION reject-rate | raw success-rate max-min gap (median) |
|---|---|---|---|---|
| H0_competence | 0.000 | 0.439 | 0.04 (n=200) | 0.390 |
| H1_specialize | 0.480 | 0.000 | 0.97 (n=200) | 0.341 |

## Section 2 — ceiling inflates the raw gap under PURE competence (no interaction)

At the observed high ens_3b base rates, give the strong agent a CONSTANT +1.2 log-odds on every type (uniform competence, exactly zero interaction). Expected per-type success rates and the resulting raw max-min gap (no sampling, just arithmetic):

| type | weak agent rate | strong agent rate (+1.2 logit) | per-type gap |
|---|---|---|---|
| string | 0.82 | 0.94 | +0.12 |
| math | 0.79 | 0.93 | +0.14 |
| list | 0.89 | 0.96 | +0.07 |
| logic | 0.66 | 0.87 | +0.21 |

- raw max-min per-type gap = **0.21** — large, yet there is **zero** agent×type interaction by construction. The gap is biggest on the HARDEST type (logic) and compressed on near-ceiling types (list), which is exactly the spurious 'profile difference' a naive success-rate reading would mislabel as specialization. (This is also why the reviewer's predicted 'per-type gap ~0.41' is consistent with NO specialization at all.)

## Pre-registered interpretation rules for het_swarm_ensemble

- **The differentiation claim rests ONLY on the agent×type INTERACTION LR (logit scale).** Under uniform competence (H0) the interaction test stays at ~alpha — it does NOT false-positive on a pure size advantage — while the agent MAIN effect is strongly significant and the raw success-rate gap is large. So:
- A significant **main effect** in het_swarm is EXPECTED and is NOT specialization (it just says the 3B is better — trivially true).
- A large **raw per-type success-rate gap** is NOT sufficient evidence: under uniform competence + ceiling it is already inflated. Do not report it as differentiation without the interaction test.
- Only a significant **interaction** (each agent differentially better at different types) supports 'functional differentiation emerged'. The matched homogeneous control **ens_3b_n3** must show NO interaction (it already does in ens_3b); het showing interaction where ens_3b_n3 does not = heterogeneity-driven specialization, the cleanest possible positive result.
