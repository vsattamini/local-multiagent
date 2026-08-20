# DECISIVE RESULT — Heterogeneous-ensemble functional differentiation (2026-06-09)

> **CANONICAL VERDICT (v3 — read this; the v1/v2 blocks below are RETRACTED, kept for provenance):**
> Under correct **task-clustered** inference there is **NO** significant agent×type / size×type
> interaction (ANOVA F=2.57, p=0.056; GEE cluster=task_id p=0.082) — a **powered true negative**.
> The het result is a **UNIFORM competence MAIN effect** (10/10 seeds, 3B>1.5B everywhere) only.
> The v1 "interaction 0/10" (per-seed vote-count) AND the v2 "p≈6e-7 / t=12.3 interaction real"
> (pseudo-replication: shared 164 problems + shared 1.5B weights, within-task φ=0.69) are BOTH
> wrong — **cite neither**. Identical-agent emergent differentiation: null and well-powered.

het_swarm_ensemble (2×1.5B + 1×3B, every-agent-every-task) vs ens_3b_n3 control (3×3B,
weight-only-different). Per-seed logistic decomposition: agent MAIN effect (does an agent
differ in OVERALL success?) and agent×type INTERACTION (does an agent's advantage DIFFER by
task type? = genuine functional specialization).

| Condition | n | agent MAIN effect sig | agent×type INTERACTION sig | median per-type gap |
|---|---|---|---|---|
| HET [2×1.5B+1×3B] | 10 | **10/10** | **0/10** | 0.378 |
| CONTROL [3×3B]    |  8 |   0/8   |    0/8    | 0.100 |

## Interpretation (decisive)
- Heterogeneity produces a LARGE, robust agent MAIN effect (10/10) — the 3B is UNIFORMLY
  better. The big raw per-type gap (0.378 ≈ the "~0.41" predicted) is **competence, not
  specialization** — exactly the main-effect-vs-interaction trap the foil review flagged.
- The agent×type INTERACTION is NULL (0/10) — the strong model's advantage does NOT vary by
  task type, so there is NO genuine functional specialization even with heterogeneous weights.
- The CONTROL confirms the main effect is weight-driven: identical 3B agents show neither
  main effect nor interaction (0/8, 0/8).
- The earlier "live rejection signal" (1.5B>3B on `logic` sign-flip) did NOT produce a robust
  interaction across seeds — it was noise.

## Bottom line (now firm, het test settled — did NOT flip)
Across IDENTICAL and HETEROGENEOUS agents, on HumanEval (and MBPP+ headroom), with ensemble
(computable LR), router-temperature/population/context/persona interventions:
- **Allocation emerges** (routing concentration S, small, temperature-driven).
- **Genuine functional (type-based) differentiation does NOT emerge** in any condition.
- Heterogeneity yields a uniform competence advantage (main effect), not specialization.
"Differentiation requires heterogeneity" is therefore also refuted: heterogeneity gives a
main effect, not an interaction.

---
## ⚠️ CORRECTION (2026-06-09, after Review-2 design+stats audit + independent re-verification)

The "INTERACTION 0/10" above was a **per-seed vote-counting artifact**: each single seed's
`logic` cell has only ~10 obs/agent (logic is the smallest, fallback-labeled category), so the
per-seed LR is underpowered — but the type-dependent effect is CONSISTENT across the 10
independent seeds. Treating **seed as the independent unit** (the correct test), the het
interaction is strongly significant:

- HET per-seed interaction contrast C = mean(string,math,list large−small gap) − (logic gap):
  mean **0.305**, one-sample t over 10 seeds **t=12.3, p=6.3e-7 → INTERACTION REAL**.
- Per-type large(3B)−small(1.5B) gap: string +0.326, math +0.223, list +0.216, **logic −0.050**.
- CONTROL ens_3b_n3 (3×3B identical): C=0.009, p=0.77 → **cleanly null**.
- Converges with design-agent pooled logit p=2.2e-6 and MixedLM(size*type,(1|seed)) p=2.6e-5.

### Corrected interpretation (scoped — protects the actual RQ)
- This interaction is a **model-SCALE competence-by-type fact** (the 3B's edge over 1.5B varies
  by task type), NOT emergent role specialization: the ensemble has no division of labor
  (S≡0, router moot), so there is no role formation to detect — only frozen per-model skill.
- **Identical agents → NO emergent differentiation** (type/competence/style): homogeneous
  controls null and well-powered (lr_power_validation.md). The thesis's RQ answer is unchanged: NO.
- **Heterogeneous agents** trivially differ by competence (scaling) AND give a real **+20pp
  any@N ensemble gain** (solution_diversity / REVIEW2_SYNTHESIS P2) — substrate heterogeneity
  (engineering), not emergence.
- CAVEAT: the interaction lives in the `logic` cell, which is a noisy fallback label
  (only 6 explicit logic problems) — refine taxonomy / add a label-free latent-axis test before
  naming the mechanism.

**Bottom line (corrected):** "no differentiation in ANY condition" was WRONG. Correct:
identical agents show no emergent specialization (null, well-powered); heterogeneous agents show
a genuine type-dependent competence interaction (a scaling fact, not emergent roles) plus useful
ensemble diversity. The earlier per-seed 0/10 must not be cited.

---
## ✅ FINAL (v3, 2026-06-09) — supersedes the v2 "interaction real" note above

v2 was ALSO wrong: its pooled p=2e-6 (and a seed-unit t=12.3) are PSEUDO-REPLICATION —
the 10 seeds reuse the SAME 164 problems and the two 1.5B agents SHARE weights (within-task
φ=0.69), so neither "independent obs" nor "independent seeds" holds. Under the CORRECT unit
(task_id / cluster-robust), the het interaction is NOT significant:

- ANOVA per-problem gap ~ type (task as unit, n=164): **F=2.57, p=0.056 (NS)**; logic-vs-rest p=0.078 (NS).
- GEE cluster=task_id: p=0.082; cluster-bootstrap strong:logic CI [−3.08,0.93]; Fisher-combined p=0.77; logic sign-test 6/10 p=0.75 — ALL NS.
- Design IS powered (≥0.86 for the observed 0.376 spread; calibrated at δ=0) and NO type is near ceiling (max rate 0.73) → this is a TRUE negative for a meaningful interaction, not low power.

**VERDICT (settled):** No significant type-dependent (agent×type / size×type) interaction in any
condition under proper task-clustered inference — including the heterogeneous swarm. The het
result is a large, UNIFORM competence MAIN effect (3B > 1.5B everywhere, 10/10) with NO
generalizing type interaction (a borderline logic dip, p≈0.06, is a non-significant trend, not a
finding). Identical-agent emergent differentiation: null and well-powered.

**Methodological lesson (for the write-up):** prove the null with the **cluster-robust (GEE,
cluster=task_id) interaction test + power curve**, NOT the per-seed "0/10" vote-count — the naive
"pool for power" check yields a SPURIOUS p<0.0001 and must be avoided. Separate real positive
result stands: heterogeneous ensembling gives +20pp any@N (solution_diversity / REVIEW2 P2).
