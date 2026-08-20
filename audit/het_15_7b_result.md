# Maximum-heterogeneity ensemble [1.5B, 7B] — RESULT (2026-06-09)

> **VERDICT:** The widest weight gap in the study (1.5B + 7B, every-agent-every-task ensemble, 10
> seeds) produces a **large competence MAIN effect (+25.4 pp) with NO significant agent×type
> interaction under task-clustered inference** (ANOVA F=2.15, p=0.096; GEE cluster=task_id p=0.129).
> Identical verdict to `het_interaction_result.md` v3 (het_swarm_ensemble [2×1.5B+1×3B], F=2.57,
> p=0.056). The dose-response ladder confirms: **scaling the weight gap scales the uniform competence
> advantage, NOT a type-dependent role.** Allocation S ≡ 0 (ensemble, router moot, as designed).

Config: `config/het_15_7b.yaml` (agents [1.5b, 7b], ensemble_assignment=all, ctx 4096, type_filtered
K20). 10 seeds, all complete (`results_phase3/het_15_7b/seed_*`).

## 10-seed aggregate
- pass@1: mean **0.701** (sd 0.013, range 0.683–0.729)
- S (routing concentration): **5.7e-11 ≈ 0** — ensemble design, router moot (as intended)
- Competence MAIN effect: agent0 (1.5B) success **0.574** vs agent1 (7B) **0.827** → **+25.4 pp**

## Agent×type interaction — canonical task-clustered test (NOT the per-seed χ²)
The per-seed functional-differentiation χ² is 0/10 significant, but that per-seed vote-count is the
underpowered statistic the project retired (see het_interaction_result.md). The correct test clusters
on the unit of replication (`task_id`, since 10 seeds reuse the same 164 problems):

| Test | het_15_7b | (cf. het_swarm_ensemble v3) |
|---|---|---|
| ANOVA per-task gap ~ type (task=unit, n=164) | **F=2.15, p=0.096 (NS)** | F=2.57, p=0.056 (NS) |
| GEE cluster=task_id, joint interaction Wald | **p=0.129 (NS)** | p=0.082 (NS) |
| drop-logic ANOVA | p=0.131 (NS) | p≈0.33 (NS) |

Per-type 7B−1.5B gap (task=unit): string **+0.361**, math **+0.263**, list **+0.168**, logic **+0.060**.

## Interpretation (honest, v3-consistent)
- There is a **non-significant gradient**: the 7B's competence edge trends larger on string/math than
  on list/logic. It does NOT reach significance under task-clustered inference (p≈0.10–0.13), exactly
  as in het_swarm_ensemble (p≈0.06–0.08). Two independent heterogeneity levels give the same
  borderline-but-NS pattern → a non-significant trend, not a finding. Do NOT describe het as "uniform
  across types" (the raw gaps visibly span 0.06–0.36); describe it as "a large competence main effect
  with no significant type interaction under proper clustering."
- This is **model-scale competence-by-task variation** (a frozen-weights scaling fact), NOT emergent
  role specialization: the ensemble has S≡0 (no division of labor), so there is no role to form.
- The dose-response (het_swarm_ensemble [2×1.5B+1×3B] → het_15_7b [1.5B+7B]): widening the weight gap
  grows the MAIN effect (+25.4 pp here vs the smaller 3B-over-1.5B edge) while the interaction stays
  NS. "Scale buys competence, not roles" holds at the maximum gap tested.

## Thesis use
cap4 §4.5: add het_15_7b as the dose-response confirmation rung. Frame as a NON-significant
type-gradient trend + a large competence main effect — consistent with the identical-agent null and
with het_swarm_ensemble v3. Reproduce via the snippet in this analysis (task-clustered ANOVA + GEE).
