# CORRECTION — the mech_7b_rrpersona "persona penalty" is NOT a robust interaction (2026-06-09)

> **CANONICAL VERDICT:** Under the correct **task-clustered** inference (GEE, cluster=`task_id`,
> controlling task-type difficulty) the round-robin-persona own-type penalty is **NOT significant
> (own coef = −0.503, p = 0.24; drop-logic p = 0.38)**. The headline "−0.042, t = −5.45, p = 0.0004"
> is **pseudo-replication** — the identical artifact class already retracted for the het interaction
> (see `het_interaction_result.md` v3). It must NOT be cited as "the only robust interaction" or as
> evidence of a persona-induced specialization penalty. Documented here because the number lived
> ONLY in `hostile_5d_final.py` console output, with no prior doc of record.

## What was claimed (and why it is wrong)

The hostile pass reported, for `mech_7b_rrpersona` (round-robin persona assignment, 7B):
- within-agent own-type diagonal mean = **−0.042, t = −5.45, p = 0.0004, signs 0/10**;
- survives naive drop-logic (mean −0.042, t = −5.10, p = 0.0006).

Those numbers reproduce **exactly**. They are nonetheless invalid as a generalizing finding:

1. **Round-robin is fully deterministic.** 0 of 164 `task_id`s ever go to a different agent across
   the 10 seeds — only decode sampling differs. So the per-seed t over 10 seeds reuses the
   **identical (agent, problem) pairs**; it is pseudo-replication on the same 164 problems, the
   exact error condemned for the het interaction. The contrast is not generalizing across 10
   independent units.
2. **The logic own-cell is degenerate.** agent3's "own logic" is the **same single problem
   (HumanEval/95) in all 10 seeds** (2/10 pass). The −0.65 logic diff is one repeated problem.

## Correct task-clustered result

- **GEE, cluster = `task_id`, controlling type difficulty:** own coef = **−0.503, p = 0.24 (NS)**;
  drop-logic **p = 0.38 (NS)**.
- **Proper between-agent within-type contrast** (fully removes difficulty): −0.197 (all) /
  −0.048 (drop-logic), still negative per-seed but **collapses to p = 0.14 (NS)** once both logic
  AND math are dropped. The residual signal is carried almost entirely by the **math** cell
  (agent1 own-math 0.72 vs others 0.81) — a single cell, not a type-general pattern.

## Why this STRENGTHENS the thesis (it does not weaken it)

The persona penalty was the one place the corpus claimed a "robust interaction." Removing it makes
the null cleaner, not muddier: **no condition** — identical agents, heterogeneous agents, router
temperature / population / context-filter sweeps, OR adversarial round-robin **persona** prompts —
produces a task-clustered agent×type interaction. Even when we explicitly *push* agents toward
distinct roles via persona, no generalizing functional specialization survives proper inference.

## Action items
- [x] Document the corrected clustered result here (was citation-orphaned in `hostile_5d_final.py`).
- [ ] In any chapter that discusses persona/mechanism interventions: report the penalty as **NS
  under task-clustering (GEE p=0.24)**, not "p=0.0004 robust"; frame as "even persona prompts do
  not induce a robust interaction" — additional support for the null, not a standalone finding.
- [ ] Throughline (same as het): prove/deny interactions with cluster-robust GEE on `task_id`,
  never per-seed vote-counts or per-seed t over shared problems.
