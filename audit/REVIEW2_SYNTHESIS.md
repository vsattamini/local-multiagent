# Review-2 synthesis — results+methodology adversarial pass (2026-06-09)

4 agents: statistics, design/confounds, operationalization/metrics, integrity. This file
collects findings; actions taken after all 4 return + linchpin (ceiling-masking) resolved.

## From: Operationalization / metrics agent ✅ (numbers independently recomputed from task_logs)
NEW positive results (computable, no new runs):
- **P2 — ensemble utility (buried positive):** any@N − pass@1 = het **+20.3pp** [19.6,21.1]
  (pass@1 .656 → any@N .859; broad: string +24, math +21, logic +18, list +16).
  Homogeneous ens_3b +6.6pp is just pass@1→pass@4 self-consistency (majority −1.7pp, not multi-agent).
  → heterogeneous ensembling with test-selection is a REAL gain; add an ensemble-utility section.
- **P1 — style differentiation present in het:** co-correct solution Jaccard dist 0.51 (het) vs
  0.13 (homogeneous). Agents reach correctness via ~4× more divergent code when heterogeneous.
  solution_diversity.md only ran on ens_3b (homogeneous, null). Extend B+C to het_swarm_ensemble.
  → "no differentiation" overreaches; correct = "no type-dependent COMPETENCE differentiation";
    style differentiation tracks the competence basis (heterogeneity), absent among identical agents.
Reframes/corrections:
- **P0** functional-diff = success-rate interaction ONLY → near-foreclosed by frozen weights; scope the claim.
- **P3** S is REAL adaptive allocation (S_affinity>S_random, g≈1.1–1.4) — "artifact" is a definitional
  move; reframe as emergent division-of-labor; S can't tell monopoly (greedy S=0) from uniform →
  report active-agent count / load-Gini next to every S.
- **P4** D is DEAD: corr(D,pass@1)=−0.05, corr(D,S)=+0.17 across 395 seeds; demote to a one-line
  negative; DELETE the cap4 single-seed causal table (D=0.836↔S=0.390 was a single-seed coincidence).
- **P5 (CRITICAL=GAP 0)** chapters cap4/5/6 still state the refuted single-seed story (7B χ²=15.60 p=.008
  → actually 1/13 seeds; 3B "distinct functional profiles" → null; cap5 §5.5.1 still lists "single-seed"
  as a limitation though 10–20 seeds exist). Rewrite to the multi-seed/audit numbers.
- **P6 minimal metric set** to make "no specialization" airtight (all no-GPU): per-agent task-type
  ATTEMPT distribution under affinity; co-correct solution-Jaccard (both designs); active-agent
  count/Gini beside S; any@N/majority@N per-type; agent-identity-from-style classifier on het.

## From: Design & confounds agent ✅ — ⚠️ **P0 "FALSE NULL" CLAIM BELOW IS ITSELF RETRACTED (see note)**
> **SUPERSEDED:** the design agent's "p≈6e-7 BOMBSHELL / false null" (and my matching seed-unit t=12.3)
> are **pseudo-replication** — the 10 seeds reuse the same 164 problems and the two 1.5B agents share
> weights (within-task φ=0.69). Under correct **task-clustered** inference the het interaction is **NS**
> (ANOVA F=2.57 p=0.056; GEE cluster=task_id p=0.082) → het_interaction_result.md **v3** is canonical:
> uniform competence MAIN effect only, NO type interaction. The design agent's *methodological* catch
> (per-seed vote-counting is the wrong test) was correct; its *proposed fix* (naive pooling) was not.
- **P0 (BOMBSHELL, VERIFIED):** the het "interaction 0/10" was per-seed VOTE-COUNTING; each seed's
  `logic` cell ~10 obs/agent → underpowered, but the effect is consistent across seeds. Correct
  test (seed = unit / pooled / mixed-model): het interaction **p≈6e-7** (my seed-unit t=12.3),
  pooled logit p=2.2e-6, MixedLM(size*type,(1|seed)) p=2.6e-5. Per-type 3B−1.5B gap: string
  +0.33, math +0.22, list +0.22, **logic −0.05**. Control 3×3B: null (C=0.009, p=0.77).
  → het_interaction_result.md CORRECTED. "no differentiation in ANY condition" was WRONG.
  BUT interpretation: this is model-SCALE competence-by-type (scaling fact), NOT emergent roles
  (ensemble has S≡0, router moot → no division of labor). Identical-agent RQ answer unchanged: NO.
- **P1 (scoping):** the ENSEMBLE definitionally cannot test "emergent roles" — every agent does
  every task (S≡0 verified), router moot; S (allocation) and the interaction are measured in
  DISJOINT designs, never co-observed. The role-forming design (single-assignment + affinity +
  heterogeneous + replicated seeds) was never run with a computable interaction stat.
- Context convergence is NOT a problem (D stays 0.36–0.72) — design sound there.
- `logic` is a fallback catch-all (6 explicit labels) carrying the whole interaction → refine
  taxonomy / add label-free latent-axis (difficulty/length/AST) test. NO GPU.
- het_15_7b (max gap) queued but NOT yet run — predicted to amplify the (real) interaction.

## From: Integrity & reproducibility agent ✅ — **DATA CLEAN; only the prose is stale**
- het (10/10 main), ens_3b_n3 (8 valid), ens_7b (5/5) reproduce EXACTLY from raw logs.
- **0.00% fence-bug** across all 15 phase3 configs (~46k rows); contaminated logs were genuinely
  wiped+re-run. Decisive runs all clean.
- 4 partial seeds from crashes/rc=-6 (ens_3b_n3/1819, mbpp_3b_baseline/2021, mbpp_3b_lowtemp/1011,
  rand_1.5b/123) all lack final_metrics.json → correctly EXCLUDED, never counted.
- Seeding non-degenerate (distinct solutions per agent); het loads 3 distinct GGUFs.
- All 135 phase3 final_metrics reproduce; audit docs mutually consistent.
- **Only issue = GAP 0:** cap4–6 still print single-seed S=0.390/three-zones/−3.6pp trade-off
  (truth: S=0.334, +3.0pp, LR null) and list ensemble as "future work" though it's done.

## From: Statistics & inference agent ⏳ (will formalize the pooling/power/ceiling-masking verdict — core already confirmed independently)
## From: Design & confounds agent ⏳ (does the ensemble definitionally preclude role specialization? context convergence? taxonomy/heterogeneity-strength)
## From: Integrity & reproducibility agent ⏳ (recompute het/control/7B; fence-bug & crash-truncation scan; seed determinism; cross-artifact/chapter consistency)
