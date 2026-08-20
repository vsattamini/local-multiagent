# Fan-out synthesis — opportunities & new runs (2026-06-07)

Four parallel agents combed the project. This file collects actionable items;
NEW RUNS get queued into a phase-3 orchestrator (after rrpersona, TB last).

## From: Metrics & Statistics agent  ✅
### NEW RUNS (GPU)
- **[QUEUE] Causal context-shuffle test** (Tier-2, #1 highest-leverage new run):
  reconstruct each agent's FINAL context buffer (last ≤5 successes per agent from
  task_log.jsonl + problem text via loader); re-infer all 164 with routing fixed
  to observed, under 2 conditions: (control) own final buffer, (treatment)
  swapped/scrambled buffers; recompute Pass@1 & S; feed to
  RobustnessMetrics.context_shuffle_sensitivity. Conditions: 3B-baseline,
  3B-low-temp, 7B. ≥5 seeds × 2 modes. ~single-digit GPU-h. Tests whether
  specialization is CAUSAL for performance or epiphenomenal.

### NO-GPU RE-ANALYSES (do in the chapter-rewrite pass)
- Between-condition inference: Welch t / Mann-Whitney on per-seed S & Pass@1;
  Hedges' g + bootstrap CI; **Holm-Bonferroni** family correction; retrospective power.
- Temporal dynamics from snapshots.jsonl: snapshot S/D are CUMULATIVE → reconstruct
  WINDOWED S from task_log; fit constant/saturating/linear/change-point per seed,
  compare by AIC; report time-to-stabilize τ + change-point distribution. NB: 3B
  low-temp S DECAYS 0.67→0.38 (relaxation, not buildup) — current "gradual_drift" label wrong.
- **TOST equivalence tests** on per-seed LR effects → make the NULL functional-
  differentiation result rigorous (not just p>0.05). Note 7B "1/10 sig" == 5% FP rate.
- D predicts nothing: pooled (n=130) corr(D,S) r=−0.07, corr(D,Pass@1) r=+0.13,
  corr(S,Pass@1) r=−0.12 — all null; D flat ~0.66 across all conditions. Revise cap4 §4.4.3
  and cap3 ("D>0.3 required", "D crescente indica diferenciação") — both unsupported.
- Bootstrap/BCa (and optional Bayesian BEST + ROPE) CIs on the CONTRASTS, not just cells.

### SCOPE-ONLY (future work, do not run)
- PID / time-delayed MI (Riedl): infeasible here (one agent per task → no synchronous
  multi-agent state; 16 coarse snapshots data-starved). Optional caveated transfer-entropy
  on per-agent success streams. Real PID needs a coalition/voting design (see exp-levers agent).

### CAP4 single-seed artifacts to CORRECT (multi-seed already in repo)
- S 0.390 → 0.334 [0.281,0.386] (3B low-temp)
- D 0.836 → 0.666 (3B low-temp; == baseline)
- ΔPass@1 −3.6pp → ≈+0.4pp (1.5B low-temp vs baseline) — central trade-off narrative at risk.

## From: Experimental-levers agent  ✅
**KEY (verified): single-assignment is the structural cause of LR non-computability** — even
balanced round-robin (41/41/41/41) → singular interaction matrix (each agent sees each problem
≤once). Fix = ensemble. (So rrpersona also can't give clean LR — but its χ²/Cramér's V up to
0.44 on 1.5B persona DOES show the count-apparatus detects induced differentiation → keep as control.)
> NOTE: the separate per-seed "7B rrpersona own-type penalty p=0.0004" is RETRACTED as
> pseudo-replication (deterministic round-robin reuses the same 164 problems; logic cell = a single
> repeated problem). Task-clustered GEE → p=0.24 NS. See `persona_penalty_correction.md`. The
> Cramér's V positive-control claim on THIS line is unaffected.
### NEW RUNS (GPU), priority order
- **[QUEUE #1] Ensemble / k-of-N assignment** (CODE: assignment loop in experiment.run; log one
  row per (agent,task); ExperimentConfig `ensemble_assignment="all"|k`, `vote`). Every problem
  attempted by all N agents → identifies GLMM `success ~ type*agent + (1|task_id)`. Models 3B
  (n=10) + 7B (n=5); n_agents 4; type_filtered K=20. Also yields voting metrics (#2).
  → extend analyze_glmm.py with mixedlm/(1|task_id). Closes the central methodological gap.
- **[QUEUE #2] Ensemble VOTING accuracy** (same runs): report pass@1 vs any@N vs majority@N —
  first chance swarm beats single-model. Models 1.5/3/7B, n_agents 3,5.
- **[QUEUE #3] Heterogeneous swarm** (CODE: per-agent model handle; YAML `models:` list).
  2×1.5B+1×3B (fits 8GB; avoid 3B+7B+1.5B). affinity τ0.3, n=10. Most likely GENUINE emergent
  differentiation (router → strong model into high-value niche). Strong potential positive result.
- **[QUEUE #4] Greedy router** multi-seed (NO code): 3B, n_agents3, router=greedy, n=10. Fills router-policy axis.
- **[QUEUE #5] Large-N population** multi-seed (NO code): 1.5B, n_agents∈{8,12,16}, τ0.1, type_filtered, n=10. Replaces discredited single-seed pop sweep.
### NO-GPU
- Difficulty stratification: bin 164 tasks by cross-seed/model mean success → re-run S/LR analysis with difficulty as grouping (free); optional difficulty-taxonomy run.
### SCOPE-ONLY: epochs (#6), cooperative context sharing (#7).
### CAP4 note: state explicitly that LR non-computability is STRUCTURAL to single-assignment (justifies ensemble pivot, strengthens the honest null).

## From: Benchmark-headroom agent  ✅
- **[QUEUE] MBPP+ (EvalPlus)** = best next benchmark. 1.5B~59/3B~62/7B~72 pass@1 (headroom), 378 tasks,
  assert-based → reuses HumanEvalExecutor, keeps 4 task types (apples-to-apples). CODE: vendor
  data/MBPPPlus.jsonl, src/swarm/mbpp.py loader, scripts/categorize_mbpp.py, execute_mbpp wrapper,
  `benchmark` field in experiment.load_tasks. ~half day. Run core conditions (baselines + low-temp, 3 sizes).
- BigCodeBench-Full (stretch, most headroom + natural library-domain task types) — Docker executor +
  extend TaskType enum. ~2-3 days. Defer/optional.
- **SWE-bench: RETIRE for differentiation** — measured 0% resolved on 1.5B. Floor, not headroom.

## From: Code-validity agent  ✅  (headline findings ROBUST; confounds bias toward spurious differentiation → null reinforced; contamination CLEAN)
- **H1 [CODE FIX]** 0.5 prior persisted in stored affinity_matrix/task_type_performance → fabricated "0.5 generalist" profiles. Report None+n_attempts; recompute cap4 qualitative profiles from task_log only.
- **H2 [QUEUE control]** cold-start lock-in manufactures S + tiny-n separation cells → **RandomRouter multi-seed** (matched configs, 3 sizes); report S_affinity − S_random.
- **M1 [CODE FIX]** one decode seed reused for all 164 gens AND seed couples routing+decoding → decouple (decode seed = seed+task_index).
- **M2 [CODE FIX]** `_clean_solution` keeps out-of-fence prose → can fail CORRECT 1.5B code → extract only inside first fence.
- **M3 [CODE FIX]** two divergent categorization maps + dead `search` bucket → single source of truth.
- **M4 [QUEUE sensitivity]** noisy keyword categorizer (114 labels) → re-categorization sensitivity to show null is label-robust.
- L1 logic n=10, L2 perm-null RNG, L3 FIFO buffer type-agnostic → caveats.

---
# CONSOLIDATED NEW-RUN QUEUE (priority)
1. **Ensemble/k-of-N** (3B n=10, 7B n=5) — CODE — real GLMM + any@N/majority@N voting. CENTRAL.
2. **Heterogeneous swarm** 2×1.5B+1×3B (n=10) — CODE — best shot at GENUINE differentiation.
3. **RandomRouter control** (3 sizes baseline, n=10) — NO CODE — deconfounds S (H2).
4. **MBPP+ core** (3 sizes × baseline+low-temp, n=10) — CODE(new files) — headroom.
5. **Context-shuffle causal test** (3B-base, 3B-low-temp, 7B) — CODE(standalone) — causal?
6. **Greedy router** (3B) + **Large-N pop** (1.5B n_agents 8/12/16) — NO CODE — cheap fills.
CODE FIXES before new runs: H1, M1, M2, M3. NO-GPU re-analyses (rewrite): Holm/effect-sizes/power, TOST(null), temporal dynamics, D-predicts-nothing, bootstrap CIs.
