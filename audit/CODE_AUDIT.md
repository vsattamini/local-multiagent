# CODE AUDIT — independent re-verification + synthesis (2026-06-09)

Auditor: resuming agent (prior audit `a558795f68e987943` was rate-limited before writing this).
Scope: READ-ONLY re-verification of the analysis-code layer for the identical-SLM-swarm
specialization study. Everything below was recomputed from scratch, not accepted on trust.
Environment: `.venv/bin/python`, statsmodels 0.14.6, `LD_LIBRARY_PATH` set for CUDA-linked libs.

---

## 1. Verdict table

| # | Claim under review | Verdict | Recomputed numbers |
|---|---|---|---|
| 1 | `final_metrics.json` gate excludes only genuinely-partial seeds | **CONFIRMED (with update)** | Of the alleged 4 partials, **3 are now COMPLETE**: `ens_3b_n3/1819` = **492 rows + FM** (not 132 — finished since prior audit), `mbpp_3b_baseline/2021` = 366 + FM, `mbpp_3b_lowtemp/1011` = 366 + FM. **Only `rand_1.5b/seed_123` (94 rows, no final_metrics)** is genuinely partial, and it is correctly excluded. All 14 hostile scripts gate on `final_metrics.json` (via `_hostile_loader.load_run` or an explicit check), so the partial seed never enters any analysis. |
| 2 | Het interaction p≈0.056 reproduces; attack-2c pseudo-hit collapses | **CONFIRMED** | Het interaction (task-as-unit one-way ANOVA on per-task 3B-advantage, grouped by task_type) = **F=2.57, p=0.0560** (reproduced in `hostile_4` `orig_4way`). Attack-2c: full cluster-robust stat obs=2.74, **p=0.0037**; after dropping the degenerate agent0-string n=5 cell → obs=0.29, **p=0.7141**. Collapse confirmed. |
| 3 | Attacks 3/3b: temporal "late>early" is a sparsity artifact | **CONFIRMED** | Raw: popN_12 late−early=+0.0124, p=0.0012; popN_8 p=0.022. Under within-type permutation null all reverse to NS / wrong-direction: popN_8 p=0.9991, popN_12 p=0.9578, popN_16 p=0.9197, het p=0.3767. All p>0.37. |
| 4 | Attack 4: all alternative taxonomies NS, Holm min adj-p≈0.448 | **CONFIRMED** | 8 partitions; raw min p=0.056 (orig_4way); **Holm min adj-p = 0.4480**; any significant: False. |
| 5 | Attack 6: any@N−pass@1=+0.203, genuine coverage +0.049 = rescue rate | **CONFIRMED** | mean pass@1=0.656, any@N=0.859 → **+0.203**; any@N − 3B-alone(0.810) = **+0.049**, t=16.06, p≈0. Rescue (3B fails, 1.5B succeeds) = **0.049** (== any@N − best-3B). Honest oracle best-of-N framing holds. |
| 6 | `validate_lr_power.py`: 4·delta/3 identity correct; deterministic re-run byte-identical | **CONFIRMED** | synth boosts own type by +delta, damps others by −delta/3 → max−min gap = 4·delta/3 (algebraically exact). Deterministic re-run with `RandomState(20260608)` is **IDENTICAL** to committed `audit/lr_power_validation.md` on all numeric lines (power 0.00→0.080, 0.15→0.905, 0.20→0.998; real data min p=0.961, median 0.993, frac<0.05=0). |
| 7 | `executor.py` strict `_clean_solution` correct on all 4 fence cases; fixes legacy prose-append bug | **CONFIRMED** | `test_executor_sanity.py` passes (correct/wrong_constant/wrong_inverted/raises). Direct `_clean_solution(clean_mode="strict")` on no-fence / leading-reopened / trailing-only / prose-after → all 4 return exactly the code body, no fence, no prose. |
| 8 | `experiment.py` `seed = random_seed + i*131 + agent_id` distinct per (agent,task) for <131 agents | **CONFIRMED** | Code at experiment.py:222. For a1,a2<131, `i1*131+a1 == i2*131+a2` ⟺ i1==i2 ∧ a1==a2. Distinct. `decouple_decode_seed` plumbed and serialized to run config. |
| 9 | `metrics.py` zero-marginal row/col drop fires only on degenerate tables | **CONFIRMED** | metrics.py:217-218 (`sum(axis)>0`). On real runs ens_3b (4×4) and het_swarm_ensemble (4×3) it drops **0 rows/0 cols** → dof unchanged on stored data. |

---

## 2. `scripts/analyze_glmm.py` fix verdict

**Correct & non-perturbing: YES. Guard fires on real separation: YES. `mle_retvals` caveat: none.**

- Docstring (lines 14-22) now states the truth: no Firth; non-convergence is DETECTED via
  `mle_retvals['converged']` and returns `{"error":"non-converged (separation)"}`; separated cells
  handled by exclusion + permutation tests.
- The convergence guard is present and logically correct in **both** `lr_test_logit`
  (lines 89-91) and `lr_main_effect_logit` (lines 120-122), checking BOTH null and full models.
- **Non-perturbing on a converged fit** (`results_phase3/ens_3b/seed_1011`): interaction
  **p = 0.9993679…** , main effect **p = 0.8408366…** — matches the homogeneous null (≈0.9994 / 0.8408),
  unchanged by the guard.
- **Guard demonstrably fires on real separation**: `results/exp2.1_experimental/seed_42` — the full
  interaction model returns `converged=False` (null converged=True), and `lr_test_logit` now returns
  `"non-converged (separation)"`. Without the guard it would have emitted p=0.7178 from an unreliable
  non-converged log-likelihood. Other separated single-assignment runs (`exp_3b_low_temp/seed_42`,
  `exp_low_temp/seed_42`) error EARLIER at the fit call with `Singular matrix` and are caught by the
  existing `except` — also correct (no bogus p). So the guard backstops exactly the residual case
  where statsmodels returns a model object instead of raising.
- **`mle_retvals` caveat**: for statsmodels 0.14.6, `Logit` results expose `mle_retvals` as a plain
  `dict` containing key `"converged"` (verified: `type=dict`, `'converged' in mle_retvals == True`).
  The `.get("converged", True)` default is appropriate and never silently defaults on these models.

---

## 2b. any@N coverage CI — provenance anchor

The het ensemble coverage gain over the **best single agent** is **+4.9 pp, 95% CI [4.3, 5.5]**
(pass@1 0.656 → any@N 0.859; best agent = 3B at 0.810; t=16.06, p≈0). The point estimate + t/p are in
the verdict table (row 6 / stats re-verification CLAIM 3); the **[4.3, 5.5] interval is the seed-matched
bootstrap CI from that same CLAIM 3** — recorded here so the figure cited in
`thesis/cap4_resultados.md §4.5.2` traces to a committed artifact, not only an agent transcript.

---

## 3. Persona-penalty adjudication

**Pseudo-replication: YES. The per-seed t-test is invalid as a generalizing finding. The stats
re-verifier is right; `audit/persona_penalty_correction.md` is ACCURATE.**

Independently recomputed on `mech_7b_rrpersona` (10 seeds, 1640 rows):

- **(a) Round-robin is fully deterministic across seeds:** of 164 task_ids, **0** map to a different
  agent_id in any seed (only decode sampling differs). The 10 per-seed diagonal means are repeated
  measurements of ONE fixed (agent,problem) assignment over the SAME 164 problems — not 10
  independent draws.
- **(b) The logic own-cell is one repeated problem:** agent3 (logic persona) attempts exactly
  **1 distinct logic task_id (HumanEval/95)**, repeated 10× (10 rows). The −0.65 logic diff is a
  single problem. (agent0/string=6 ids, agent1/math=17, agent2/list=10.)
- The per-seed t-test reproduces exactly: own−other diagonal = **−0.042, t=−5.45, p=0.0004, signs 0/10**.
- **Correct task-clustered inference (GEE, cluster=task_id, controlling type):** own coef = **−0.503,
  p = 0.239 (NS)**; drop-logic own coef = **−0.387, p = 0.383 (NS)**. Matches the correction doc's
  −0.503/p=0.24 and p=0.38.

Conclusion: the "−0.042, t=−5.45, p=0.0004" is the same pseudo-replication artifact already retracted
for the het interaction (per-seed vote-counting / t over shared problems). Under proper task-clustering
the persona penalty is NOT significant. Persona prompts do not induce a robust agent×type interaction.
This strengthens the null. Persona conclusion correctly rests on the truly-homogeneous ens_3b_n3
(perm p=0.9993) and ens_7b (perm p=0.9970).

---

## 4. New / minor issues found (ranked; headline-changing flag)

1. **Stale "132-row partial" claim in the task prompt / prior notes — headline-changing: NO.**
   `ens_3b_n3/seed_1819` is now COMPLETE (492 rows + final_metrics). It was already correctly gated,
   so no analysis used a partial copy; the gate would have excluded it when partial. No data change
   needed; just note the run finished.

2. **attack-5c code comment mislabels `het_swarm_ensemble` as a "null control" — headline-changing: NO.**
   Confirmed `het_swarm_ensemble` config has genuine model heterogeneity (2×1.5B + 1×3B), so its
   interaction perm p=0.0003 is EXPECTED model-size scaling, not a homogeneous-null violation. The
   label is a comment inaccuracy only; the persona/null conclusion rests on the homogeneous ens_3b_n3
   / ens_7b controls, which are both null (p≈0.997). Recommend fixing the comment text; no result moves.

3. **attack-5c TEST 4 / TEST 3 reuse the same per-seed (unclustered) tests and the degenerate logic
   n=10(=1 problem) cell — headline-changing: NO.** Same pseudo-replication class as 5d, already
   covered by `persona_penalty_correction.md`. The displayed persona "penalties" (e.g. logic
   diff −0.38/−0.59, p=0.037/0.001) are driven by the single-problem logic cell; they should NOT be
   cited as robust without task-clustering. Already documented; no new exposure beyond the persona
   correction.

4. **attack-1 k-means difficulty test is latently circular — headline-changing: NO.** It clusters on
   per-agent success then tests the 3B advantage across clusters; author flags it as a fishing
   expedition. Confirmed it is not load-bearing for any reported conclusion (the null rests on the
   LR/GEE/permutation analyses, not attack-1).

5. **`has_loop` median-split can NaN on a binary feature — headline-changing: NO.** Benign,
   Bonferroni-corrected exploratory branch; does not feed a headline.

6. **`data/mbpp_difficulty_axis.json` — headline-changing: NO.** Valid JSON, 366 entries, fields
   {task_id, prompt_chars, prompt_tokens, ast_nodes, cyclomatic, canon_lines, difficulty_z,
   difficulty_tercile}. **NOT referenced by any `.py`** (confirmed by grep). Pre-registration artifact
   for a future xval_roles experiment; nothing depends on it yet.

**Adversarial checks that came up CLEAN (no new issue):**
- **agent_id→type label alignment**: `mech_7b_rrpersona.yaml` personas are ordered
  agent0=STRING, agent1=MATH, agent2=LIST, agent3=LOGIC — exactly matching the analysis mapping
  `PT={0:string,1:math,2:list,3:logic}` used in hostile_5b/5c/5d. No misalignment.
- **Permutation reproducibility**: all hostile permutation tests seed `np.random.default_rng(0)`
  (or `RandomState(20260608)` for the power script) deterministically. Reproducible.
- **Silent caps / truncation**: no sampling caps in any analysis loader; only cosmetic string
  truncations (`[:6]`, `[:60]`, `problem[:100]` for storage). No row/observation is silently dropped.
- **Non-independent units**: every place a p-value is computed on a non-independent unit (per-seed
  over deterministic assignments) is now either retracted (het v3) or corrected (persona doc). The
  load-bearing inferences are task-clustered GEE / within-type permutation / seed-matched bootstrap.

---

## 5. Bottom line

The analysis-code layer is **trustworthy for the thesis's null result.** Every load-bearing claim
re-verified independently to the committed numbers: the homogeneous LR null (p≈0.999), the well-powered
LR (≥0.80 at a 0.15 gap; observed gap below the seed-matched identical-agent noise floor), the het
"interaction" as a MAIN-effect-only / p=0.056-task-ANOVA non-result, and the any@N gain as honest
oracle best-of-N coverage (+0.049 beyond 3B), all hold. The one real code defect — the false "Firth"
docstring plus the missing separation guard in `analyze_glmm.py` — is now fixed correctly, is
non-perturbing on converged fits, and demonstrably fires on a genuinely separated real run. The single
"robust interaction" the corpus had claimed (the rrpersona persona penalty) is confirmed to be
pseudo-replication on a deterministic round-robin assignment with a one-problem logic cell, and
collapses to NS under task-clustered GEE — which makes the null cleaner, not weaker. Residual issues
are cosmetic (a mislabeled comment, a benign NaN branch, an unused pre-reg JSON) and change no headline.
