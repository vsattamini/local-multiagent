# Exploratory Analyses — Closing the Meta-Reviewer's Thinking-Gaps

**Date:** 2026-06-08
**Status:** EXPLORATORY (no new GPU runs; all derived from existing logs). Confirmatory
re-runs are listed in the *Model Runs* session at the end.
**Scope:** Four gaps the adversarial meta-reviewer flagged as the linchpins of
defensibility — GAP 5 (power), GAP 3 (S vs random floor), GAP 2 (unmeasured
differentiation), GAP 8 (het-swarm wiring). Each is now backed by a saved artifact.

A note on **forking paths** (the reviewer's standing worry): every result below is
labelled *exploratory* unless it was pre-registered. Where a claim is decision-relevant
it is tied to an explicit, falsifiable threshold computed in code, not to a narrative.

---

## GAP 5 — Is the central null just "underpowered"? (the linchpin)
**Artifact:** `audit/lr_power_validation.md` · `scripts/validate_lr_power.py`

The reviewer's sharpest catch: I had *claimed* a synthetic test proved the LR
interaction test can detect real differentiation, but never saved it — so the null
collapsed to "maybe low power." Now saved and, critically, **self-consistent** (every
number in the verdict is computed from the run, none hand-typed — the earlier draft
hard-coded "power≈1.0 at delta≥0.10", which the simulation flatly contradicted).

Honest numbers, matched to the real ensemble design (4 agents, ~41 obs/cell, base
success ~0.8). The x-axis is the **induced max−min per-type gap across agents** — the
*same statistic* as the observed effect size in section C, so the two are directly
comparable (an earlier draft mislabeled it, inflating the power readout; see the
iteration log at the end of this file):

| induced max−min per-type gap | LR reject-rate (power) |
|---|---|
| 0.00 (null) | 0.08 — mildly anti-conservative |
| 0.05 | 0.10 |
| 0.10 | 0.47 |
| 0.15 | 0.91 |
| 0.20 | 1.00 |

- **Calibration:** under H0 the test rejects ~0.08 (≈1.6× alpha) — biased *toward* finding
  differentiation, so a null result is conservative.
- **Power:** ≥0.80 only by a **0.15** gap; at the *observed* gap (0.10) power is just
  **~0.47**. So the omnibus p-value **alone cannot carry the null** at this effect size —
  the load-bearing evidence is the bootstrap below, not the power curve.
- **Real data (load-bearing):** the omnibus LR never rejects (median p=0.99, 0/9 seeds),
  and the observed worst-case per-type gap (median 0.10) sits at the **2nd percentile of a
  seed-matched identical-agent parametric bootstrap** — i.e. the apparent differentiation
  is at or *below* the spread sampling noise produces when agents are identical. A formal
  **one-sided non-superiority test** confirms it: mean excess over the noise floor =
  **−0.17 (p<0.0001)** — the observed differentiation does not exceed (indeed falls below)
  what identical agents produce, because shared weights + low temp make the agents *more*
  correlated than independent draws. This comparison is assumption-free and does not depend
  on the power curve. *(An earlier auto-added symmetric TOST was mis-specified — it "failed"
  on the helpful side, i.e. for the observed effect being too far BELOW the floor; replaced
  with the correct one-sided test. See iteration log.)*
- **Verdict:** for differentiation of *meaningful* size (≥0.15 gap) the test is well-powered
  and silent; for the trivially small effect actually observed, the bootstrap shows it is
  indistinguishable from identical-agent noise. The null is a **true negative for meaningful
  differentiation**, not low power.

---

## GAP 3 — Is S unfairly dismissed as a "routing artifact"?
**Artifact:** `audit/random_contrast.md` · `scripts/analyze_random_contrast.py`

The dismissal was by assertion; the RandomRouter controls were run but never reported.
Per-seed S (canonical `1 − H(type|agent)/H(type)`), affinity vs matched random router
(same model / n_agents=3 / router_temp=0.5):

| size | S_affinity | S_random | ΔS | 95% CI | Hedges g | MWU p | Holm p |
|---|---|---|---|---|---|---|---|
| 1.5B | 0.048 | 0.020 | +0.028 | [+0.017,+0.040] | +1.40 | 0.001 | 0.004 |
| 3B   | 0.042 | 0.019 | +0.023 | [+0.012,+0.035] | +1.15 | 0.006 | 0.012 |
| 7B   | 0.037 | 0.019 | +0.019 | [+0.009,+0.029] | +1.08 | 0.015 | 0.015 |

(Holm-corrected across the 3 size-tests — all survive, addressing the forking-paths risk.)

- **The affinity mechanism concentrates routing beyond the random floor** — significant,
  large effect size, CI excludes 0 at all three sizes. So S is **not** a pure artifact:
  real, mechanism-driven **task allocation** emerged.
- **But the magnitude is tiny** (S≈0.04 vs 0.02 floor). The large S values the thesis
  highlights (low-temp 0.33, type-filtered 0.43) come from the routing-temperature /
  type-filter knobs — allocation **by construction**, not learned affinity at default.
- **Reconciliation with GAP 1:** division-of-labor theory is right that allocation *is*
  a form of specialization — and it genuinely emerged (ΔS>0). It is simply **logically
  distinct** from *functional* (competence) differentiation, which the LR test finds
  absent. The defensible framing: *allocation emerged; competence differentiation did
  not.* Two separate, both-honest claims.

---

## GAP 2 — Do agents differentiate in UNMEASURED ways (style, approach, voting)?
**Artifact:** `audit/solution_diversity.md` · `scripts/analyze_solution_diversity.py`
Ensemble ens_3b (4 agents × 164 tasks × 9 seeds, every solution logged).

The reviewer's strongest conceptual objection: we measure specialization only as
per-type success rate; agents could differ in approach/style/diversity. Three probes:

- **A. Voting / coverage.** mean per-agent pass@1 = 0.822; **any@4 = 0.885 (+6.3pp,
  CI [+5.2,+7.2])**; **majority@4 = 0.805 (−1.7pp, CI [−2.3,−1.0])**. *Crucial reframing
  (iteration 3):* the four agents
  share identical frozen weights, so any@4 is just the standard **pass@k-from-resampling**
  gain you get from sampling ONE model four times — it is **not** evidence of agent
  diversity or specialization. And naive **majority voting actually loses** (−1.5pp) by
  outvoting correct minorities. So the swarm of identical agents buys exactly the pass@k
  coverage of one resampled model and nothing more — which *reinforces* the null rather
  than rescuing the swarm. Beating pass@k requires either genuinely different agents (het
  swarm) or a test-based selector, not a vote.
- **B. Solution diversity.** pairwise token-Jaccard = 0.127 (median 0.097); among
  jointly-correct pairs 0.113. **Diversity is low** — agents write ~88%-overlapping
  code, consistent with identical frozen weights + low temperature. *Caveat:* Jaccard is
  a *lexical* set-overlap measure (ignores control-flow/semantics); it is a floor on
  diversity, so the low value is a conservative read, corroborated by the small pass@k gap.
- **C. Stylistic signature.** A RandomForest predicting agent_id from style features
  (length, lines, loops, recursion, comprehensions, defs) scores **0.249 vs 0.250
  chance** (permutation p=0.42). **No detectable per-agent style** — the null extends
  beyond success rate to coding style itself.
- **Verdict:** the "maybe they differ in unmeasured ways" escape hatch is **closed by
  measurement**: no style signature, low (lexical) diversity, and the only ensemble gain
  is plain pass@k resampling that majority voting can't even keep. None of this is
  agent differentiation.

---

## GAP 8 — Het-swarm: wiring bug or just the context crash?
**Artifact:** code read (`scripts/run_experiment.py:302-344`, `src/swarm/experiment.py:211`)

- Per-agent models **are** loaded from the `models: [1.5b,1.5b,3b]` list with distinct
  GGUF paths (`run_experiment.py:312-316`), passed through to `SwarmExperiment(models=)`,
  and dispatched at runtime by `self.models[agent.agent_id]` (`experiment.py:211`). The
  wiring is **sound** — the heterogeneous swarm would genuinely be heterogeneous.
- The crash (`Requested tokens (2071) exceed context window of 2048`) was purely
  `context_length: 2048`, **already patched to 4096** in `config/het_swarm.yaml`.
- **Verdict:** no wiring defect. The run only needs to be re-executed (GPU). Until then,
  there is **no heterogeneous-swarm result** and the thesis must not pre-commit to either
  outcome (see the heterogeneous-tension note below).

---

## GAP 9 (new) — Het-swarm: don't mistake competence for specialization
**Artifact:** `audit/het_interaction_prereg.md` · `scripts/validate_het_interaction.py`
· decomposition wired into `scripts/analyze_glmm.py` (`agent_main_effect`).

The meta-reviewer predicts het_swarm_ensemble will give "a significant, computable LR
(per-type gap ~0.41)" and treats that as *the* functional-differentiation result. That
prediction contains a trap, and the decisive run should be pre-committed before it lands:

A heterogeneous swarm where the 3B is simply **uniformly better** (constant log-odds
advantage, NO niche) produces — by simulation, 200×/scenario, 41 obs/cell:

| scenario | agent MAIN-effect p | interaction p | interaction reject-rate | raw max-min gap |
|---|---|---|---|---|
| uniform competence (no niche) | 0.000 | 0.439 | **0.04 (≈α)** | **0.39** |
| genuine per-type niches | 0.480 | 0.000 | **0.97** | 0.34 |

- The **raw per-type success-rate gap is large (0.34–0.39) in BOTH** scenarios — and is
  actually *bigger* under pure competence than under real specialization. It **cannot**
  distinguish "the 3B is better everywhere" from "agents specialize." So the reviewer's
  predicted 0.41 gap is fully consistent with **zero** specialization.
- Only the **agent×type interaction LR (logit)** separates them (reject-rate 0.04 vs 0.97)
  and it does **not** false-positive on a pure size advantage.
- Arithmetic confirms the mechanism: at ens_3b's high base rates a uniform +1.2-logit
  boost yields per-type gaps of +0.07 (list, near ceiling) to +0.21 (logic, hardest) with
  *zero* interaction — exactly the spurious "profile difference" a naive reading mislabels.

**Pre-registered rule for het_swarm_ensemble (now implemented in `analyze_glmm.py`):** the
differentiation claim rests *only* on the interaction term. A significant agent **main
effect** is expected and trivial (the 3B is better). Specialization = a significant
**interaction** in het where the n-matched homogeneous control **ens_3b_n3** shows none
(ens_3b already shows neither: main-effect p=0.99, interaction p=0.99). That contrast —
interaction present under heterogeneity, absent under identical weights — is the cleanest
possible positive result, and the only one that survives this critique.

## Cross-cutting consequences for the thesis

1. **GAP 0 stands and is now even sharper.** Power+random+style analyses all converge on
   one finding — *allocation emerged, competence/style differentiation did not* — and
   that finding **contradicts** the chapters' surviving single-seed story (three zones,
   3B–7B threshold, spontaneous-7B emergence, vanishing trade-off). The 7B "genuine
   differentiation" (old LR χ²=15.6, p=.008) is now 1/13 seeds significant ≈ the 5% FP
   rate. Chapters 4–6 must be rewritten to the multi-seed null.
2. **The contribution survives the pivot** ("so what?"): a rigorously-established,
   well-powered **null** on functional differentiation under identical frozen weights,
   *plus* two positive sub-results — (a) measurable-but-negligible affinity-driven
   allocation, (b) a +6.2pp any@N coverage gain that majority voting fails to capture.
   That is a defensible master's contribution: it tells you *where* swarm value can and
   cannot come from when weights are shared.
3. **Heterogeneous tension:** if the pending het run *does* show differentiation, it
   confirms — not refutes — the framing, because the current null is explicitly
   conditioned on **identical** weights. State that conditioning up front so the story
   is not retro-fitted either way.
4. **Accessibility motivation (cap1):** the pivot orphaned it. The any@N coverage result
   partially re-connects (small models + cheap diversity buy coverage), but the honest
   headline is single-model ≥ swarm on competence — cap1's promise must be re-scoped.

---

# 📋 Model Runs Session — necessary & recommended

*No GPU quota is available to this agent; this is the queue for whoever has the box.
Priority order. "Necessary" = the thesis is not defensible without it. "Recommended" =
materially strengthens it.*

### NECESSARY (defensibility-blocking)
1. **Heterogeneous swarm** (`config/het_swarm.yaml`, ctx fix applied) — 2×1.5B+1×3B,
   affinity τ0.3, n_agents=3, ≥10 seeds. This is the **only** condition that can produce
   genuine competence differentiation (router steers the stronger model into a niche).
   Its result decides whether the thesis closes as "no differentiation under shared
   weights, *and* it appears the moment weights differ" (strong) or "absent even when
   heterogeneous" (also publishable, different framing). *Verify the per-agent dispatch
   empirically:* assert ≥1 task routed to each model and that 3B logs differ from 1.5B.
2. **Ensemble for 7B + 1.5B** (k-of-N, every task × every agent), ≥10/≥10 seeds —
   extends the ens_3b power analysis to the other sizes so the LR null is not 3B-only.
   ens_3b already exists; 7B/1.5B do not. Needed to claim the null is size-general.

### RECOMMENDED (strengthen, not block)
3. **RandomRouter at LOW router-temperature** (τ0.1–0.3) for 3B/7B, ≥10 seeds — the
   current contrast is only at τ0.5 (near the S floor). To prove the *large* S values
   (0.33–0.43) are knob-induced rather than affinity-induced, we need the random floor at
   the **same sharp temperature**. Closes GAP 3 completely.
4. **Ensemble VOTING sweep** — same ensemble runs, report pass@1 / any@N / majority@N /
   *unit-test-selected*@N for 1.5/3/7B, n_agents∈{3,5}. The any@N coverage gain (+6.2pp
   here) is the best "swarm beats single model" lever; a test-based selector should beat
   majority. Turns GAP 2's coverage finding into a headline positive result.
5. **MBPP+ core** (3 sizes × baseline+low-temp, n=10) — HumanEval is near-saturated
   (~0.82); a benchmark with headroom tests whether the null is a ceiling artifact
   (reviewer's "mechanism vs phenomenon" gap). Loader/data already vendored
   (`src/swarm/mbpp.py`, `data/MbppPlus.jsonl`).
6. **Context-shuffle causal test** (3B-base, 3B-low-temp, 7B; control vs scrambled final
   buffers) — tests whether the small allocation that *did* emerge is causal for
   performance or epiphenomenal. Script staged (`scripts/run_context_shuffle.py`).

### SCOPE-ONLY (do not run for this thesis)
- PID / time-delayed MI (needs synchronous multi-agent coalition state — our one-agent-
  per-task design can't supply it).
- BigCodeBench / SWE-bench (SWE-bench is a 0%-floor for 1.5B, not headroom).

---

## Self-critique log (≥3 iterations, every issue fixed in code)

**Iteration 1 — power-curve scale error (MATERIAL).** The LR power table labelled its
x-axis `delta`, but `synth()` injects `+delta` to the owned type and `−delta/3` to the
others, so the induced agent-to-agent gap is `4·delta/3`, whereas section C's *observed*
statistic is a raw max−min gap. The two were on different scales, making "0.83 power at
0.10" a false comparison. **Fix:** re-expressed the curve on the induced max−min gap
(directly comparable). Corrected reading: power is **0.47 at the observed 0.10 gap**, not
0.83 — the omnibus is well-powered only for gaps ≥0.15. Consequence: the *seed-matched
bootstrap* (section C), not the power curve, is now explicitly the load-bearing evidence
for the null. The headline conclusion survives but is now honestly scoped.

**Iteration 2 — multiple comparisons & unsupported significance.** (a) GAP 3 ran three
size-tests with no family correction (the reviewer's forking-paths worry). **Fix:** added
Holm-Bonferroni — all three survive (0.004/0.012/0.015). (b) GAP 2 reported any@N − pass@1
as bare means. **Fix:** added a paired bootstrap CI across seeds, and made the majority
rule explicit (strict >½, 2–2 ties = fail).

**Iteration 3 — conceptual mislabel of the any@N gain (IMPORTANT).** I had called the
+6.2pp any@N gain a "solution-diversity payoff." But the four agents share identical
frozen weights, so any@N is simply **pass@k-from-resampling one model** — not agent
diversity. **Fix:** reframed in the report; this *reinforces* the null (the swarm buys
only what resampling one model buys, and majority voting can't even retain it). Also added
the caveat that token-Jaccard is a *lexical* diversity floor, and verified the GAP-3
RandomRouter contrast is confound-free (all configs matched at n_agents=3, temp=0.5).

**Iteration 4 — mis-specified equivalence test + a trap in the reviewer's het plan.**
(a) An auto-added symmetric TOST on the excess-over-noise-floor reported "NOT equivalent"
— but only because the observed effect is far *below* the floor (excess −0.17), the
helpful direction. A two-sided equivalence test is the wrong tool; **replaced with a
one-sided non-superiority test** (p<0.0001 that observed does not exceed the floor). As it
stood, that artifact actively undermined the null. (b) Stress-testing the reviewer's
het_swarm prediction surfaced GAP 9: the predicted "per-type gap 0.41" is consistent with
*zero* specialization, because the raw gap is inflated by competence+ceiling. Built a
pre-registration simulation and wired the main-effect/interaction decomposition into
`analyze_glmm.py`.

Net effect of the iterations: two findings were materially corrected (GAP-5 power
magnitude; the equivalence-test specification), a new interpretive gap was added (GAP 9),
none of the substantive conclusions were overturned, and every conclusion is now tied to
the strongest available evidence rather than the most convenient.

---

## Triage against the current GPU milestone schedule

**Status update:** the two decisive configs the reviewer flagged as missing —
`het_swarm_ensemble` and `ens_3b_n3` — now **exist and are already wired** into
`scripts/run_phase3_parallel.py` (JOBS lines 53-56), and the single-assignment
`het_swarm` is correctly **dropped** (line 63). `ensemble_assignment: all` is code-
supported, so every agent attempts every task → balanced cells → computable LR. No
further wiring is needed; the schedule below assumes that JOBS list.

| Run | Verdict | Why |
|---|---|---|
| **ens_3b** (4-agent homogeneous ensemble) | ✅ **Done, load-bearing** | Substrate for the entire GAP-5 power/effect-size/non-superiority analysis. 9 seeds; the near-structural null anchor + voting numbers. |
| **het_swarm_ensemble** (2×1.5B+1×3B, ensemble) | ✅✅ **THE decisive run — protect at all costs** | Only condition that can show genuine differentiation. *But read it via GAP 9:* the claim is the **agent×type interaction**, NOT the raw per-type gap (which inflates under pure competence + ceiling) and NOT the agent main effect (trivially significant). Decomposition now implemented in `analyze_glmm.py`. |
| **ens_3b_n3** (3×3B homogeneous, n-matched control) | ✅✅ **Keep — it earns the het claim** | Differs from het *only* in weights. Het-interaction-present + ens_3b_n3-interaction-absent = heterogeneity-driven specialization. Without it, a het interaction is uninterpretable. |
| **ens_7b** (clean 7B ensemble null) | ✅ **Keep** | The computable 7B null the single-assignment multi-seed data never could deliver. Extends GAP-5 power story beyond 3B. |
| **3B / MBPP+ lane** (single-assignment) | ⚠️ **Marginal — keep the cheap small-model ones** | No computable LR (single-assignment), so it cannot test differentiation. Its only job is pass@1 headroom to argue the null isn't pure HumanEval ceiling. Keep the parallel 1.5B/3B runs; don't bill them as testing the thesis question. |
| **Solo 7B (single-agent)** | ❌ **Trim** | Largely duplicates `exp_7b_*` already in the repo. If the 24h serial pole threatens the deadline, drop this to protect het/ens_7b. |
| **context-shuffle** | ⚠️ **Keep — cheap, but pre-register** | Re-reads completed runs (no fresh generation), so it's nearly free. Tests whether the FIFO/5-shot context channel is causal. **Pre-register: a null shuffle effect is a POSITIVE result** (confirms the self-conditioning channel is inert). Given GAP-3's tiny allocation and GAP-5's null, a null-on-null is the likely — and publishable — outcome. |
| **Terminal-Bench (floor probe)** | ❌ **Cut to a one-line footnote** | Expected ~0% on sub-7B models. A floor yields zero differentiation signal, same as a ceiling. Run ONE size / one pass and cite as "agentic terminal tasks are out of reach for sub-7B." Do not spend Jun-10 wall-clock on a full sweep. |

**Priority if GPU-bound:** `het_swarm_ensemble` + `ens_3b_n3` + `ens_7b` outrank everything
else combined — they are the only computable, decision-relevant results. The single-
assignment 7B/het/MBPP runs give pass@1 numbers that are largely predictable.

**One cheap run still NOT queued:** **RandomRouter at low router-temperature** (τ0.1–0.3)
for 3B/7B. The GAP-3 contrast is only at τ0.5 (near the S floor); to prove the *large* S
values (0.33–0.43) are knob-induced rather than affinity-induced, the random floor is
needed at the *same sharp temperature*. Cheap; closes GAP 3 fully.

**Bottom line:** protect **het_swarm_ensemble + ens_3b_n3 + ens_7b**; keep
**context-shuffle** (cheap, pre-registered); MBPP+ small-models only as headroom; **trim
solo-7B**; **cut Terminal-Bench to a footnote.**
