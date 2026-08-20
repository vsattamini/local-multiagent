# Adversarial-foil report (A + C) with self-review
**Date:** 2026-06-08 · Companion to `ADVERSARIAL_FOIL_REVIEW.md` (critique) and `EXPLORATORY_RESULTS.md` (registered no-GPU numbers).
**Scope:** consolidate the critique (A) + results section (C); suggest GPU runs (no quota — not executed); register exploratory results; then attack my own claims ≥3× and apply every correction.

---

## Part I — What changed on disk this pass
- **B (done):** `config/het_swarm.yaml` ctx 2048→4096 (the crash fix); **new** `config/het_swarm_ensemble.yaml` (the decisive computable-LR cell).
- **C (done, registered):** `audit/exploratory_analysis.py` → `audit/EXPLORATORY_RESULTS.md` (voting, disagreement, S-contrast, computability map; reproducible, no GPU).
- **A (revised):** two over-claims in `ADVERSARIAL_FOIL_REVIEW.md` corrected by the self-review below (the "tautology"/"power moot" line, and the "swarm beats single model" voting line).
- No thesis-chapter edits (deferred per instruction).

## Part II — The findings, as they stand AFTER self-review
> Blunt version: the thesis's old story is dead, the meta-reviewer's "refuted across 20 seeds" is also overstated, my own first pass had two wrong claims, and the one experiment that would settle it hasn't been run. Nobody in this loop has been rigorous enough. Details below.
1. **The computable LR-null (`ens_3b`) is *near-structural*, not informative-as-framed.** Identical weights + ensemble ⇒ agents ~symmetric ⇒ interaction ≈ 0. The only symmetry-breaking channel is type-filtered **self-conditioning** (each agent few-shots on its *own* past same-type successes; decode-decoupling makes those sets differ — 13.7% of tasks show between-agent disagreement, so buffers genuinely diverge). The null is therefore the *real but modest* finding: **self-conditioning does not bootstrap functional differentiation**. The corrected power artifact bounds that channel to δ<~0.15.
2. **The decisive experiment was never run: heterogeneous weights + ensemble** (`het_swarm_ensemble`). Only this cell has both a real competence basis AND a computable LR — *but* it tests an **interaction** (is the scale advantage *type-dependent*?), not a main effect (3B better overall is already partialled out). It can still null if scale helps all types uniformly. That is itself a publishable question about Qwen2.5-Coder scaling.
3. **Voting "win" is self-consistency, not swarm emergence.** any@N=0.885 vs pass@1=0.823 (+6.2pp) is the pass@1→pass@N lift of *one* 3B model sampled 4× with test-selection (agents share weights). Honest, useful (TDD setting), but not a multi-agent result. The genuine multi-agent voting test is `het_swarm_ensemble`.
4. **S is real concentration without consequence.** Affinity 0.042 vs random 0.019 (2.2× chance, matched n/temp); low-temp/pop → ~0.5. Reclaim as allocational division of labor; don't call it functional specialization. (Caveat: S degenerates at the monopoly boundary — greedy collapses to 1 agent yet S=0, so S cannot distinguish "uniform" from "monopoly"; report active-agent count alongside S.)
5. **The multi-seed "refutation" is itself thin — call it out.** The core configs are single-assignment, so the LR test is **non-computable in 35–95% of seeds** (`exp_3b_low_temp`: computable in 1/20). Where computable, significance runs 1/12–2/11 ≈ the false-positive rate. So "7B differentiation refuted across 20 seeds" really means "1 of 13 *computable* seeds significant." The honest verdict on the old 7B claim is **"not replicable AND not cleanly testable in this design,"** not "refuted." Only `ens_3b` (ensemble, 9/9 computable) can test it cleanly, and there it's homogeneous → near-structural null. ⇒ the meta-reviewer's GAP-0 "the data refutes all of it" overstates what a mostly-non-computable test can establish.
6. **The decisive experiment is *predicted to succeed* — and that's a trap to pre-empt (GAP 7).** From existing per-type rates, the 3B−1.5B advantage is strongly type-dependent (list +0.28, math +0.20, string +0.30, **logic −0.12**; range 0.41). So `het_swarm_ensemble` will *probably* yield a significant, genuine, computable LR — but it is **trivial differentiation (different weights have different skills), NOT emergence.** Pre-commit the framing NOW: "functional differentiation requires a competence basis (heterogeneous capability); it does not emerge among identical agents from in-context self-conditioning." (Caveat: the predicted signal leans on the n=10 logic cell and conflates size with run-condition — the actual run controls both.)
7. **Contributions:** only the LR-vs-χ² methodology survives intact; reframe around it + allocational specialization + the corrected scaling/voting questions. Democratization is strengthened by F1.

## Part III — Suggested runs → **see the "SESSION — Model runs" at the END of this doc** (necessary + recommended, with commands).

## Part IV — Adversarial self-review log (attacking my OWN claims)

### Iteration 1 — target: "the LR null is structural / tautological; power is moot"
- **Attack:** I claimed E[interaction]=0 "by construction." Is it *exactly* 0? type-filtered retrieval feeds each agent its own past same-type successes. In an ensemble all agents see all tasks, BUT decode-seed decoupling (`seed = random_seed + i*131 + agent_id`) makes them succeed on different subsets → their buffers differ → few-shot prompts differ across agents → a self-conditioning feedback loop that *could* bootstrap per-agent, per-type differences.
- **Verify:** disagreement rate 13.7% (EXPLORATORY_RESULTS §2) confirms agents' success sets — hence buffers — genuinely diverge. Memory is per-agent (task_log carries per-agent `agent_total_tasks`/`agent_success_rate`).
- **Verdict:** claim **too strong**. Null is *near*-structural; a weak symmetry-breaking channel exists. ⇒ power analysis is NOT moot — it bounds that channel (δ<~0.15).
- **Change applied:** rewrote TL;DR #1 and GAP-5 framing in the foil doc; Part II item 1 above.

### Iteration 2 — target: "any@N is an unreported swarm-beats-single-model positive result"
- **Attack:** the 4 agents in `ens_3b` share identical 3B weights. So "4 agents attempt every task" = "4 stochastic samples from one 3B model." any@N with test-selection = **pass@N of a single model** — a textbook result, not multi-agent emergence. Framing it as "swarm beats single model" is misleading.
- **Verify:** config `ens_3b.yaml` — homogeneous `model.path` 3B, no per-agent `models:` list. Only decode seed and (weakly) self-conditioned context differ. So agents ≈ i.i.d.-ish samples of one model. majority@N *hurts* (−1.5pp), exactly what you'd expect from near-duplicate samples (ties + no diversity). The +6.2pp is the pass@1→pass@4 lift.
- **Verdict:** claim **wrong as framed**. Downgrade: it's self-consistency, real but not novel; genuine swarm voting needs heterogeneous agents.
- **Change applied:** corrected TL;DR #3(b) in the foil doc; Part II item 3; redirected the voting result to `het_swarm_ensemble`.

### Iteration 3 — target: "het_swarm_ensemble LR can reject for a real reason (the decisive cell)"
- **Attack 1 (interaction vs main effect):** the LR null model is `C(task_type)+C(agent_id)`; the agent main effect (3B uniformly better) is *already partialled out*. The interaction term tests only whether the 3B's advantage *varies by type*. If scale helps all types ~equally, het_ensemble's LR is **still null** — heterogeneity is necessary but not sufficient.
- **Attack 2 (comparability):** I called `ens_3b` the "matched control," but ens_3b is n_agents=4 and het_ensemble is n=3. Not matched.
- **Verify:** confirmed against `analyze_glmm.py:77-78` (interaction LR) and the two configs (n=4 vs n=3).
- **Verdict:** claim **over-stated**. het_ensemble tests a *type-dependent* scale advantage, not "are agents different." And it needs an n-matched homogeneous control.
- **Change applied:** Part II item 2 reworded to "interaction = type-dependent advantage"; Part III adds the n=3 homogeneous control (`ens_3b_n3`) and reframes the het hypothesis. (Config header of `het_swarm_ensemble.yaml` notes the round_robin/router-moot point; will add the n-match caveat there too.)

### Iteration 4 — target: residual claims (S metric, power numbers, greedy)
- **Attack (S boundary):** greedy S=0.000 was reported as "collapses to 1 agent." But S=0 also means "perfectly uniform." S cannot tell monopoly from uniformity — a real metric limitation, not just a greedy quirk.
- **Verify:** greedy seed_0 → agent 0 did 164/164 tasks (monopoly), yet S=0. So S indeed conflates the two extremes when the active-agent count collapses.
- **Attack (power estimate noise):** my re-run power 0.557@δ=0.10 is one 300-sim estimate; 95% CI ≈ ±0.056. Doesn't change "powered for δ≥0.15," but the artifact should report the sim CI, not a bare point.
- **Verdict:** both **valid refinements**, neither overturns a headline.
- **Change applied:** Part II item 4 caveat (report active-agent count with S); flagged sim-CI for the power artifact rewrite (iter1 ask).

### Iteration 5 — target: my own deference to the multi-seed aggregate as "ground truth"
- **Attack:** I kept saying refutations "rest on the multi-seed aggregate, untouched." But the core multi-seed configs are single-assignment — can the LR test the aggregate leans on even *run*?
- **Verify (computed):** computable LR seeds — exp_3b_baseline 12/20, exp_7b_baseline 13/20, exp_7b_model 10/20, exp2.1 11/20, **exp_3b_low_temp 1/20**. Significant where computable: 1, 1, 1, 2, 0. That's ~5–18% ≈ the false-positive rate — consistent with null *and* with a small effect surviving only in lucky-computable seeds.
- **Verdict:** "refuted across 20 seeds" is **overstated** (mine and the meta-reviewer's). The single-seed 7B p=.008 is non-replicable, but the design can't cleanly test differentiation; the clean test (ens_3b) is homogeneous → near-structural. Downgrade "refuted" → "not replicable + untestable in this design."
- **Change applied:** Part II item 5; the runs session makes the clean test (het_ensemble + matched control) NECESSARY, not optional.

### Iteration 6 — target: "het_swarm_ensemble can reject for a real reason" (will it actually?)
- **Attack:** stop hand-waving — predict the outcome from data before burning GPU.
- **Verify (computed):** 3B−1.5B per-type gap = list +0.28, math +0.20, string +0.30, logic −0.12; **range 0.41** → strongly type-dependent → interaction likely significant. So het_ensemble probably "succeeds."
- **Foil-to-self:** the −0.12 logic gap rests on n=10 (the known-sparse cell), and the 1.5B vs 3B numbers come from *different run conditions* (rand_1.5b temp0.5 single vs ens_3b temp0.3 ensemble type_filtered) — size is confounded with condition. So "likely" is real but not clean; the actual within-run het experiment is what settles it.
- **Verdict:** prediction stands *with caveats*; the bigger point is the **framing trap** — a positive het result is trivial (weights differ), not emergent. Pre-commit the narrative.
- **Change applied:** Part II item 6; het config header already carries the interaction-vs-main-effect note.

### Net effect of self-review (6 iterations)
Materially downgraded: voting = self-consistency (not swarm); null = near-structural (not tautology); **"refuted across 20 seeds" = overstated** (mostly-non-computable test). Sharpened: decisive cell tests a *type-dependent* interaction, is *predicted to reject* from existing data, but a positive there is *trivial weights-driven* differentiation — pre-commit the framing. Caveats added: S monopoly boundary; power sim-CI; logic n=10 + cross-condition confound in the het prediction. **The thesis's old single-seed story is still dead** — but the correct epitaph is "not replicable + untestable in this design," not "refuted by 20 seeds." Surviving positives are smaller and more honest than my first pass: LR-vs-χ² methodology + allocational division-of-labor + a precisely-scoped, likely-positive-but-trivial het scaling result.

---

## SESSION — Model runs (necessary + recommended)
> NOT executed (no GPU quota). Env preamble for every run (from `scripts/run_multiseed.sh`):
> `export LD_LIBRARY_PATH=$PWD/.venv/lib/python3.12/site-packages/nvidia/{cuda_runtime,cublas,cuda_nvrtc}/lib`
> Then per config: `python scripts/run_experiment.py --config <yaml> --seeds 42 123 456 789 1011 1213 1415 1617 1819 2021`
> Standing convention: every report in this audit ends with this session.

### NECESSARY (the thesis cannot make a clean differentiation claim without these)
| # | Run | Config | Why it is necessary | Predicted result |
|---|-----|--------|--------------------|------------------|
| N1 | **Heterogeneous ensemble** | `config/het_swarm_ensemble.yaml` (n≥10) | The ONLY design with a real competence basis AND a computable LR interaction. Settles "can functional differentiation be detected at all." | **Likely SIGNIFICANT** (3B−1.5B per-type gap range 0.41, type-dependent) — but trivial/weights-driven, not emergent. Frame accordingly. |
| N2 | **n-matched homogeneous control** | `config/ens_3b_n3.yaml` (n≥10) | Without it, het-vs-homogeneous differs in BOTH weights and n_agents (4 vs 3). This isolates weights as the cause. | Near-structural null (matches ens_3b). |
| N3 | **Corrected power/equivalence artifact** | `scripts/validate_lr_power.py` (**NO GPU — run now**) | The current artifact has a HARDCODED verdict, claims a tighter bound (δ<0.10) than its own curve supports (powered only δ≥~0.15), and contains NO actual TOST despite claiming "TOST-style." It currently *weakens* the null. | Honest bound δ≥~0.15; real TOST/CI on the ens_3b interaction; report sim-CI. |

### RECOMMENDED (materially strengthen the story; not strictly blocking)
| # | Run | Config / action | Why |
|---|-----|-----------------|-----|
| R1 | **het_swarm re-run** | `config/het_swarm.yaml` (ctx now 4096) | Confirm the 2048 crash is gone; get the allocational-S story for mixed sizes (LR will be non-computable — document it). |
| R2 | **Break the HumanEval ceiling** | MBPP+ ensemble (`mbpp_*` loaders exist) or harder benchmark, ensemble + het | HumanEval is near-saturated (pass@1 0.77–0.85, only 13.7% between-agent disagreement) → almost no headroom for agents to differ. Differentiation may be untestable purely due to ceiling. A harder benchmark is the real test of whether the null is about the *mechanism* or the *benchmark*. |
| R3 | **Difficulty-stratified voting re-analysis** | `audit/exploratory_analysis.py` extension (**NO GPU**) | Does the any@N (self-consistency) gain concentrate on mid-difficulty tasks? Tightens the voting claim. |
| R4 | **Holm + exploratory/confirmatory labels** | re-analysis pass (**NO GPU**) | Garden-of-forking-paths defense: family-correct the multi-seed contrasts; label every post-hoc number (incl. everything in `EXPLORATORY_RESULTS.md`) as exploratory. |

### DO-NOT-RUN (explicitly out of scope — log so silence isn't read as coverage)
- 7B ensemble het (3B+7B+1.5B): exceeds 8GB VRAM (FANOUT note). - PID/transfer-entropy: needs synchronous multi-agent state this design lacks. - Greedy multi-seed beyond what exists: collapses to 1 agent (S degenerate); no new info.

---

## GAP 10 (foil, 2026-06-08 post-fanout) — the bottom line over-generalizes "true negative"
The fan-out's plumbing is sound (verified: het_swarm_ensemble/ens_3b_n3/ens_7b are in the DECISIVE list; single-assignment het_swarm dropped; `agent_main_effect` wired into analyze_glmm.py; power curve fixed — base 0.8, x-axis = induced 4δ/3 gap, seed-matched bootstrap, one-sided non-superiority excess −0.17 p<1e-4; style classifier at chance 0.246 vs 0.250). **But the summary claim "competence/style differentiation did not emerge, the null is a true negative" is scoped wrong:**

1. **Everything proven is proven for IDENTICAL agents only.** ens_3b interaction null, no stylistic signature, seed-matched bootstrap below noise floor — all homogeneous. The power artifact itself (now) scopes its verdict to "for meaningful differentiation" and "homogeneous-ensemble null only." The one-line summary drops that scope.
2. **The decisive test is not done — by the team's own pre-registration.** `het_interaction_prereg.md` states the differentiation claim rests ONLY on the het_swarm_ensemble agent×type interaction, which is queued, not run. Declaring "differentiation did not emerge" before it lands is the *mirror image* of the GAP-7 trap the prereg was written to avoid — pre-committing to the NULL.
3. **There is a live rejection signal.** Existing (fragile, n=10, cross-condition) per-type data shows a SIGN-FLIPPED logic gap (1.5B > 3B on logic). A sign flip cannot come from uniform competence + ceiling (the prereg's H0) — it is a genuine logit-scale interaction, exactly what het_swarm_ensemble's LR would detect. So het may well REJECT (trivially, weights-driven). The summary should not foreclose this.
4. **Even the homogeneous "true negative" is near-structural.** Identical weights ⇒ no competence basis ⇒ the meaningful effect is foreclosed by design. "True negative" over-dignifies a foregone absence; the honest phrase is "no differentiation is possible among identical agents, confirmed; the power analysis rules out underpowering as the explanation."

**Corrected bottom line:** *Among identical agents, neither competence nor style differentiation emerges (well-evidenced, but near-structural). Allocation emerges (S 2.2× chance at baseline, ~0.5 at low temp). Whether HETEROGENEOUS agents show functional differentiation is the open decisive test (het_swarm_ensemble vs ens_3b_n3); it is predicted to show a significant agent MAIN effect (trivial) and possibly a real type-dependent INTERACTION (the logic sign-flip) — report only the interaction as differentiation. GAP 0 (chapters cap4–6) remains the largest open task.*

**Change applied:** scoped the `validate_lr_power.py` header tagline to match its own verdict (was unscoped "true negative").
