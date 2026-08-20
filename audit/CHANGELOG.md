# Changelog — Systematic Results & Bibliography Review (2026-06-05)

Every change below is evidence-based. Verdicts/sources are in `audit/FINDINGS.md`.
Status legend: ✅ done · ⏳ pending multi-seed results (`results_multiseed/`).

---

## 1. Environment (new, isolated)
- ✅ Created `.venv` on the big disk (`/media/.../dissertacao/.venv`), Python 3.12, to avoid the 92%-full home partition.
- ✅ Installed: torch (CPU), numpy/scipy/pandas/statsmodels/scikit-learn/matplotlib/seaborn/sentence-transformers/datasets/pyyaml; **llama-cpp-python 0.3.26 (CUDA cu124)**; `nvidia-cuda-runtime-cu12` + `nvidia-cublas-cu12` (the wheel needs `libcudart.so.12`).
- ⚙️ **To run GPU code you MUST set** `LD_LIBRARY_PATH` to `.venv/lib/python3.12/site-packages/nvidia/{cuda_runtime,cublas,cuda_nvrtc}/lib` (handled by `scripts/run_multiseed.sh`).

## 2. Data-integrity verification (no code change)
- ✅ `audit/recompute_metrics.py`: independently recomputed S, Pass@1, χ², Cramér's V from every raw `task_log.jsonl`. **All 56 stored `final_metrics.json` reproduce exactly.** No fabrication / no metric bug.
- ✅ `audit/test_executor_sanity.py`: independent regression test for the HumanEval 96%-false-positive bug. Wrong solutions are rejected, correct accepted, exceptions caught → **bug fix confirmed**.

## 3. Source-code fixes (`src/`, `scripts/`)
- ✅ `src/models/llama_cpp.py`: added `seed` to `__init__`/`Llama(...)`; `generate()` now passes `top_p` and per-call `seed` to `create_completion`. **Generation is now reproducible** (verified: same seed → identical output). Previously generation was unseeded.
- ✅ `src/swarm/experiment.py`: `ExperimentConfig` gained `generation_temperature` (0.2) and `generation_top_p` (0.95); `run()` now passes temperature/top_p/seed explicitly. **Fixes the silent bug where `model.temperature: 0.7` in every YAML was ignored and generation actually ran at the hardcoded 0.2.** 0.2 is kept to stay comparable with all prior data; logged in config dump.
- ✅ `src/swarm/metrics.py`: `functional_differentiation` now drops zero-marginal rows/cols and guards `chi2_contingency` against the zero-expected-frequency crash. **No-op on valid 164-task tables** (verified: stored S=0.3896 / χ²=115.90 still reproduce).
- ✅ `src/swarm/humaneval.py`: loader now reads a **vendored local `data/HumanEval.jsonl`** (downloaded canonical 164 problems) first, HF as fallback. Fixes breakage from new `datasets`/`huggingface_hub`; improves reproducibility.
- ✅ `scripts/run_experiment.py`: reads `generation_temperature/top_p`; **fixed cumulative seed-dir nesting bug** (`config.output_dir` was mutated in place → `seed_42/seed_123/...`). Now seed dirs are flat siblings.
- ✅ New: `scripts/run_multiseed.sh` — **resumable** (skips completed seeds), **n=20 on the 5 KEY conditions** (exp2.1_experimental, exp_low_temp, exp_3b_baseline, exp_3b_low_temp, exp_7b_model) and **n=10 on the 3 REST** (exp_5_agents, exp_3b_model, exp_3b_5_agents) → `results_multiseed/`. 20 fixed seeds. `scripts/aggregate_multiseed.py` (mean±95% CI + per-seed permutation/LR significance counts; can be run incrementally on partial results).

## 4. Bibliography (every cited source verified — `audit/FINDINGS.md §D`)
- ✅ New `thesis/referencias.md`: full verified reference list (~45 works, arXiv IDs/venues/DOIs), Dochkina flagged ⚠.
- ✅ `thesis/cap2_revisao.md` corrections:
  - Qwen2.5-Coder 7B HumanEval **84.1% → 88.4%**; cite as **Hui et al., 2024**; Phi-3/Phi-4 → **Abdin 2024a/2024b**; Mixtral **56B → ~47B**; TinyLlama "3× larger" → "comparable size".
  - MetaGPT "top 1.2%" removed (unverifiable).
  - Guo et al. "four orchestration patterns" → reworded (own synthesis; survey's actual four aspects noted).
  - La Malfa "three problems" → **four areas** (social agency, environment, coordination/communication, measuring emergence); added arXiv + venue.
  - Yang contamination **~25% → 8–18%** (not GPT-4-specific).
  - EvalPlus GPT-4 HumanEval+ **79% → 76.2%**.
  - SWE-bench Verified: corrected (OpenAI Aug 2024; coexists with Lite, not "superseded"; 59.4% from a 2026 audit of 138 instances).
  - LiveCodeBench/BigCodeBench: years → 2024; LiveCodeBench adds AtCoder; added authors/arXiv.
  - Takata "ALIFE 2025" → preprint; Riedl "ICLR 2026 poster" → arXiv:2510.05174 (2025); MapCoder-Lite → benchmark specified (**xCodeEval**), authors added; Jimenez-Romero arXiv added; arXiv:2604.02621 reframed (judge-augmented, Shen et al.).
- ✅ `thesis/cap6_conclusao.md`: fixed **misattribution** — judge-vs-verifiable-rewards finding was wrongly cited as "Choi et al. (2024)"; corrected to **Shen et al. (2026, arXiv:2604.02621)** (Choi is the debate-martingale paper).

## 5. Methodology (`thesis/cap3_metodologia.md`)
- ✅ License fixed: **Apache-2.0 = 1.5B & 7B**; Qwen-Research = **3B only** (was backwards).
- ✅ §3.3.3 Functional differentiation rewritten to match what is actually computed (χ² of success counts *plus* the logistic-regression **LR test** as the primary test; design note on why the GLMM random intercept is unidentifiable here).
- ✅ Added generation parameters (temp 0.2, top-p 0.95) and **multi-seed protocol (5 seeds, seeded decoding)**; updated reproducibility (§3.5.3) and software table (Python 3.12, llama-cpp 0.3.x CUDA, scipy 1.17, sentence-transformers 5.x).
- ✅ Task categorization described accurately as **hybrid** (50 manual + 114 keyword-heuristic) with explicit caveats (LOGIC n=10; labeling ambiguity).

## 5b. Phase-2 experiments (added 2026-06-06 — diagnosis-driven)
**Diagnosis (audit/recompute + per-cell inspection):** across seeds, agents' *per-type success rates* are essentially equal (e.g. 7B math .70/.83/.78); apparent differences live in tiny-n cells (separation artifacts). So the frozen model's per-type competence is **agent-independent** → S/count-χ² (concentration) move, but the LR test (genuine differentiation) is null *everywhere* (incl. 7B 1/10 seeds). The single-seed 7B "genuine differentiation" (LR p=.008) was a lucky draw. **The null LR is a real result, not a measurement gap.**

New runs (resumable driver phase 2):
- ✅ Full **7B coverage** n=20: `exp_7b_baseline` (T=0.5), `exp_7b_low_temp` (T=0.1), `exp_7b_5_agents` (cheap, ~6 min/seed).
- ✅ **Mechanism A — type-filtered context** (`mech_{3b,1_5b}_typefilter`): large buffer (K=20) + retrieve an agent's OWN same-type examples as few-shot (show 5) + low router temp. Tests whether coupling accumulated experience to task type induces genuine per-type skill differences (LR). n=10 probe.
- ✅ **Mechanism B — persona positive control** (`mech_{3b,1_5b}_persona`): 4 agents, explicit type-specialist personas (Riedl's lever). Validity anchor — if even personas don't move LR, the null reflects task/model ceiling; if they do, the emergent null is a meaningful negative result. n=10 probe.

Code (all **config-gated, default behaviour preserved**, unit-tested in `audit`):
- `src/swarm/agent.py`: `build_prompt` gains `task_type`/`type_filtered`/`max_show`; `system_prompt_override` for personas.
- `src/swarm/experiment.py` + `scripts/run_experiment.py`: `context_retrieval` ("fifo"|"type_filtered"), `context_show`, `personas` config keys, wired + logged.
- `src/models/llama_cpp.py`: import guard broadened (ImportError/OSError/RuntimeError) so off-GPU dry-runs/tooling degrade gracefully.

## 5c. Code-generation coverage runs (added 2026-06-06) — to run SEQUENTIALLY after the sweep
Goal: "cover the bases" on code-gen capability with standardized, comparable runs alongside the swarm.
- **Agentic self-debug HumanEval** (`scripts/run_humaneval_agentic.py` + `.sh`): single model, iterate-with-test-feedback loop (≤K attempts; failed code+error fed back) — the HumanEval analog of Terminal-Bench's verifier loop. Reports **pass@1 (clean single-shot baseline, comparable to published)** and **solve@K-with-feedback** per size, by task type. All 3 sizes, n=5 seeds, seeded. Resumable. Directly informs RQ3 (does verifiable feedback help a *single* model, by scale). Logic unit-tested (no GPU).
- **Terminal-Bench floor probe** (decided: all 3 sizes, single-model `terminus`, core set, sequential): isolated `.venv-tb`; `scripts/tb_serve_model.sh` + `scripts/tb_run.sh`; `docs/terminal_bench_plan.md`. Oracle Docker smoke run for harness sanity. Expect single-digit/~0 (capability/headroom probe, not specialization).
- Both run AFTER the swarm sweep frees the GPU (user: sequentially). GPU note: 3B+1.5B co-reside (~5.6GB); 7B solo.

## 6. Pending (require the multi-seed results now running)
- ⏳ Aggregate `results_multiseed/` → mean ± 95% CI (S, D, Pass@1, χ², V) and per-seed permutation/LR significance counts.
- ⏳ Rewrite `cap4_resultados.md` to report multi-seed aggregates (replacing single-seed point values) and re-examine whether the **central "3B baseline fails / threshold between 3B–7B" claim replicates across seeds**.
- ⏳ Update `cap5`/`cap6` numbers + the "three zones" narrative and the chi²-vs-LR discussion to the aggregated evidence; update `cap1` H1–H4 status.
- ⏳ Move "multiple seeds" from future-work/limitations to completed; keep honest residual limitations.

## 7. Adversarial gap-closing — exploratory analyses (added 2026-06-08)
Triggered by the meta-reviewer's prioritized thinking-gaps. All NO-GPU; full write-up in
`audit/EXPLORATORY_ANALYSES.md` (incl. a ≥4-round self-critique log and a GPU-run triage).
- ✅ **GAP 5 (linchpin)** `scripts/validate_lr_power.py` → `audit/lr_power_validation.md`:
  LR power/calibration now a saved, self-consistent artifact (no hardcoded prose). Power
  curve re-expressed on the induced max−min per-type gap (a prior draft mislabeled the
  scale, inflating power); honest reading = ≥0.80 power only by a 0.15 gap, ~0.47 at the
  observed 0.10. Load-bearing evidence is a **seed-matched parametric bootstrap** +
  **one-sided non-superiority test** (mean excess over noise floor −0.17, p<0.0001): the
  observed differentiation does not exceed identical-agent noise → **true negative**, not
  low power.
- ✅ **GAP 3** `scripts/analyze_random_contrast.py` → `audit/random_contrast.md`: reports
  the previously-unreported RandomRouter contrast. Affinity concentrates S beyond the
  random floor (ΔS≈0.02, Holm p≤0.015, g≈1.1–1.4) → real but tiny *allocation*; the large
  S values are knob-induced, not learned affinity. Allocation emerged; competence
  differentiation did not.
- ✅ **GAP 2** `scripts/analyze_solution_diversity.py` → `audit/solution_diversity.md`:
  any@4−pass@1 = +6.3pp [CI +5.2,+7.2] but majority vote −1.7pp; solution diversity low
  (Jaccard 0.13); agent identity unrecoverable from style (acc 0.249 ≈ 0.250 chance,
  p=0.42). The any@N gain is pass@k-from-resampling identical agents, not differentiation.
- ✅ **GAP 8** verified `het_swarm` per-agent model wiring is **sound** (run_experiment.py
  loads distinct GGUFs per agent; dispatched by agent_id). Crash was `context_length:2048`,
  already patched to 4096. No wiring defect.
- ✅ **GAP 9 (new, on top of the reviewer)** `scripts/validate_het_interaction.py` →
  `audit/het_interaction_prereg.md`: pre-registration proving the het_swarm differentiation
  claim must rest on the **agent×type interaction**, not the raw per-type gap (large under
  pure competence+ceiling) or the agent main effect (trivially significant). Decomposition
  implemented in `scripts/analyze_glmm.py` (`agent_main_effect`). ens_3b confirms the
  homogeneous baseline (main p=0.99, interaction p=0.99).

## 8. Het result settled + persona correction + GAP-0 chapter rewrite (2026-06-09)

- ✅ **Het interaction FINAL = v3 null** (`audit/het_interaction_result.md`). Independent
  stats re-verifier reproduced all three sub-tests to the decimal: ANOVA F=2.573 p=0.0560,
  GEE cluster=`task_id` p=0.0822, drop-logic p=0.327. The het result is a **uniform
  competence MAIN effect** (10/10 seeds, 3B>1.5B everywhere) with **no** task-clustered
  type interaction. v1 ("0/10" per-seed vote-count) and v2 ("p≈6e-7") are both RETRACTED as
  pseudo-replication (shared 164 problems + shared 1.5B weights, within-task φ=0.69).
  Canonical-verdict banners added to `het_interaction_result.md`, `lr_power_validation.md`,
  `REVIEW2_SYNTHESIS.md`.
- ✅ **Persona-penalty RETRACTED** → `audit/persona_penalty_correction.md`. The
  `mech_7b_rrpersona` "own-type penalty" (−0.042, t=−5.45, p=0.0004, signs 0/10) was the
  SAME pseudo-replication: round-robin is deterministic (0/164 task_ids ever reassign across
  seeds), and the logic own-cell is a single repeated problem (HumanEval/95). Under
  task-clustered GEE (cluster=`task_id`, controlling type) the penalty is **NS** (coef
  −0.503, p=0.24; drop-logic p=0.38). The number lived only in `hostile_5d_final.py` output;
  now documented + pointer added to `FANOUT_SYNTHESIS.md`. Strengthens the null (even persona
  prompts do not induce a robust interaction).
- ✅ **GAP-0 thesis rewrite (cap1, cap3, cap4, cap5, cap6)** to the multi-seed source of
  truth (`audit/RESULTS_MULTISEED.md`). Replacements applied throughout:
  - **No capacity threshold:** all three baselines near-null (S: 1.5B 0.048, 3B 0.042, 7B
    0.037); the "spontaneous-7B" S=0.116 / χ²=15.60 was 1 seed in 13.
  - **S tracks router temperature, scale-invariantly:** low-temp S≈0.33–0.38 at all sizes
    (ΔS≈0.30); it is *allocation* (load concentration), not functional differentiation.
  - **LR functional differentiation null everywhere** (1–2/10–13 seeds = nominal FPR);
    well-powered (≥0.90 for a 0.15 gap) → true negative.
  - **Trade-off vanishes:** 1.5B low-temp ΔPass@1 −3.6pp (single seed) → **+0.4pp NS** (20
    seeds). "Three zones" / "3B accessibility frontier" removed.
  - **D demoted:** corr(D,Pass@1)=−0.05, corr(D,S)=+0.17 (395 seeds); D=0.836↔S=0.390
    coupling was a single-seed coincidence. The conjunctive criterion S>0∧D>0.3∧F dropped.
  - **Het + ensemble folded in:** uniform-competence main effect (10/10), no type
    interaction (NS); any@N ensemble gain **+4.9pp over best agent** (engineering, not
    coordination); persona NS.
  - cap1 H1/H4 marked REFUTED (verdicts updated, hypotheses kept as a-priori record);
    contributions reframed to the 4 survivors (well-powered null; allocation-vs-
    differentiation method; router-driven scale-invariant allocation; ensemble coverage).
  - cap3 seed count corrected (cinco → 10–20); gen-temp 0.2 + decode-seeding already present.
- ⏳ Pending: `scripts/run_xval_roles.py` (held-out generalization, ~42–58 GPU-h, queue after
  pipeline); biblio fixes into `cap2`.

## 9. Code-audit salvage + analyze_glmm separation guard (2026-06-09)

The code-audit meta-agent (a558795f68e987943) **died mid-run** (~95% complete; no final
synthesis emitted). Its 178-line transcript was salvaged. It independently RE-VERIFIED (sound):
the `final_metrics.json` partial-seed gate; het p=0.056; attack-2c (mbpp_3b_lowtemp pseudo-hit
collapses to p=0.71 after dropping a degenerate n=5 cell); attacks 3/3b (temporal "signal" is a
sparsity artifact that vanishes under the within-type permutation null); attack 4 (all alt
taxonomies NS, Holm p=0.448); any@N +4.9pp decomposition; the LR-power `synth()` `4·δ/3`
effect-size math + byte-identical re-run; the strict fence cleaner (regression-tested, all 4
cases, fixes the legacy prose-after-fence bug); `decouple_decode_seed` (distinct seeds
guaranteed for <131 agents); F-score zero-marginal drop (0 rows/cols dropped on all real runs).

- ✅ **Data note (correction to integrity agent):** of the "4 partial seeds", `mbpp_3b_baseline/2021`
  and `mbpp_3b_lowtemp/1011` are now COMPLETE (366 rows + final_metrics); only `ens_3b_n3/1819`
  (132 rows) and `rand_1.5b/123` (94 rows) are genuinely partial. The gate excludes the right ones.
- ✅ **FIXED `scripts/analyze_glmm.py` Firth overclaim + separation guard.** The docstring claimed
  "Firth-style penalized likelihood is used" under separation — there is NO Firth. On
  perfect/quasi-complete separation the ML fit silently returns a non-converged model with inflated
  log-likelihood → spuriously tiny LR p with no warning. Fix: (a) docstring corrected to state the
  truth (no Firth; separated cells handled by exclusion + permutation tests); (b) added a
  `mle_retvals['converged']` guard to BOTH `lr_test_logit` and `lr_main_effect_logit` → non-converged
  fits now return `{"error": "non-converged (separation)"}` instead of a bogus p. **Verified
  non-perturbing:** converged ens_3b fit unchanged (interaction p=0.9994, main p=0.8408); single-seed
  separated runs still error as before. Does not change any headline (homogeneous nulls converge;
  het carried by permutation/cluster-robust tests).
- ℹ️ **Disagreement noted, resolved in favor of stats re-verifier:** the code-auditor (block 23)
  accepted the per-seed persona-penalty t-test as valid; the stats re-verifier showed it is
  pseudo-replication (deterministic round-robin → same 164 problems → per-seed means are not
  independent draws). The stronger argument wins → `persona_penalty_correction.md` (GEE p=0.24 NS)
  stands.
- ℹ️ Minor (no fix needed): attack-5c mislabels het_swarm_ensemble as a "null control" (it has real
  model heterogeneity; its p=0.0003 interaction is expected scaling, not a null violation — the
  persona conclusion correctly rests on ens_3b_n3/ens_7b, p≈0.997); attack-1 k-means difficulty test
  is latently circular (author flags it as a fishing expedition); `has_loop` median-split can NaN on a
  binary (benign, Bonferroni-corrected exploratory).
- ✅ Verified `data/mbpp_difficulty_axis.json` (the check the agent died during): valid JSON, 366
  entries, all documented fields (prompt_chars/tokens, ast_nodes, cyclomatic, canon_lines,
  difficulty_z, difficulty_tercile); not yet referenced by any .py (pre-registration for xval_roles).

## 10. Replacement code-audit completed → `audit/CODE_AUDIT.md` (2026-06-09)

After the rate-limit reset, a fresh audit agent (af6c16600278eb2b5) re-ran the dead agent's audit
INDEPENDENTLY and wrote the synthesis the dead one never produced. All 9 salvaged findings
**independently CONFIRMED to the committed numbers** (het F=2.57 p=0.0560; 2c collapse 0.0037→0.7141;
temporal artifact reverses under permutation; taxonomy Holm 0.448; any@N +0.049 over best-3B; LR-power
4·δ/3 + byte-identical re-run; strict cleaner 4/4 cases; seed decoupling; metrics zero-marginal 0 drops).
- ✅ **analyze_glmm.py separation guard verified on LIVE separated data:** `exp2.1_experimental/seed_42`
  has full-model `converged=False` → the guard now returns "non-converged (separation)" instead of the
  bogus **p=0.7178** the old code emitted. Converged ens_3b unchanged (p=0.99937). `mle_retvals` is a
  dict with `converged` in statsmodels 0.14.6 → `.get(...,True)` safe. Other separated runs error earlier
  (Singular matrix, caught). Fix is correct, non-perturbing, and demonstrably effective.
- ✅ **Persona-penalty pseudo-replication CONFIRMED:** 0/164 task_ids reassign across 10 seeds
  (deterministic round-robin); logic-persona own cell = HumanEval/95 repeated 10×; per-seed p=0.0004 →
  task-clustered GEE p=0.239 (NS). `persona_penalty_correction.md` confirmed accurate.
- ℹ️ **Stale note corrected:** `ens_3b_n3/seed_1819` is now COMPLETE (492 rows + final_metrics) — it
  finished since the integrity audit; the "132-row partial" note is outdated. Only `rand_1.5b/seed_123`
  (94 rows) remains genuinely partial. Not headline-changing.
- ✅ New adversarial sweep (label alignment, permutation seeding, sampling caps): agent_id→persona map
  correct, all permutation tests deterministically seeded, **no silent sampling caps anywhere**.
- **Verdict: the analysis-code layer is trustworthy for the thesis's null result.** No headline-changing
  issue remains. Full detail in `audit/CODE_AUDIT.md`.

## 11. Adversarial review of the GAP-0 chapter rewrite + fixes (2026-06-09)

An adversarial meta-reviewer (a8f8c0bdd39950945) checked the rewritten cap1/3/4/5/6 against the audit
ledger (READ-ONLY). It independently re-verified every cap4 Tabela 1 value and recomputed every prose
delta (ΔS +0.297/+0.292/+0.347, ΔPass@1 +0.4/+3.0/+1.0 pp, scale +19/+7 pp, any@N − best +4.9 pp,
ΔS≈0.30) — all correct — and confirmed **no surviving live refuted claim** and no allocation/specialization
conflation. It found **2 BLOCKERs (both localized to cap4 §4.5.2)**, now FIXED after I verified them
against sources:
- **B1:** homogeneous any@N gain was written **+6.6 pp** → corrected to **+6.3 pp** (0.885−0.822=0.063;
  the +6.6 was a stale REVIEW2_SYNTHESIS figure; solution_diversity.md / EXPLORATORY_ANALYSES.md give +6.3).
- **B2:** the diversity numbers (any@N, −1.7 pp majority, Jaccard 0.13) were labeled **"3×3B"** but come
  from the **4-agent** `ens_3b` (n_agents=4), not the 3-agent control `ens_3b_n3` — the config→agent-count
  trap. Relabeled "4×3B, `ens_3b`".
- **M3 (cross-chapter):** cap1 objective said population "3 a 10 agentes" while cap4 reports "3 a 5" →
  cap1 harmonized to "3 a 5 nas condições centrais, com varredura de sensibilidade estendida".
- **M1:** the `+4.9 pp [4.3, 5.5]` CI was traceable only to the stats-re-verifier transcript → anchored
  in `CODE_AUDIT.md §2b` for committed provenance (number is a real seed-matched bootstrap CI, kept).
- **m1:** cap3 §3.3.3 "indica diferenciação funcional genuína" → "indicaria … (vs. mera concentração de
  roteamento) … o teste que sustenta o resultado nulo" (conditional mood; avoids echoing the retired phrase).
- ℹ️ Cross-doc note (no chapter change): `FANOUT_SYNTHESIS.md:26` reports opposite-signed D-correlations
  (r=−0.07/+0.13, n=130) vs the canonical `REVIEW2_SYNTHESIS.md:22` (−0.05/+0.17, n=395) the chapters cite.
  Chapters are faithful to the canonical doc; FANOUT line should be reconciled/retracted later.
- **Verdict: chapters are numerically faithful to the multi-seed ledger and internally consistent; the
  2 BLOCKERs were trivial and neither touched the central null.**

## 12. New confirmatory results + extras completion (2026-06-09 eve)

- ✅ **Label-free difficulty-axis interaction test EXECUTED** → `audit/difficulty_axis_result.md`,
  `audit/difficulty_axis_interaction.py`, `data/humaneval_difficulty_axis.json`. Delivers the robustness
  test cap5 §5.5.2 pre-registered but had never run (de-hostages the null from the noisy 4-way `logic`
  taxonomy). Built a HumanEval difficulty axis (same features as the MBPP axis; terciles done correctly
  55/54/55 — note the MBPP axis has a medium=0 tercile bug). GEE `success ~ C(agent)*difficulty_z`,
  cluster=task_id, on the ensemble runs: **agent×difficulty interaction NS in every run** (het 0.63,
  ens_3b_n3 0.23, ens_3b 0.091, ens_7b 0.95); het shows a competence MAIN effect (p=0.0013) only. The
  type-null is not a taxonomy artifact.
- ✅ **het_15_7b [1.5B,7B] COMPLETE (10/10 seeds)** → `audit/het_15_7b_result.md`; folded into cap4
  §4.5.1.1 as the heterogeneity dose-response rung. CUDA was fixed first (post-suspend `nvidia_uvm`
  reload — the run had silently fallen to CPU at 6.7 tok/s; after reload, full GPU offload). 10-seed:
  pass@1 0.701, S≈0 (ensemble), competence MAIN effect **+25.4 pp** (7B 0.827 vs 1.5B 0.574), and
  **agent×type interaction NS under task-clustering** (ANOVA F=2.15 p=0.096; GEE cluster=task_id p=0.129;
  drop-logic p=0.131). Per-type 7B−1.5B gap string +0.36 / math +0.26 / list +0.17 / logic +0.06 = a
  NON-significant gradient, same borderline-NS pattern as het_swarm_ensemble v3 (p=0.056). NB: the
  per-seed χ² (0/10) is the retired underpowered vote-count; the canonical task-clustered test is what is
  reported. "Scale buys uniform competence, not roles" holds at the maximum weight gap.
- ✅ **ens_1.5b [4×1.5B] COMPLETE (10/10)** — homogeneous 1.5B ensemble null topped up after the suspend.
- ℹ️ **rand_1.5b stays 9/10** — seed_123 DETERMINISTICALLY fails at ctx 4096 with a token overflow
  (`Requested tokens 4119 exceed context window 4096`) on one problem; reproduced on re-run (fixed decode
  seed → identical success pattern → same overflow at row ~94). Non-load-bearing RandomRouter control;
  n=9 is adequate. Completes only at larger context (8192); left at 4096/n=9 by user choice. Config
  unchanged (4096), so the other 9 seeds remain comparable.
- ⏳ Still open (per `audit/BACKLOG.md`): metrics+harness critique; RQ reframe (orphan contributions →
  add RQ4/RQ5; de-presuppose Objetivo Geral/RQ2); null-result defense memo; xval_roles (GPU); cap2 §5.5.2
  "pré-registramos"→"executamos" tense fix now that the difficulty axis ran.
