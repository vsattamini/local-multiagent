# Audit & Revision Findings — Dissertation Results Review

**Started:** 2026-06-05
**Auditor:** systematic review (Claude)
**Scope:** verify result solidity, recompute metrics, re-run core experiments multi-seed, verify every bibliography source, revise chapters for consistency.

---

## A. Data-layer integrity — VERIFIED SOUND

| Check | Result |
|-------|--------|
| Stored `final_metrics.json` reproduce from raw `task_log.jsonl`? | ✅ **All 56 experiment dirs reproduce exactly** (S, Pass@1, χ², Cramér's V) via independent reimplementation (`audit/recompute_metrics.py`). |
| cap4 table numbers match stored `exp_*` metrics? | ✅ All 8 core rows match (S, p, Pass@1, χ², V). |
| HumanEval false-positive bug (96%) fixed? | ✅ Fixed in `executor.py`; task logs show real solutions + execution times; Pass@1 now in plausible ranges. (Independent wrong-solution smoke test pending env.) |

**Conclusion:** No fabrication, no metric-computation bug. The numbers in the thesis are faithful to the stored experimental data, which is faithful to the raw logs.

---

## B. Threats to result solidity — CONFIRMED ISSUES

### B1. Single seed per configuration (n=1)  [SEVERITY: HIGH]
Every experiment is a single run (`seed_42`). No variance, no CIs, no replication.
- Evidence of instability: 1.5B population sweep S is **non-monotonic** across n
  (n3=0.20, n4=0.54, n5=0.43, n6=0.48, n7=0.45, n8=0.54, n9=0.60, n10=0.62) —
  the bounce is run-to-run noise, not signal.
- **Action:** re-run the 8 core configs across ≥5 seeds; report mean ± 95% CI; re-test
  key contrasts (3B baseline vs 3B low-temp, etc.) with proper between-seed statistics.

### B2. LLM generation is stochastic AND unseeded  [SEVERITY: HIGH]
- `LlamaCppModel.__init__` creates `Llama(...)` with **no seed**; `create_completion`
  passes **no seed** (`src/models/llama_cpp.py`).
- `np.random.seed(42)` only controls the **router** (NumPy), not generation.
- ⇒ cap3 §3.5.3 "semente fixa (42) ... reprodutibilidade" is **overstated**: generation
  is not reproducible. Each single-seed result is one uncontrolled stochastic draw.
- **Action:** add seed support to the model; set llama seed = run seed so each seed is
  reproducible; correct cap3 §3.5.3.

### B3. Generation temperature in configs is IGNORED  [SEVERITY: MEDIUM]
- `experiment.py:167`: `self.model.generate(prompt, max_tokens=...)` — **temperature not passed**.
- ⇒ `generate()` uses hardcoded default **temperature=0.2** (`llama_cpp.py:50`), NOT the
  `temperature: 0.7` written in every config. `top_p: 0.95` is also ignored (llama default).
- Consistent across ALL runs ⇒ comparative results remain internally valid, but configs
  and any "0.7" claim are factually wrong.
- **Action:** correct configs to `0.2` (preserve comparability with existing data) and pass
  generation params explicitly; document the discovered discrepancy.

### B4. χ² functional-differentiation conflates routing with differentiation  [SEVERITY: MEDIUM]
- `metrics.functional_differentiation` runs χ² on the agent×type table of **success counts**.
  This is driven by how many tasks each agent was routed, not by per-category success-rate
  differences. The thesis already adds the **LR test** (`analyze_glmm.py`) which is the
  correct test; only **7B** shows genuine functional differentiation (LR χ²=15.6, p=.008).
- Tension: narrative elsewhere calls 3B low-temp S=0.39 "specialização funcional genuína",
  but the LR test there is non-computable (perfect separation) — i.e. it's concentration,
  not proven functional differentiation.
- **Action:** reconcile cap4 §4.3/§4.4 narrative; foreground LR test; downgrade "genuine
  functional differentiation" claims where only concentration is established.

### B5. Sparse `logic` category (n=10)  [SEVERITY: LOW-MEDIUM]
- Category counts: math=76, string=41, list=37, **logic=10**. Small cells make χ² unreliable
  and per-category success-rate estimates noisy.
- Manual categorization (`data/humaneval_categories_full.json`) — defensibility depends on
  the labeling rule (`scripts/categorize_tasks.py`). **Action:** document criteria; consider
  inter-rater check or sensitivity to re-categorization; caveat small-cell tests.

### B6. 0.5 affinity prior in qualitative reporting  [SEVERITY: LOW]
- `agent.success_rate` returns 0.5 for untried types. This is **only** a routing prior and a
  cosmetic value in `task_type_performance` — it does **not** enter S/D/F (those use task_log).
- But cap4 sometimes describes agents using these affinity numbers (e.g. "0.5 on logic" when
  the agent never attempted logic). **Action:** caveat or recompute qualitative profiles from
  actual attempts only.

---

## C. Outstanding workstreams
- [ ] Environment: venv on big disk + CUDA llama-cpp-python (IN PROGRESS)
- [ ] Code patch: seed support + explicit generation params (preserve temp=0.2)
- [ ] Independent wrong-solution smoke test (re-confirm bug fix)
- [ ] Multi-seed re-run of 8 core configs (≥5 seeds) + aggregation + CIs
- [ ] LR/GLMM re-run across seeds
- [ ] Bibliography: verify EVERY citation in cap1–cap6 against primary sources
- [ ] Chapter revisions (cap3 temp/seed/hardware; cap4 narrative reconciliation; cap5/cap6)
- [ ] Changelog of all edits

---

## D. Bibliography verification ledger
(populated during citation check — every cited work gets: EXISTS / CLAIM-ACCURATE / verdict)
