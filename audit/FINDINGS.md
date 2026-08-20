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
(every cited work: EXISTS / CLAIM-ACCURATE / verdict + correction)

### Batch A — SLMs / compression / models  (verified, with sources)
- **Kaplan et al. 2020** (arXiv:2001.08361) — CONFIRMED.
- **Phi-3 / Abdin et al. 2024** (arXiv:2404.14219) — CONFIRMED; but **Mixtral 8×7B ≈ 47B total params, NOT 56B** (≈12.9B active). Fix cap2.
- **Phi-4** = arXiv:2412.08905 (Dec 2024), distinct paper. **Disambiguate the two "Abdin et al., 2024" (Phi-3 vs Phi-4)** → 2024a/2024b. Soften "surpasses GPT-4 on STEM" → "exceeds GPT-4o on GPQA & MATH".
- **Qwen2.5-Coder = Hui et al. 2024, arXiv:2409.12186.** AUTHORITATIVE HumanEval Pass@1 (Instruct): **1.5B=70.7%, 3B=84.1%, 7B=88.4%**; HumanEval+: 1.5B=66.5, 3B=80.5, 7B=84.1.
  - ❌ "61.6%" for 1.5B is WRONG (that's the **0.5B** score).
  - ❌ cap2 "7B = 84.1% HumanEval" is WRONG → it is 7B **HumanEval+** (or 3B HumanEval). Correct 7B HumanEval = **88.4%**.
  - ⚠️ "84.8%" is NOT an official number — it is the thesis's OWN swarm Pass@1 (exp_7b_model). Legit in cap4 as a measurement, but must NOT be presented as the published benchmark.
  - ❌ **License backwards**: 7B = **Apache-2.0** (not Qwen Research); 3B = qwen-research; 1.5B = Apache-2.0. Fix cap3 §3.4.2.
- **TinyLlama Zhang et al. 2024** (arXiv:2401.02385) — exists; "comparable to models 3× larger" NOT the paper's claim (it beats *comparable-size* models) → reword/drop.
- **GPTQ Frantar et al. 2023** (arXiv:2210.17323, ICLR 2023) — CONFIRMED.
- **AWQ Lin et al. 2024** (arXiv:2306.00978) — CONFIRMED **Best Paper MLSys 2024**.
- **QLoRA Dettmers et al. 2023** (arXiv:2305.14314, NeurIPS 2023 Oral) — CONFIRMED.
- **Belcak & Heinrich 2025** (arXiv:2506.02153, NVIDIA) — CONFIRMED incl. 10–30× cost claim + LLM→SLM conversion algorithm.

### Batch B — MAS frameworks & debate  (verified, with sources)
- **MetaGPT Hong et al. 2024** (arXiv:2308.00352, ICLR 2024 Oral) — CONFIRMED; 85.9% HumanEval CONFIRMED; **"top 1.2%" UNVERIFIABLE → drop or source.**
- **AutoGen Wu et al. 2024** (arXiv:2308.08155, COLM 2024) — CONFIRMED.
- **ChatDev Qian et al. 2024** (arXiv:2307.07924, ACL 2024) — CONFIRMED (<7 min, <$1).
- **CAMEL Li et al. 2023** (arXiv:2303.17760, NeurIPS 2023) — CONFIRMED.
- **Guo et al. 2024 survey** (arXiv:2402.01680, IJCAI 2024) — venue CONFIRMED; ❌ **"four orchestration patterns" is a misgloss** — survey is structured around four *aspects* (environments, profiling, communication, capability acquisition). Reword.
- **Du et al. 2024 debate** (arXiv:2305.14325, ICML 2024) — CONFIRMED.
- **Zhang et al. 2025 "Stop Overvaluing Multi-Agent Debate"** (arXiv:2502.08788) — CONFIRMED; include full subtitle; thesis broader than self-consistency point.
- **Choi et al.** (arXiv:2508.17536, NeurIPS 2025 Spotlight) — CONFIRMED martingale + majority-voting; **correct title = "Debate or Vote: Which Yields Better Decisions in Multi-Agent Large Language Models?"**
- **La Malfa et al. 2025** (arXiv:2505.21298, NeurIPS 2025 Position) — exists; ❌ **FOUR areas, not three** (social agency, environment design, coordination/communication, measuring emergence). Fix cap2 §2.2.4.
- **Cemri et al. 2025 MAST** (arXiv:2503.13657, NeurIPS 2025 D&B) — CONFIRMED: 14 modes / 3 categories, 41–86.7% failure, Qwen2.5 included. (Correct title: "Why Do Multi-Agent LLM Systems Fail?")

### Batch C — emergence / ICL / threshold  (verified, with sources)
- **Bonabeau, Dorigo & Theraulaz 1999** (Swarm Intelligence, OUP, ISBN 978-0195131598) — CONFIRMED.
- **Casadei 2023** (Artificial Life 29(4):433–467; arXiv:2304.05147) — CONFIRMED; ❗ **single author — drop "et al."**
- **Rahman & Schranz 2025** (arXiv:2506.14496) — CONFIRMED incl. ~300× overhead (LLM Boids vs classical).
- **Jimenez-Romero et al. 2025** (arXiv:2503.03800, Frontiers in AI) — CONFIRMED; distinct from SWE-bench Jimenez. (Note cap2 §2.7 table lists "Romero et al. 2025" — should be **Jimenez-Romero**.)
- **Brown et al. 2020** (arXiv:2005.14165, NeurIPS 2020 Best Paper) — CONFIRMED.
- **Agarwal et al. 2024** (arXiv:2404.11018, NeurIPS 2024 Spotlight) — CONFIRMED (ICL overrides pretraining biases).
- **Takata, Masumori & Ikegami 2025** (arXiv:2509.04537) — author CORRECT (not Iwasaki); ❌ **"ALIFE 2025" venue unsupported → cite as preprint**; title "Emergent Social Dynamics of LLM Agents in the El Farol Bar Problem".
- **Riedl 2025** (arXiv:2510.05174, *Christoph Riedl*, "Emergent Coordination in Multi-Agent Language Models") — all claims CONFIRMED (4 tests via PID/TDMI; Plain condition; Llama-3.1-8B ~10%; stochastic drift; persona→stable roles); ⚠️ **"ICLR 2026 poster" UNVERIFIABLE → cite as arXiv** (and as 2025, not 2026).
- **Dochkina 2026** (arXiv:2603.28990, "Drop the Hierarchy and Roles…", MIPT, submitted to IEEE Access) — EXISTS; numbers (25k tasks, 8 models, +14%, +44%, d=1.86) all verified; ❗ **fabrication flag CONFIRMED — paper genuinely cites GPT-5.4 & Gemini-3-flash (non-existent)** → keep as cautioned/unreliable preprint.
- **MapCoder-Lite 2025** (arXiv:2509.17489, Lee/Cho/Choi) — CONFIRMED; ❗ 13.2%→28.3% is on **xCodeEval** (specify benchmark).

### Batch D — benchmarks & metric sources  (verified, with sources)
- **Chen et al. 2021 HumanEval** (arXiv:2107.03374) — CONFIRMED (164 problems, Pass@k origin).
- **Austin et al. 2021 MBPP** (arXiv:2108.07732) — ❗ "500 problems" conflates the ~500-problem **test split** with dataset size (974 full / 427 sanitized). Reword.
- **Yang et al. 2023 contamination** (arXiv:2311.04850, "Rethinking Benchmark and Contamination…") — ❌ **"~25%" WRONG → actual 8–18%**, and NOT specifically "GPT-4 training data" (GPT-4 corpus is non-public). Fix cap2 §2.5.1.
- **Liu et al. 2023 EvalPlus** (arXiv:2305.01210, NeurIPS 2023) — 80× tests CONFIRMED; ❌ **GPT-4 HumanEval+ = 76.2%, not 79%**; CodeLlama-34B 53%→45% approx/unconfirmed at precision.
- **Jimenez et al. 2024 SWE-bench** (arXiv:2310.06770, ICLR 2024 Oral) — CONFIRMED (2,294 issues, 12 repos).
- **SWE-bench Verified/Lite** — ❌ mischaracterized: Verified=500, by **OpenAI Aug 2024** (not 2025); does **NOT** supersede Lite (coexist); **59.4%** is from a **2026** audit of **138** hard-unresolved instances (not "all hard instances"). Fix cap2 §2.5.3.
- **LiveCodeBench** (arXiv:2403.07974, Jain et al. **2024**) — CONFIRMED; also includes **AtCoder** (not just LeetCode/Codeforces); year is 2024.
- **BigCodeBench** (arXiv:2406.15877, Zhuo et al. 2024 / ICLR 2025) — CONFIRMED.
- **Theil 1970** (Am. J. Sociology 76(1):103–154, DOI 10.1086/224909) — CONFIRMED (origin of uncertainty coefficient).
- **Blüthgen et al. 2006** (BMC Ecology **6:9**, DOI 10.1186/1472-6785-6-9) — CONFIRMED; ❗ locator is **6:9** (article 9), not "6(1)". Source of H2′/d′.
- **Friedman & Dieng Vendi Score** (arXiv:2210.02410, **TMLR 2023**; preprint 2022) — CONFIRMED.
- **Wang et al. 2020 ROMA** (arXiv:2003.08039, ICML 2020) — CONFIRMED.
- **arXiv:2604.02621** (Shen/Tu/Wang 2026, "RL-based KD with LLM-as-a-Judge") — EXISTS; numbers confirmed; ⚠️ reframe as **"judge-augmented vs verifiable-only"** (not head-to-head "judge beats verifiable").
- **arXiv:2510.07888** (Zhang et al. 2025, MARL network topology) — EXISTS; defines **SEI** (and IEI). CONFIRMED.

**Bibliography status: COMPLETE — all ~45 cited works checked. ~18 require correction (numbers/venues/author/locator); none of the load-bearing conceptual claims (Riedl threshold, Cemri failure rates, Belcak SLM economics, ICL mechanism) collapsed.**
