# Phase 2 Execution Plan: Statistical Volume & Scaling

**Status**: Phase 1 (Pilot) Complete - GO Verdict (S=0.193)
**Objective**: Execute "Layer 2" of the benchmark strategy to gather statistically significant data on emergence.

---

## 1. Immediate Technical Tasks

### 1.1 Task Categorization Expansion
The pilot used 50 manually categorized tasks. We need to cover the full datasets.
- [ ] **Categorize remaining HumanEval tasks** (114 tasks).
  - *Action*: Create `data/humaneval_categories_full.json`.
- [ ] **Integrate MBPP Dataset** (500 tasks).
  - *Action*: Implement `MBPPLoader` in `src/swarm/humaneval.py` (or `data_loader.py`).
  - *Action*: Create categorization mechanism for MBPP (keyword-based or LLM-assisted).

### 1.2 Experiment Infrastructure Upgrades
Upgrade `scripts/run_pilot.py` to a full `scripts/run_experiment.py` that supports:
- [ ] **Full Dataset Mode**: Switch to run all 164 (HumanEval) or 500 (MBPP) tasks.
- [ ] **Multiple Seeds**: Loop experiment `N` times with different random seeds.
- [ ] **Parameter Sweeps**: Accept config overrides for `n_agents`, `model`, etc., via CLI args.
- [ ] **Resumption**: Ability to resume an interrupted run (check existing logs).

### 1.3 Analysis Enhancements
Update `scripts/analyze_results.py` to handle:
- [ ] **Multi-run Aggregation**: Compute means and confidence intervals across seeds.
- [ ] **Significance Testing**: Implement T-tests/ANOVA to compare against baselines.
- [ ] **Comparative Plots**: Overlay S(t) curves for different population sizes (N=3 vs N=5).

---

## 2. Experimental Campaign (Weeks 3-6)

### Experiment 2.1: HumanEval Baseline & Scaling
**Goal**: Establish statistical significance of the pilot finding on the full benchmark.
* **Config**: Qwen2.5-1.5B, 164 Tasks.
* **Variations**:
    1.  **Baseline**: Single Agent (Standard generation).
    2.  **Control**: 3 Agents + Random Routing.
    3.  **Experimental**: 3 Agents + Affinity Routing (Pilot replication on full set).
    4.  **Scaling**: 5 Agents + Affinity Routing.

### Experiment 2.2: Population Size Effects (RQ3)
**Goal**: Determine optimal `N` and if "too many cooks" spoil the specialization.
* **Config**: Qwen2.5-1.5B, 164 Tasks (or MBPP).
* **Variations**:
    *   N = 3
    *   N = 5
    *   N = 7
    *   N = 10 (if VRAM allows efficient execution)

### Experiment 2.3: Model Capability (RQ1)
**Goal**: Test if 3B model differentiates faster/better than 1.5B.
* **Config**: N=3 Agents, 164 Tasks.
* **Variations**:
    *   Model = Qwen2.5-Coder-1.5B
    *   Model = Qwen2.5-Coder-3B (Requires ~4-6GB VRAM)

---

## 3. Execution Schedule

| Step  | Discovery | Implementation                           | Execution               | Analysis             |
| ----- | --------- | ---------------------------------------- | ----------------------- | -------------------- |
| **1** |           | **Task Expansion** (cat. full HumanEval) |                         |                      |
| **2** |           | **Script Upgrade** (run_experiment.py)   |                         |                      |
| **3** |           |                                          | **Exp 2.1** (N=3, Full) | Calculate P-values   |
| **4** |           | **MBPP Integration**                     |                         |                      |
| **5** |           |                                          | **Exp 2.2** (N=5, 7)    | Compare S(t) curves  |
| **6** |           |                                          | **Exp 2.3** (3B Model)  | Impact of capability |

---

## 4. Definition of Done (Layer 2)
1.  Full HumanEval (164 tasks) results for N=1, N=3(Random), N=3(Affinity), N=5.
2.  Statistical significance established (p < 0.05) for Specialization Index.
3.  Comparison of Pass@1 rates between Swarm and Baseline.
4.  Pareto frontier plot (Performance vs. Tokens).

---

## 5. Next Prompt Suggestion
> "Let's start Task 1.1. Please help me generate the categorization for the remaining 114 HumanEval tasks so we have the full `data/task_types.json` ready."
