# Phase 3 Execution Plan: Ecological Validation

**Status**: Phase 2 Infrastructure Ready
**Prerequisite**: Complete Phase 2 experiments (statistical volume)
**Objective**: Test if emergent specialization transfers to realistic software engineering tasks

---

## 1. Overview

Phase 3 validates the practical relevance of emergence findings by testing on SWE-bench Lite—real GitHub issues from popular open-source repositories.

```
┌─────────────────────────────────────────────────────────────────┐
│                    LAYER 3: ECOLOGICAL VALIDATION               │
│                         SWE-bench Lite                          │
│                    (30-50 selected issues)                      │
│                                                                 │
│   Purpose: Test transfer to realistic tasks                     │
│   Primary metric: Comparative resolve rate                      │
│   Question: "Does specialization help with real problems?"      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Setup Tasks

### 2.1 Environment Setup
- [ ] Install SWE-bench harness: `pip install swe-bench`
- [ ] Install Docker (required for execution environment)
- [ ] Download and configure SWE-bench Lite dataset
- [ ] Verify execution harness works with single test issue

### 2.2 Issue Curation
Select 30-50 issues based on:

| Criterion            | Value      |
| -------------------- | ---------- |
| Estimated human time | 15min - 1h |
| Files modified       | ≤ 3        |
| Lines modified       | ≤ 100      |
| Clear tests          | Yes        |
| Complete description | Yes        |

**Repositories to include**: django, requests, flask, matplotlib, scikit-learn

### 2.3 Categorization
Categorize issues by:
- **Type**: bug, feature, refactor
- **Difficulty**: easy, medium, hard
- **Repository**: for domain specialization analysis

---

## 3. Experiment Design

### Experiment 3.1: Transfer Test

**Goal**: Test if Phase 2 emergence transfers to realistic tasks

| Condition    | Description                                            |
| ------------ | ------------------------------------------------------ |
| Baseline     | Single agent (same token budget as swarm)              |
| Control      | 3 agents + random routing                              |
| Experimental | 3 agents + affinity routing (pre-trained on HumanEval) |

**Metrics**:
- Resolve rate (absolute)
- Relative improvement vs baseline
- Specialization index on SWE-bench categories

### Experiment 3.2: Domain Specialization

**Goal**: Test if agents specialize by repository/domain

| Analysis              | Method                                     |
| --------------------- | ------------------------------------------ |
| Repository affinity   | Measure if agents prefer certain repos     |
| Bug vs feature        | Compare performance by issue type          |
| Cross-domain transfer | Test HumanEval-trained agents on SWE-bench |

---

## 4. Implementation Tasks

### 4.1 Create SWE-bench Loader
```
File: src/swarm/swebench.py

class SWEBenchLoader:
    def load_dataset() -> None
    def get_curated_subset(n_issues: int) -> list[Task]
    def get_issues_by_repo(repo: str) -> list[Task]
    def categorize_issues() -> dict[str, str]
```

### 4.2 Create SWE-bench Executor
```
File: src/swarm/swebench_executor.py

class SWEBenchExecutor:
    def setup_environment(issue: Task) -> None
    def execute_patch(solution: str, issue: Task) -> ExecutionResult
    def run_tests(issue: Task) -> TestResult
```

### 4.3 Adapt Experiment Runner
```
File: scripts/run_swebench_experiment.py

- Load SWE-bench issues instead of HumanEval
- Use SWEBenchExecutor for validation
- Handle longer execution times (10-30min per issue)
- Checkpoint progress frequently
```

### 4.4 Analysis Script
```
File: scripts/analyze_swebench_results.py

- Resolve rate by condition
- Qualitative analysis of solved issues
- Failure mode categorization
- Comparison with Phase 2 HumanEval results
```

---

## 5. Expected Outcomes

### Optimistic Scenario
- Resolve rate: 3-5% for experimental vs 1-2% baseline
- Visible repository specialization
- Some agents "prefer" certain repos/issue types

### Neutral Scenario
- Resolve rate: Similar across conditions
- Specialization exists but doesn't improve performance
- Identifies limits of context-based emergence

### Pessimistic Scenario
- Resolve rate: ~0-1% for all conditions
- No measurable specialization on SWE-bench
- Documents that HumanEval emergence doesn't transfer

---

## 6. Timeline

| Week     | Task                                          |
| -------- | --------------------------------------------- |
| Week 1   | Setup SWE-bench environment + Docker          |
| Week 2   | Curate 50-issue subset + categorization       |
| Week 3   | Implement loader + executor                   |
| Week 4-5 | Run experiments (30-50 issues × 3 conditions) |
| Week 6   | Analysis + documentation                      |

---

## 7. Risk Mitigations

| Risk                      | Mitigation                                  |
| ------------------------- | ------------------------------------------- |
| Very low resolve rate     | Focus on relative comparison, not absolute  |
| Docker/environment issues | Use SWE-bench Lite (simpler setup)          |
| Long execution times      | Checkpoint frequently, use resumption       |
| Compute budget            | Start with 30 issues, expand if time allows |

---

## 8. Handover Checklist

Before starting Phase 3, ensure:
- [ ] Phase 2 experiments completed with statistical significance
- [ ] Phase 2 results documented in `results/` folder
- [ ] SWE-bench harness installed and tested
- [ ] Docker environment configured
- [ ] GPU availability for model inference

---

## 9. Next Prompt Suggestion

> "Let's set up the SWE-bench environment. Please install the SWE-bench harness and verify it works with a single test issue from django or requests."

---

*Plan Version: 1.0*
*Based on: docs/plans/pivot/benchmark-strategy-en.md*
