# Benchmark Strategy: Justification and Layered Approach

## Executive Summary

This document justifies the choice of a three-layer evaluation strategy for emergence experiments in SLM swarms. The approach combines controlled benchmarks (HumanEval, MBPP) for rigorous statistical validation with realistic benchmarks (SWE-bench Lite) for ecological validation, balancing scientific rigor with practical relevance.

---

## 1. The Benchmark Selection Problem

### 1.1 The Fundamental Dilemma

Benchmark selection involves a trade-off between three dimensions:

```
                    PRACTICAL RELEVANCE
                           ▲
                          /|\
                         / | \
                        /  |  \
                       /   |   \
                      /    |    \
                     /     |     \
                    /______|______\
    CONTROLLABILITY ←────────────→ DIFFICULTY
```

| Benchmark | Controllability | Difficulty | Relevance |
|-----------|------------------|-------------|------------|
| HumanEval | ✅ High | ⚠️ Medium | ❌ Low |
| MBPP | ✅ High | ⚠️ Medium | ❌ Low |
| SWE-bench Lite | ⚠️ Medium | ✅ High | ✅ High |
| SWE-bench Full | ❌ Low | ✅ Very High | ✅ High |

**There is no perfect benchmark.** Each choice implies limitations that must be explicitly acknowledged.

### 1.2 Why Not Use Only HumanEval?

**Critical limitations:**

1. **Artificial problems**: These are algorithmic puzzles created specifically for the benchmark, not representative of real development work

2. **Limited scope**: Each problem is an isolated 5-20 line function; doesn't test context comprehension, codebase navigation, or multi-file editing

3. **Saturation**: State-of-the-art models already achieve >90% on HumanEval; little room to demonstrate significant improvements

4. **Common criticism**: Reviewers frequently question the relevance of HumanEval results to real applications

**Example of typical HumanEval problem:**
```python
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """Check if any two numbers in list are closer than threshold."""
    # ~10 lines solution
```

This does not represent the work of a real developer.

### 1.3 Why Not Use Only SWE-bench?

**Critical limitations:**

1. **SLM performance is very low**: Small models (<7B) typically solve <5% of issues, making statistical analysis of emergence difficult

2. **Setup complexity**: Requires Docker, repository cloning, environment configuration — dramatically increases experiment time

3. **High variance**: Issues vary enormously in difficulty (5 minutes to 4+ hours for humans); makes controlled comparisons difficult

4. **Computational cost**: Each issue requires multiple iterations of generation + execution of complete test suites

**Reported performance on SWE-bench Verified:**

| System | Resolve Rate |
|---------|--------------|
| Verdent (multi-agent, frontier models) | 76.1% |
| Claude Sonnet 4.5 | ~50% |
| GPT-4o | ~30% |
| Open-source 70B models | ~15-20% |
| 7B models | ~5-10% |
| **1.5-3B models (our case)** | **<5% estimated** |

With such low resolution rates, we would need hundreds of issues to have sufficient statistical power to detect emergence.

---

## 2. Proposed Strategy: Layered Approach

### 2.1 Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    LAYER 3: ECOLOGICAL VALIDATION               │
│                         SWE-bench Lite                          │
│                    (30-50 selected issues)                      │
│                                                                 │
│   Purpose: Test transfer to realistic tasks                     │
│   Primary metric: Comparative resolve rate                      │
│   Question: "Does specialization help with real problems?"      │
├─────────────────────────────────────────────────────────────────┤
│                   LAYER 2: STATISTICAL VOLUME                   │
│                     Full HumanEval + MBPP                       │
│                       (164 + 500 = 664 problems)                │
│                                                                 │
│   Purpose: Statistical power for emergence metrics              │
│   Primary metrics: S, D, F with confidence intervals            │
│   Question: "Is emergence statistically significant?"           │
├─────────────────────────────────────────────────────────────────┤
│                    LAYER 1: PILOT AND VALIDATION                │
│                     HumanEval Subset (50 problems)              │
│                                                                 │
│   Purpose: Validate infrastructure, detect initial signal       │
│   Primary metrics: S, D working, baseline established           │
│   Question: "Does the mechanism work? Worth continuing?"        │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Layer 1: Pilot and Validation

**Benchmark:** HumanEval subset (50 problems)

**Purpose:**
- Validate that infrastructure works correctly
- Verify that emergence metrics are computable
- Detect preliminary differentiation signal
- Go/no-go decision before investing in full experiments

**Configuration:**
- 3 agents
- Qwen2.5-Coder-1.5B
- 50 manually categorized problems

**Success criteria:**
- Infrastructure runs without errors
- S > 0.1 at some point (differentiation signal)
- Execution time < 4 hours

**Estimated time:** 1 weekend

### 2.3 Layer 2: Statistical Volume

**Benchmarks:** Full HumanEval (164) + MBPP (500)

**Purpose:**
- Sufficient volume for statistical significance
- Direct comparison with literature (MetaGPT, ChatDev, AgentCoder)
- Full factorial analysis of independent variables

**Configuration:**
- N agents variation: 3, 5, 7, 10
- Model variation: 1.5B, 3B
- Feedback variation: binary, graded
- Multiple seeds for confidence intervals

**Primary metrics:**
- Pass@1, Pass@5
- Specialization Index (S) with 95% CI
- Context Divergence (D) with 95% CI
- Functional Differentiation (F) with χ² test
- Tokens per solution (efficiency)

**Analyses:**
- ANOVA for N agents effect
- Regression for S × performance relationship
- Emergence curves over time

**Estimated time:** 4-6 weeks

### 2.4 Layer 3: Ecological Validation

**Benchmark:** SWE-bench Lite (curated subset of 30-50 issues)

**Purpose:**
- Test if emergent specialization transfers to realistic tasks
- Practical relevance validation
- Identification of limitations and future directions

**Issue Selection:**

Curate subset based on:
1. **Estimated difficulty**: Focus on 15min-1h issues (intermediate category)
2. **Task type**: Include bug fixes, small features, refactors
3. **Repo diversity**: Distribute across different repositories

**Selection criteria:**
```python
criteria = {
    "estimated_human_time": "15min - 1h",
    "files_modified": "<= 3",
    "lines_modified": "<= 100",
    "clear_tests": True,
    "complete_description": True
}
```

**Comparisons:**
- Emergent swarm vs. single model (same token budget)
- Emergent swarm vs. random routing (control)
- Emergent swarm vs. explicit roles (if time permits)

**Metrics:**
- Absolute resolve rate
- Relative resolve rate vs. baseline
- Qualitative analysis of success/failure cases

**Realistic expectations:**
- Resolve rate will likely be low (<10%)
- Value is in **relative comparison**, not absolute numbers
- Even 2% → 5% would be evidence that specialization helps

**Estimated time:** 2-3 weeks

---

## 3. Task Categorization

### 3.1 HumanEval/MBPP: Manual Categorization

To measure specialization, we need to categorize problems by type. Proposed taxonomy:

| Category | Description | HumanEval Examples |
|-----------|-----------|-------------------|
| **String** | Text manipulation, parsing, formatting | separate_paren_groups, remove_vowels |
| **Math** | Arithmetic, number theory, calculations | truncate_number, sum_squares |
| **List** | List operations, filtering, transformation | filter_by_prefix, intersperse |
| **Logic** | Conditionals, algorithms, control structures | correct_bracketing, monotonic |
| **Search** | Search, sorting, optimization | find_closest_elements, sort_by_binary |

**Categorization process:**
1. Two annotators categorize independently
2. Calculate inter-annotator agreement (Cohen's κ)
3. Resolve disagreements through discussion
4. Publish categorization as contribution

### 3.2 SWE-bench: Metadata-Based Categorization

SWE-bench issues have metadata that enable categorization:

| Dimension | Values |
|----------|---------|
| **Repository** | astropy, django, flask, matplotlib, etc. |
| **Type** | bug, feature, refactor, docs |
| **Difficulty** | <15min, 15min-1h, 1-4h, >4h |
| **Files** | 1, 2-3, 4+ |

This enables analysis of:
- Repository-based specialization (does an agent "learn" Django?)
- Task type specialization
- Difficulty × specialization interaction

---

## 4. Justification for Each Benchmark

### 4.1 HumanEval

**Why include:**
- Most cited benchmark in code generation literature
- MetaGPT, ChatDev, AgentCoder all report results
- Enables direct comparison and work positioning
- Problems are fast to execute (seconds each)

**Acknowledged limitations:**
- Does not represent real development
- Problems are isolated and artificial
- Saturation by large models

**How to address in dissertation:**
> "We use HumanEval as the primary benchmark due to its wide adoption in the literature, enabling direct comparison with previous work. We acknowledge that isolated single-function problems do not capture the complexity of real software development, motivating our additional validation with SWE-bench."

### 4.2 MBPP

**Why include:**
- 500 problems provide statistical volume
- Problems are slightly more diverse than HumanEval
- Also widely reported in literature

**Acknowledged limitations:**
- Same fundamental limitations as HumanEval
- Some problems have ambiguous descriptions

**How to use:**
- Combined with HumanEval for aggregate analysis
- Statistical power to detect small effects
- Cross-validation of patterns observed in HumanEval

### 4.3 SWE-bench Lite

**Why include:**
- Only benchmark that tests real development
- Issues are from popular open-source repositories
- Requires skills that should benefit from specialization:
  - Codebase navigation
  - Context comprehension
  - Multi-file editing

**Acknowledged limitations:**
- SLM performance will be very low
- Setup is complex and time-consuming
- High variance between issues

**How to use:**
- Not as primary success metric
- As "stress test" of practical relevance
- Qualitative analysis beyond quantitative
- Curated subset for feasibility

---

## 5. Risks and Mitigations

### 5.1 Risk: Null Results on SWE-bench

**Scenario:** Swarm resolves 0-2% of issues, indistinguishable from baseline

**Mitigation:**
- Focus on relative comparison (swarm vs. single model)
- Qualitatively analyze the few successful cases
- Document as "limitations and future work"
- Argument still valid: "emergence detected in HumanEval does not transfer to complex tasks — implications for multi-agent system design"

### 5.2 Risk: Emergence in HumanEval But Not in SWE-bench

**Scenario:** S > 0.3 in HumanEval, but no difference in SWE-bench

**Interpretation:**
- Specialization works for simple/repetitive tasks
- Complex tasks require capabilities beyond specialization
- Identifies limit of proposed paradigm

**How to report:**
> "We observed significant emergent specialization in single-function benchmarks, but this specialization did not translate to measurable improvement in realistic software engineering tasks. This suggests that multi-agent coordination with SLMs may be effective for simple problem decomposition, but insufficient for tasks requiring long-range reasoning over complex codebases."

### 5.3 Risk: HumanEval Too Easy to Detect Emergence

**Scenario:** Single model already solves >70%, little room for improvement

**Mitigation:**
- Measure not only Pass@1, but also quality and efficiency
- Focus on emergence metrics (S, D) independent of absolute performance
- Use MBPP for more diverse problems

---

## 6. Metrics by Layer

### 6.1 Metrics Common to All Layers

| Metric | Definition | Purpose |
|---------|-----------|-----------|
| **Pass@1** | % solved on first attempt | Performance |
| **S (Specialization)** | 1 - H(type\|agent)/H(type) | Emergence |
| **D (Divergence)** | 1 - mean(cos_sim(contexts)) | Differentiation |
| **Tokens/solution** | Average tokens generated | Efficiency |

### 6.2 Layer-Specific Metrics

**Layer 1 (Pilot):**
- Total execution time
- Infrastructure errors
- Qualitative feasibility

**Layer 2 (Volume):**
- 95% confidence intervals for all metrics
- Significance tests (ANOVA, χ²)
- Effect sizes (Cohen's d, Cramér's V)

**Layer 3 (Ecological):**
- Resolve rate by issue category
- Case analysis (successes and failures)
- Comparison with existing systems (if data available)

---

## 7. Integrated Timeline

| Week | Activity | Benchmark | Deliverable |
|--------|-----------|-----------|------------|
| 1 | Infrastructure setup | - | Working code |
| 2 | Pilot | HumanEval (50) | Pilot report, go/no-go decision |
| 3-4 | HumanEval experiments | HumanEval (164) | Raw data |
| 5-6 | MBPP experiments | MBPP (500) | Raw data |
| 7 | Statistical analysis | HumanEval + MBPP | Layer 2 results |
| 8 | SWE-bench setup | - | Configured environment |
| 9-10 | SWE-bench experiments | SWE-bench Lite (50) | Raw data |
| 11 | Integrated analysis | All | Results chapter |
| 12+ | Writing | - | Dissertation |

---

## 8. Response to Anticipated Criticisms

### Criticism 1: "HumanEval does not represent real development"

**Response:**
> We agree. That is why we include validation on SWE-bench Lite, which uses real GitHub issues. Our use of HumanEval is justified by the need for comparison with existing literature and the statistical volume required to detect emergence. SWE-bench results, even if limited, provide evidence about transfer to realistic tasks.

### Criticism 2: "SWE-bench performance will be too low to be useful"

**Response:**
> Our goal is not to compete with state-of-the-art systems on SWE-bench, but to test whether emergent specialization improves relative performance compared to appropriate baselines (single model, random routing). Even small improvements (2% → 4%) would be evidence that the mechanism has value for complex tasks.

### Criticism 3: "Why not use only SWE-bench from the start?"

**Response:**
> The expected resolution rate for SLMs on SWE-bench (<5%) would make emergence detection statistically impractical. We would need hundreds of issues to have sufficient power. HumanEval/MBPP provide the necessary volume (664 problems) for robust emergence analysis, while SWE-bench validates the practical relevance of findings.

---

## 9. Conclusion

The layered strategy enables:

1. **Fast iteration** in pilot before committing resources
2. **Statistical rigor** with sufficient problem volume
3. **Comparability** with existing literature
4. **Practical relevance** through ecological validation
5. **Honesty** about each benchmark's limitations

No benchmark is perfect. The proposed combination maximizes the strengths of each while explicitly acknowledging their limitations.

---

*Document Version: 1.0*
