# Emergence Metrics: Operationalization Guide

## Overview

This document defines the quantitative metrics used to detect, measure, and characterize emergent role specialization in populations of identical small language models. The goal is to distinguish **genuine emergence** (system-level structure arising from local interactions) from **random drift** (statistical noise) or **trivial differentiation** (differences that don't improve performance).

---

## 1. Specialization Index (S)

### What It Measures
The degree to which specific agents become associated with specific task types. High specialization means certain agents consistently handle certain tasks; low specialization means task-agent assignment is random.

### Formal Definition

```
S = 1 - H(task_type | agent) / H(task_type)
```

Where:
- `H(task_type)` = entropy of task type distribution (baseline uncertainty)
- `H(task_type | agent)` = conditional entropy of task type given agent assignment

### Intuition
- **S = 0**: Knowing which agent handled a task tells you nothing about what type of task it was. Assignment is random.
- **S = 1**: Knowing which agent handled a task tells you exactly what type of task it was. Perfect specialization.
- **S ∈ (0, 1)**: Partial specialization. Higher = more specialized.

### Computation

```python
import numpy as np
from collections import Counter

def specialization_index(task_log: list[dict]) -> float:
    """
    task_log: List of {"agent_id": int, "task_type": str, "success": bool}
    Returns: Specialization index S ∈ [0, 1]
    """
    # Count task types
    task_types = [t["task_type"] for t in task_log]
    type_counts = Counter(task_types)
    total = len(task_log)
    
    # H(task_type) - baseline entropy
    p_type = np.array(list(type_counts.values())) / total
    H_task = -np.sum(p_type * np.log2(p_type + 1e-10))
    
    # H(task_type | agent) - conditional entropy
    agents = set(t["agent_id"] for t in task_log)
    H_conditional = 0
    
    for agent in agents:
        agent_tasks = [t for t in task_log if t["agent_id"] == agent]
        p_agent = len(agent_tasks) / total
        
        agent_type_counts = Counter(t["task_type"] for t in agent_tasks)
        p_type_given_agent = np.array(list(agent_type_counts.values())) / len(agent_tasks)
        H_type_given_agent = -np.sum(p_type_given_agent * np.log2(p_type_given_agent + 1e-10))
        
        H_conditional += p_agent * H_type_given_agent
    
    # Specialization index
    S = 1 - (H_conditional / (H_task + 1e-10))
    return max(0, min(1, S))  # Clamp to [0, 1]
```

### Interpretation Guidelines

| S Value | Interpretation |
|---------|----------------|
| 0.0 - 0.1 | No specialization (random assignment) |
| 0.1 - 0.3 | Weak specialization (possibly noise) |
| 0.3 - 0.5 | Moderate specialization (likely real) |
| 0.5 - 0.7 | Strong specialization |
| 0.7 - 1.0 | Very strong specialization (verify not degenerate) |

### Statistical Validation
To confirm S > 0 is not due to chance:
1. Generate null distribution by shuffling agent-task assignments 1000 times
2. Compute S for each shuffled version
3. If observed S > 95th percentile of null distribution, specialization is significant (p < 0.05)

---

## 2. Context Divergence Score (D)

### What It Measures
How different agents' accumulated few-shot examples become over time. If agents are developing distinct "expertise," their context windows should contain different examples.

### Formal Definition

```
D(t) = 1 - mean(cosine_sim(embed(C_i), embed(C_j))) for all agent pairs (i, j)
```

Where:
- `C_i` = concatenated few-shot examples in agent i's context
- `embed()` = sentence embedding function (e.g., all-MiniLM-L6-v2)
- `cosine_sim()` = cosine similarity between embeddings

### Intuition
- **D = 0**: All agents have identical contexts. No differentiation.
- **D = 1**: All agents have completely orthogonal contexts. Maximum divergence.
- **D increasing over time**: Agents are accumulating different experiences → emergence signal.

### Computation

```python
from sentence_transformers import SentenceTransformer
from itertools import combinations
import numpy as np

def context_divergence(agent_contexts: dict[int, list[str]], 
                       model: SentenceTransformer) -> float:
    """
    agent_contexts: {agent_id: [list of few-shot example strings]}
    Returns: Divergence score D ∈ [0, 1]
    """
    # Concatenate each agent's context into single string
    agent_texts = {
        aid: "\n".join(examples) 
        for aid, examples in agent_contexts.items()
    }
    
    # Embed each agent's context
    agent_ids = list(agent_texts.keys())
    embeddings = model.encode([agent_texts[aid] for aid in agent_ids])
    
    # Compute pairwise cosine similarities
    similarities = []
    for (i, j) in combinations(range(len(agent_ids)), 2):
        cos_sim = np.dot(embeddings[i], embeddings[j]) / (
            np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]) + 1e-10
        )
        similarities.append(cos_sim)
    
    # Divergence = 1 - mean similarity
    D = 1 - np.mean(similarities)
    return max(0, min(1, D))
```

### Temporal Analysis
Track D(t) over task iterations to identify:
- **Monotonic increase**: Steady differentiation
- **Plateau**: Stable specialization reached
- **Phase transition**: Sudden jump indicating self-organization event

```python
def divergence_timeline(snapshots: list[dict[int, list[str]]], 
                        model: SentenceTransformer) -> list[float]:
    """
    snapshots: List of agent_contexts at different time points
    Returns: List of D values over time
    """
    return [context_divergence(snap, model) for snap in snapshots]
```

### Interpretation Guidelines

| D Value | Interpretation |
|---------|----------------|
| 0.0 - 0.2 | Low divergence (agents still similar) |
| 0.2 - 0.4 | Moderate divergence |
| 0.4 - 0.6 | Substantial divergence (clear differentiation) |
| 0.6 - 1.0 | High divergence (verify contexts are still coherent) |

---

## 3. Functional Differentiation Score (F)

### What It Measures
Whether performance differences across agents and task types are statistically significant—i.e., whether different agents are genuinely *better* at different things, not just randomly assigned to different things.

### Formal Definition
Chi-square test of independence on the contingency table:

```
            | Task Type A | Task Type B | Task Type C | ...
Agent 1     |   success   |   success   |   success   |
Agent 2     |   success   |   success   |   success   |
Agent 3     |   success   |   success   |   success   |
```

### Intuition
- **p < 0.05**: Agent performance depends on task type. Functional differentiation exists.
- **p ≥ 0.05**: No evidence that agents perform differently on different task types.

### Computation

```python
from scipy.stats import chi2_contingency
import pandas as pd

def functional_differentiation(task_log: list[dict]) -> dict:
    """
    task_log: List of {"agent_id": int, "task_type": str, "success": bool}
    Returns: {"chi2": float, "p_value": float, "significant": bool}
    """
    # Build contingency table: rows = agents, cols = task_types
    # Values = success rate (or count of successes)
    df = pd.DataFrame(task_log)
    
    # Contingency table of success counts
    contingency = pd.crosstab(
        df["agent_id"], 
        df["task_type"], 
        values=df["success"], 
        aggfunc="sum"
    ).fillna(0)
    
    # Chi-square test
    chi2, p_value, dof, expected = chi2_contingency(contingency)
    
    return {
        "chi2": chi2,
        "p_value": p_value,
        "dof": dof,
        "significant": p_value < 0.05,
        "contingency_table": contingency
    }
```

### Advanced: Effect Size (Cramér's V)

Chi-square tells you *if* there's a difference; Cramér's V tells you *how big* it is.

```python
def cramers_v(contingency_table: pd.DataFrame) -> float:
    """
    Returns effect size V ∈ [0, 1]
    """
    chi2, _, _, _ = chi2_contingency(contingency_table)
    n = contingency_table.sum().sum()
    min_dim = min(contingency_table.shape) - 1
    V = np.sqrt(chi2 / (n * min_dim + 1e-10))
    return V
```

| Cramér's V | Effect Size |
|------------|-------------|
| 0.0 - 0.1 | Negligible |
| 0.1 - 0.3 | Small |
| 0.3 - 0.5 | Medium |
| 0.5+ | Large |

---

## 4. Emergence Timeline Analysis

### What It Measures
The *dynamics* of emergence—when and how quickly specialization develops.

### Key Patterns to Identify

**Pattern A: Gradual Drift**
```
S(t): 0.05 → 0.08 → 0.10 → 0.12 → 0.15
```
Slow, linear increase. May indicate random accumulation rather than true emergence.

**Pattern B: Phase Transition**
```
S(t): 0.05 → 0.06 → 0.07 → 0.35 → 0.42 → 0.45
```
Sudden jump followed by plateau. Classic signature of self-organization.

**Pattern C: Oscillation**
```
S(t): 0.10 → 0.25 → 0.12 → 0.30 → 0.15
```
Unstable specialization. May indicate competing attractors or insufficient task diversity.

### Detection Algorithm

```python
def detect_phase_transition(S_timeline: list[float], 
                            threshold: float = 0.15) -> dict:
    """
    Detect sudden jumps in specialization index.
    threshold: Minimum S increase to count as transition
    """
    transitions = []
    for i in range(1, len(S_timeline)):
        delta = S_timeline[i] - S_timeline[i-1]
        if delta > threshold:
            transitions.append({
                "time_step": i,
                "delta_S": delta,
                "S_before": S_timeline[i-1],
                "S_after": S_timeline[i]
            })
    
    return {
        "transitions_detected": len(transitions) > 0,
        "transitions": transitions,
        "final_S": S_timeline[-1],
        "pattern": classify_pattern(S_timeline)
    }

def classify_pattern(S_timeline: list[float]) -> str:
    """Classify emergence pattern."""
    diffs = [S_timeline[i] - S_timeline[i-1] for i in range(1, len(S_timeline))]
    max_jump = max(diffs) if diffs else 0
    variance = np.var(diffs)
    
    if max_jump > 0.15:
        return "phase_transition"
    elif variance > 0.01:
        return "oscillation"
    else:
        return "gradual_drift"
```

---

## 5. Robustness Metrics

### 5.1 Agent Removal Recovery Rate

**What It Measures**: If you remove the "best" agent, does another agent take over its role? This distinguishes genuine system-level specialization from individual-agent quirks.

```python
def removal_recovery_test(system, best_agent_id: int, test_tasks: list) -> dict:
    """
    1. Identify best agent for a task category
    2. Remove that agent
    3. Run system on tasks from that category
    4. Measure if another agent achieves similar performance
    """
    # Performance before removal
    perf_before = system.evaluate(test_tasks)
    
    # Remove best agent
    system.remove_agent(best_agent_id)
    
    # Performance after removal
    perf_after = system.evaluate(test_tasks)
    
    # Recovery rate
    recovery = perf_after / (perf_before + 1e-10)
    
    return {
        "performance_before": perf_before,
        "performance_after": perf_after,
        "recovery_rate": recovery,
        "system_resilient": recovery > 0.7
    }
```

### 5.2 Context Shuffle Sensitivity

**What It Measures**: Is specialization encoded in context content, or is it accidental? Shuffling contexts should destroy genuine specialization.

```python
def context_shuffle_test(system, test_tasks: list) -> dict:
    """
    1. Record performance with current contexts
    2. Shuffle contexts between agents
    3. Measure performance drop
    """
    perf_original = system.evaluate(test_tasks)
    S_original = specialization_index(system.task_log)
    
    # Shuffle contexts
    system.shuffle_contexts()
    
    perf_shuffled = system.evaluate(test_tasks)
    S_shuffled = specialization_index(system.task_log)
    
    return {
        "S_original": S_original,
        "S_shuffled": S_shuffled,
        "S_drop": S_original - S_shuffled,
        "perf_drop": perf_original - perf_shuffled,
        "specialization_is_causal": (S_original - S_shuffled) > 0.1
    }
```

---

## 6. Summary: Claiming "Emergence Happened"

To make a defensible claim that emergence occurred, you need **ALL** of the following:

| Criterion | Metric | Threshold |
|-----------|--------|-----------|
| Specialization exists | S | > 0.2 AND p < 0.05 vs null |
| Contexts diverged | D | > 0.3 |
| Functional benefit | F | p < 0.05 (chi-square) |
| Not random drift | Timeline | Phase transition OR sustained S > 0.3 |
| System-level property | Recovery | > 0.7 after agent removal |
| Causally linked to context | Shuffle test | S drops > 0.1 after shuffle |

If any criterion fails, you have a **partial result** (still publishable, but with caveats).

---

## 7. Null Results and What They Mean

| Observation | Interpretation |
|-------------|----------------|
| S ≈ 0, D ≈ 0 | No emergence. System behaves as random ensemble. |
| S > 0, D ≈ 0 | Routing specialization without context learning. Interesting but limited. |
| S ≈ 0, D > 0 | Contexts diverge but don't affect routing. Context accumulation is noise. |
| S > 0, F not significant | Agents are assigned different tasks but aren't *better* at them. |
| All metrics high, shuffle test fails | Correlation without causation. Context isn't the mechanism. |

Each null result is **informative** and publishable—it tells us what conditions are insufficient for emergence.

---

*Document Version: 1.0*
