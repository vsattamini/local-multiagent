# Specialization Experiments: Pushing Toward Emergence

**Date**: 2026-01-24
**Status**: Running
**Baseline**: exp2.1_experimental (S=0.009, D=0.741, Pass@1=57.32%)

---

## Problem Statement

Initial Phase 2 results showed **no specialization emergence**:
- High Context Divergence (D=0.741) — agents accumulated different contexts
- Near-zero Specialization Index (S=0.009) — routing didn't differentiate by task type
- Uniform agent performance (0.4-0.7 success rate across all types)

**Question**: Can we push the system toward specialization?

---

## Root Cause Analysis

### Why Specialization Didn't Emerge

1. **Uniform Success Rates**
   - 57% Pass@1 across all task types
   - All agents converge to ~0.57 success rate for all types
   - Softmax over `[0.57, 0.57, 0.57]` → uniform routing

2. **High Router Temperature**
   - `temperature=0.5` with softmax
   - Small score differences → near-uniform probabilities
   - Little exploitation of performance differences

3. **Coarse Categories**
   - Only 4 task types (string, math, list, logic)
   - May not capture meaningful skill differences

---

## Experiment Design

### Interventions Being Tested

| Experiment          | Hypothesis                                               | Key Change         |
| ------------------- | -------------------------------------------------------- | ------------------ |
| **Low Temperature** | Lower temp → more exploitation → faster lock-in          | `temperature: 0.1` |
| **5 Agents**        | More agents → more competition → specialization pressure | `n_agents: 5`      |
| **Random Control**  | Baseline to compare if affinity adds value               | `router: random`   |
| **7B Model**        | Stronger capability → clearer success/failure signal     | 7B instead of 1.5B |

### Expected Outcomes

```
If low_temp.S > baseline.S:
    → Exploitation helps specialization emerge
    
If 5_agents.S > baseline.S:
    → Competition drives differentiation
    
If random.S ≈ affinity.S:
    → Affinity routing not adding value
    
If 7b.S > 1.5b.S:
    → Capability is the limiting factor
```

---

## Experiment Matrix

| Config                         | Router     | Temp    | Agents | Model  |
| ------------------------------ | ---------- | ------- | ------ | ------ |
| exp2.1_experimental (baseline) | affinity   | 0.5     | 3      | 1.5B   |
| exp_low_temp                   | affinity   | **0.1** | 3      | 1.5B   |
| exp_5_agents                   | affinity   | 0.3     | **5**  | 1.5B   |
| exp_random_control             | **random** | -       | 3      | 1.5B   |
| exp_7b_model                   | affinity   | 0.3     | 3      | **7B** |

---

## Scientific Context

### What "Emergence" Means Here

Emergence = agents naturally develop distinct roles/specializations through interaction with tasks, without explicit programming.

**Positive finding**: S significantly > 0, agents have measurably different success profiles
**Null finding**: S ≈ 0, no natural differentiation despite mechanism presence

Both outcomes are scientifically valuable:
- Positive → validates few-shot context as specialization mechanism
- Null → characterizes limits of emergent specialization in small models

### Research Questions Addressed

1. **RQ1 (Capability)**: Does model size affect emergence? (7B vs 1.5B)
2. **RQ2 (Mechanism)**: Does affinity routing beat random? 
3. **RQ3 (Population)**: Does N=5 specialize more than N=3?

---

## Commands

```bash
# Run all experiments
./scripts/run_specialization_experiments.sh

# Or individually
python scripts/run_experiment.py --config config/exp_low_temp.yaml --seeds 42 --use-full-categories
python scripts/run_experiment.py --config config/exp_5_agents.yaml --seeds 42 --use-full-categories
python scripts/run_experiment.py --config config/exp_random_control.yaml --seeds 42 --use-full-categories
python scripts/run_experiment.py --config config/exp_7b_model.yaml --seeds 42 --use-full-categories
```

---

## Results Template

| Experiment                    | S     | D     | Pass@1 | S p-value | Pattern      |
| ----------------------------- | ----- | ----- | ------ | --------- | ------------ |
| baseline (0.5 temp, 3 agents) | 0.009 | 0.741 | 57.32% | 0.767     | no_emergence |
| low_temp (0.1)                | TBD   | TBD   | TBD    | TBD       | TBD          |
| 5_agents                      | TBD   | TBD   | TBD    | TBD       | TBD          |
| random_control                | TBD   | TBD   | TBD    | TBD       | TBD          |
| 7b_model                      | TBD   | TBD   | TBD    | TBD       | TBD          |

---

*Analysis pending experiment completion*
