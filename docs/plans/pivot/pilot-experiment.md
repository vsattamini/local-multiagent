# Pilot Experiment Design

## Purpose
Validate the experimental infrastructure and get preliminary signal on whether emergence is even possible before committing to full factorial experiments.

**Time Budget**: 1 weekend (8-16 hours)
**Compute Budget**: Your local GPU
**Success Criteria**: Infrastructure works, metrics are calculable, preliminary data suggests either promise or clear pivot needed

---

## Experiment Configuration

### Fixed Parameters
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Model | Qwen2.5-Coder-1.5B-Instruct (Q4_K_M) | Smallest viable code model, fits easily in VRAM |
| Population size | 3 agents | Minimum for meaningful specialization |
| Benchmark | HumanEval subset (50 problems) | Fast iteration, well-understood |
| Context window | 4096 tokens | Fits ~5-8 few-shot examples |
| Max few-shot examples | 5 per agent | Manageable context size |

### Variable Being Observed
| Variable | How Measured |
|----------|--------------|
| Specialization (S) | Computed every 10 tasks |
| Divergence (D) | Computed every 10 tasks |
| Pass@1 | Computed at end |
| Tokens used | Logged per task |

---

## Task Type Categorization

For this pilot, manually categorize HumanEval problems into 4 types:

| Type | Description | Example Problems |
|------|-------------|------------------|
| **String** | String manipulation, parsing | has_close_elements, separate_paren_groups |
| **Math** | Arithmetic, number theory | truncate_number, below_zero |
| **List** | List operations, filtering | filter_by_prefix, intersperse |
| **Logic** | Conditionals, algorithms | correct_bracketing, monotonic |

This categorization enables you to measure if agents specialize by problem type.

---

## Protocol

### Phase 1: Baseline (Tasks 1-10)
```
For each task:
    1. Route to random agent
    2. Agent generates solution (no few-shot examples yet)
    3. Execute tests
    4. Log: {task_id, agent_id, task_type, success, tokens_used}
    
Purpose: Establish pre-emergence baseline
Expected S: ~0 (random assignment, no context differences)
```

### Phase 2: Context Accumulation (Tasks 11-50)
```
For each task:
    1. Route based on current affinity scores (initially uniform)
    2. Agent generates solution using its accumulated few-shot examples
    3. Execute tests
    4. If success:
        - Add (problem, solution) to agent's context buffer
        - Increase agent's affinity for this task_type
    5. If failure:
        - Log failure pattern (optional: add to negative examples)
        - Decrease agent's affinity for this task_type
    6. Log everything
    
Every 10 tasks:
    - Snapshot all agent contexts
    - Compute S, D
    - Log metrics
```

### Phase 3: Analysis
```
After all 50 tasks:
    1. Compute final S, D, F
    2. Plot S(t) and D(t) over task iterations
    3. Run chi-square test for functional differentiation
    4. Compute Pass@1 and compare to single-model baseline
```

---

## Routing Mechanism (Simple Version)

```python
class SimpleRouter:
    def __init__(self, n_agents: int, task_types: list[str]):
        # Affinity scores: how much each agent "likes" each task type
        # Initialize uniformly
        self.affinity = {
            agent_id: {tt: 1.0 for tt in task_types}
            for agent_id in range(n_agents)
        }
        self.temperature = 0.5  # Controls exploration vs exploitation
    
    def route(self, task_type: str) -> int:
        """Select agent for task using softmax over affinities."""
        scores = [self.affinity[aid][task_type] for aid in range(len(self.affinity))]
        probs = softmax(np.array(scores) / self.temperature)
        return np.random.choice(len(self.affinity), p=probs)
    
    def update(self, agent_id: int, task_type: str, success: bool):
        """Update affinity based on outcome."""
        delta = 0.2 if success else -0.1
        self.affinity[agent_id][task_type] += delta
        # Clamp to [0.1, 5.0] to prevent extremes
        self.affinity[agent_id][task_type] = max(0.1, min(5.0, self.affinity[agent_id][task_type]))
```

---

## Context Accumulation (Simple Version)

```python
class AgentContext:
    def __init__(self, max_examples: int = 5):
        self.examples = []  # List of (problem, solution) tuples
        self.max_examples = max_examples
    
    def add_success(self, problem: str, solution: str):
        """Add successful example to context."""
        self.examples.append((problem, solution))
        # Keep only most recent examples (FIFO)
        if len(self.examples) > self.max_examples:
            self.examples.pop(0)
    
    def build_prompt(self, new_problem: str, system_prompt: str) -> str:
        """Build prompt with few-shot examples."""
        prompt = system_prompt + "\n\n"
        
        for prob, sol in self.examples:
            prompt += f"### Problem:\n{prob}\n\n### Solution:\n```python\n{sol}\n```\n\n"
        
        prompt += f"### Problem:\n{new_problem}\n\n### Solution:\n```python\n"
        return prompt
```

---

## Logging Schema

```python
# Per-task log entry
task_entry = {
    "task_id": "HumanEval/0",
    "task_type": "string",
    "agent_id": 2,
    "timestamp": "2026-01-19T14:30:00",
    "success": True,
    "tokens_generated": 142,
    "latency_ms": 1250,
    "solution": "def has_close_elements(...)...",
    "error_message": None  # or error string if failed
}

# Periodic snapshot (every 10 tasks)
snapshot = {
    "tasks_completed": 20,
    "specialization_index": 0.15,
    "context_divergence": 0.22,
    "agent_contexts": {
        0: ["example1...", "example2..."],
        1: ["example3...", "example4..."],
        2: ["example5..."]
    },
    "affinity_matrix": {
        0: {"string": 1.4, "math": 0.8, "list": 1.0, "logic": 1.0},
        1: {"string": 0.9, "math": 1.3, "list": 0.9, "logic": 1.1},
        2: {"string": 1.0, "math": 1.0, "list": 1.2, "logic": 0.9}
    }
}
```

---

## Expected Outcomes

### Optimistic Scenario
- S increases from ~0 to 0.2-0.3 over 50 tasks
- D increases from ~0 to 0.2-0.4
- Clear visual trend in S(t) plot
- Pass@1 slightly higher than random-routing baseline
- **Conclusion**: Promising signal, proceed to full experiments

### Neutral Scenario
- S stays flat around 0.05-0.1
- D increases slightly but S doesn't
- No clear performance improvement
- **Conclusion**: Mechanism isn't strong enough. Consider: more tasks, different routing, larger context window

### Pessimistic Scenario
- S ≈ 0, D ≈ 0 throughout
- No difference from random baseline
- **Conclusion**: 50 tasks too few, or 3 agents insufficient, or 1.5B model lacks capacity for ICL-based learning. Pivot needed.

---

## Pilot Analysis Checklist

After running the pilot, answer these questions:

1. **Infrastructure**
   - [ ] Model loads and generates without errors
   - [ ] Test execution works reliably
   - [ ] All metrics compute correctly
   - [ ] Logging captures everything needed

2. **Preliminary Signal**
   - [ ] Does S show any upward trend?
   - [ ] Does D show any upward trend?
   - [ ] Is there any visual pattern in the S(t) plot?
   - [ ] Do agent contexts look meaningfully different?

3. **Performance**
   - [ ] What's the single-model baseline Pass@1?
   - [ ] What's the swarm Pass@1?
   - [ ] How many tokens per solution?
   - [ ] What's the wall-clock time per task?

4. **Go/No-Go Decision**
   - If S > 0.1 at any point: **GO** to full experiments
   - If S ≈ 0 but D > 0.2: **GO** with modified routing
   - If S ≈ 0 and D ≈ 0: **PIVOT** (more tasks, larger model, different mechanism)

---

## Time Breakdown

| Activity | Time |
|----------|------|
| Setup infrastructure (model, benchmark, logging) | 2-3 hours |
| Implement router and context accumulation | 2-3 hours |
| Run pilot (50 tasks × 3 agents) | 2-4 hours |
| Compute metrics and analyze | 1-2 hours |
| Document findings | 1 hour |
| **Total** | 8-13 hours |

---

## After the Pilot

If results are promising, the next step is to expand systematically:
- Increase to 100-164 tasks (full HumanEval)
- Test with 5 and 7 agents
- Test with 3B model
- Run multiple seeds for statistical significance

The pilot gives you a **fast feedback loop** to validate your approach before committing significant time.

---

*Document Version: 1.0*
