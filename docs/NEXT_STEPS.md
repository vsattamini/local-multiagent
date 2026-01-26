# Next Steps: Swarm System Implementation
**Actionable Task List**
*Date: 2026-01-24*

---

## Quick Start: Get Pilot Running in <4 Hours

This document provides step-by-step instructions to get from current state to a working pilot experiment.

---

## Phase 1: Critical Fixes (2-4 hours) 🔴

### Task 1.1: Fix Model Interface Async/Sync Mismatch

**File**: `src/models/llama_cpp.py`
**Effort**: 30 minutes
**Why**: Experiment expects synchronous generate(), but model is async

**Option A: Make Model Synchronous** (RECOMMENDED for pilot)

```python
# src/models/llama_cpp.py - Remove async

def load(self) -> None:
    """Load the model into memory using llama.cpp"""
    if not HAS_LLAMA_CPP:
        raise ImportError("llama-cpp-python is not installed.")

    if not os.path.exists(self.model_path):
        raise FileNotFoundError(f"Model file not found at {self.model_path}")

    print(f"Loading model {self.model_name} from {self.model_path}...")

    self._llm = Llama(
        model_path=self.model_path,
        n_ctx=self.n_ctx,
        n_gpu_layers=self.n_gpu_layers,
        verbose=True
    )
    self.is_loaded = True
    print(f"Model {self.model_name} loaded successfully.")

def unload(self) -> None:
    """Unload the model from memory"""
    if self._llm:
        del self._llm
        self._llm = None
    self.is_loaded = False

def generate(self, prompt: str, **kwargs) -> str:
    """Generate text from the model"""
    if not self.is_loaded or not self._llm:
        raise RuntimeError("Model not loaded")

    max_tokens = kwargs.get("max_tokens", 512)
    temperature = kwargs.get("temperature", 0.2)
    stop = kwargs.get("stop", ["```\n", "User:", "<|endoftext|>"])

    output = self._llm.create_completion(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=stop
    )

    return output["choices"][0]["text"]
```

**Also update** `src/models/interface.py`:

```python
# Remove async from abstract methods
from abc import ABC, abstractmethod

class ModelInterface(ABC):
    # ... existing __init__ ...

    @abstractmethod
    def load(self) -> None:
        """Load the model into memory"""
        pass

    @abstractmethod
    def unload(self) -> None:
        """Unload the model from memory"""
        pass

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from the model"""
        pass

    # ... rest unchanged ...
```

**Test**:
```bash
python -c "
from src.models.llama_cpp import LlamaCppModel
m = LlamaCppModel('test', 'path/to/model.gguf')
print('Import successful')
"
```

---

### Task 1.2: Fix run_pilot.py Model Constructor

**File**: `scripts/run_pilot.py`
**Effort**: 5 minutes
**Why**: Missing `model_name` parameter

**Fix** line 134:

```python
# Before:
model = LlamaCppModel(
    model_path=str(model_path),
    n_ctx=config.context_length,
    n_gpu_layers=-1,
    verbose=False
)

# After:
model = LlamaCppModel(
    model_name="qwen2.5-coder-1.5b",  # ADD THIS
    model_path=str(model_path),
    n_ctx=config.context_length,
    n_gpu_layers=-1
)
# Note: removed verbose param (not in constructor)
```

**Test**:
```bash
python scripts/run_pilot.py --help
# Should not crash on import
```

---

### Task 1.3: Add Model Load Call

**File**: `scripts/run_pilot.py`
**Effort**: 5 minutes
**Why**: Model must be loaded before use

**Add** after model creation (around line 140):

```python
model = LlamaCppModel(...)
print("✓ Model instance created")

# ADD THIS:
print("Loading model into memory...")
model.load()
print("✓ Model loaded successfully")
```

---

### Task 1.4: Verify sentence-transformers Installation

**Effort**: 10 minutes
**Why**: Required for Context Divergence (D) metric

```bash
pip install sentence-transformers

# Test it works:
python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
print('✓ sentence-transformers working')
"
```

---

### Task 1.5: Smoke Test (3 Tasks)

**Effort**: 30 minutes
**Why**: Validate everything works before full pilot

**Create** `scripts/smoke_test.py`:

```python
#!/usr/bin/env python3
"""Quick smoke test with 3 tasks."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.llama_cpp import LlamaCppModel
from swarm.experiment import SwarmExperiment, ExperimentConfig
from swarm.types import TaskType

config = ExperimentConfig(
    model_path="models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
    n_agents=2,
    n_tasks=3,  # Just 3 tasks
    output_dir="results/smoke_test",
    snapshot_interval=2
)

model = LlamaCppModel(
    model_name="qwen2.5-coder-1.5b",
    model_path=config.model_path,
    n_ctx=config.context_length,
    n_gpu_layers=-1
)

print("Loading model...")
model.load()

experiment = SwarmExperiment(model, config)
experiment.load_tasks()

print(f"\nRunning smoke test with {len(experiment.tasks)} tasks...")
metrics = experiment.run()

print(f"\n{'='*60}")
print("SMOKE TEST RESULTS")
print(f"{'='*60}")
print(f"Pass@1: {metrics['summary_stats']['pass_at_1']:.2%}")
print(f"S: {metrics['specialization_index']:.3f}")
print(f"D: {metrics['context_divergence']:.3f}")
print(f"\n✓ Smoke test complete!")
```

**Run**:
```bash
python scripts/smoke_test.py
```

**Expected output**: Should complete without errors, generate 3 task results.

---

## Phase 2: Pilot Readiness (1-2 hours) 🟠

### Task 2.1: Verify Pilot Configuration

**File**: `config/pilot.yaml`
**Effort**: 10 minutes

Check these settings:
- `model.path` points to correct .gguf file
- `agents.n_agents: 3` (as per pilot plan)
- `tasks.n_tasks: 50` (as per pilot plan)
- `router.type: "affinity"` (as per pilot plan)

### Task 2.2: Pre-download HumanEval Dataset

**Effort**: 10 minutes
**Why**: Avoid download issues during experiment

```bash
python -c "
from datasets import load_dataset
ds = load_dataset('openai_humaneval', split='test')
print(f'✓ Downloaded {len(ds)} HumanEval problems')
"
```

### Task 2.3: Disk Space Check

**Effort**: 5 minutes

```bash
# Check available space
df -h .

# Estimate needed:
# - Model: ~1-2 GB
# - Logs: ~10-50 MB per run
# - Plots: ~5 MB per run
# Total: ~2 GB minimum recommended
```

### Task 2.4: Run Full Pilot

**Effort**: 1-4 hours (mostly waiting)
**Why**: This is the main experiment

```bash
# Create results directory
mkdir -p results

# Run pilot
python scripts/run_pilot.py --config config/pilot.yaml

# Expected runtime: 1-4 hours depending on:
# - Model inference speed
# - GPU availability
# - Test execution time
```

**Monitor progress**: Watch console output for task completion.

**If interrupted**: Results are logged incrementally to `results/pilot_001/task_log.jsonl`

### Task 2.5: Analyze Results

**Effort**: 10 minutes

```bash
python scripts/analyze_results.py results/pilot_001
```

**Outputs**:
- `timeline.png` - S(t) and D(t) evolution
- `agent_specialization.png` - Heatmap
- `performance_by_type.png` - Success by task type
- `agent_performance.png` - Success by agent
- `contingency_table.png` - Chi-square visualization
- `analysis_report.txt` - Full text report

**Check**:
- [ ] All plots generated
- [ ] Report shows verdict (GO/MODIFY/PIVOT)
- [ ] S value is calculated
- [ ] D value is calculated
- [ ] F test results shown

---

## Phase 3: Documentation & Validation (2-3 hours) 🟡

### Task 3.1: Document Pilot Results

**File**: Create `results/pilot_001/README.md`
**Effort**: 30 minutes

```markdown
# Pilot Experiment Results

**Date**: [DATE]
**Model**: qwen2.5-coder-1.5b-instruct (Q4_K_M)
**Tasks**: 50 (HumanEval subset)
**Agents**: 3
**Router**: Affinity (temperature=0.5)

## Key Findings

- **Specialization Index (S)**: [VALUE]
- **Context Divergence (D)**: [VALUE]
- **Functional Differentiation (F)**: p=[VALUE]
- **Pass@1**: [VALUE]%

## Verdict

[✓ GO / ⚠️ MODIFY / ✗ PIVOT]

[Reasoning based on decision criteria...]

## Next Steps

[Based on verdict...]
```

### Task 3.2: Validate Metrics Manually

**Effort**: 1 hour

Create `scripts/validate_metrics.py`:

```python
"""Validate metrics with known test data."""

import numpy as np
from src.swarm.metrics import MetricsEngine

# Test 1: Perfect specialization should give S ≈ 1
task_log_perfect = [
    {"agent_id": 0, "task_type": "string", "success": True},
    {"agent_id": 0, "task_type": "string", "success": True},
    {"agent_id": 1, "task_type": "math", "success": True},
    {"agent_id": 1, "task_type": "math", "success": True},
]

metrics = MetricsEngine()
S_perfect = metrics.specialization_index(task_log_perfect)
print(f"Perfect specialization S: {S_perfect:.3f} (expect ~1.0)")
assert S_perfect > 0.9, f"S too low: {S_perfect}"

# Test 2: Random assignment should give S ≈ 0
task_log_random = [
    {"agent_id": 0, "task_type": "string", "success": True},
    {"agent_id": 1, "task_type": "string", "success": True},
    {"agent_id": 0, "task_type": "math", "success": True},
    {"agent_id": 1, "task_type": "math", "success": True},
]

S_random = metrics.specialization_index(task_log_random)
print(f"Random assignment S: {S_random:.3f} (expect ~0.0)")
assert S_random < 0.1, f"S too high: {S_random}"

print("\n✓ All metric validation tests passed!")
```

Run:
```bash
python scripts/validate_metrics.py
```

### Task 3.3: Create Experiment Log

**File**: `docs/EXPERIMENT_LOG.md`
**Effort**: 30 minutes

Keep track of all runs:

```markdown
# Experiment Log

## Run 1: Pilot (2026-01-24)
- **Config**: pilot.yaml
- **Status**: ✓ Complete
- **S**: [VALUE]
- **D**: [VALUE]
- **Verdict**: [GO/MODIFY/PIVOT]
- **Notes**: [Any observations]
- **Files**: results/pilot_001/

## Run 2: [Next experiment]
...
```

---

## Phase 4: Extensions (Optional, 4-8 hours) 🟢

### Task 4.1: Expand Task Categorization

**File**: `src/swarm/humaneval.py`
**Effort**: 2-3 hours (if manual) OR 30 min (if heuristic)

**Option A: Use Heuristic** (Quick):
```python
from src.swarm.humaneval import HumanEvalLoader, expand_task_categorization

loader = HumanEvalLoader()
loader.load_dataset()

full_categorization = expand_task_categorization(loader)
print(f"Categorized {len(full_categorization)}/164 tasks")

# Save for future use
import json
with open('data/humaneval_full_categorization.json', 'w') as f:
    json.dump(
        {k: v.value for k, v in full_categorization.items()},
        f,
        indent=2
    )
```

**Option B: Manual** (Better quality):
- Review HumanEval problems 51-164
- Assign to {string, math, list, logic}
- Update TASK_CATEGORIZATION dict

### Task 4.2: Add Core Unit Tests

**Create**: `tests/test_metrics.py`
**Effort**: 2-3 hours

```python
import pytest
import numpy as np
from src.swarm.metrics import MetricsEngine
from src.swarm.agent import SwarmAgent
from src.swarm.types import TaskType, FewShotExample

class TestSpecializationIndex:
    def test_perfect_specialization(self):
        task_log = [
            {"agent_id": 0, "task_type": "string"},
            {"agent_id": 1, "task_type": "math"},
        ] * 10

        metrics = MetricsEngine()
        S = metrics.specialization_index(task_log)
        assert S > 0.9

    def test_random_assignment(self):
        task_log = [
            {"agent_id": i % 2, "task_type": t}
            for i, t in enumerate(["string", "math"] * 10)
        ]

        metrics = MetricsEngine()
        S = metrics.specialization_index(task_log)
        assert S < 0.1

    def test_empty_log(self):
        metrics = MetricsEngine()
        S = metrics.specialization_index([])
        assert S == 0.0

class TestContextDivergence:
    def test_identical_contexts(self):
        agent1 = SwarmAgent(agent_id=0)
        agent2 = SwarmAgent(agent_id=1)

        # Same examples
        example = FewShotExample(
            problem="test",
            solution="def test(): pass",
            task_type=TaskType.STRING
        )
        agent1.context_buffer.append(example)
        agent2.context_buffer.append(example)

        metrics = MetricsEngine()
        D = metrics.context_divergence([agent1, agent2])
        assert D < 0.1  # Should be nearly identical

    def test_empty_contexts(self):
        agent1 = SwarmAgent(agent_id=0)
        agent2 = SwarmAgent(agent_id=1)

        metrics = MetricsEngine()
        D = metrics.context_divergence([agent1, agent2])
        assert D == 0.0

# Run with: pytest tests/test_metrics.py -v
```

### Task 4.3: Create Baseline Comparison

**Create**: `scripts/run_baseline.py`
**Effort**: 2-3 hours

```python
#!/usr/bin/env python3
"""Run single-model baseline for comparison."""

# Similar structure to run_pilot.py but:
# - Use only 1 agent
# - No context accumulation
# - Random task assignment
# - Log results to results/baseline/

# Compare:
# - Pass@1 vs swarm
# - Tokens used vs swarm
# - Execution time vs swarm
```

### Task 4.4: Implement Agent Removal Experiment

**Create**: `scripts/run_ablation_removal.py`
**Effort**: 2 hours

```python
#!/usr/bin/env python3
"""Test system resilience to agent removal."""

# Protocol:
# 1. Run first 30 tasks normally
# 2. Identify best agent for most common task type
# 3. Remove that agent
# 4. Run remaining 20 tasks
# 5. Measure performance drop

# Metrics:
# - Recovery rate (performance after removal / performance before)
# - S before vs after
# - Time to restabilize
```

---

## Phase 5: Full Thesis Experiments (1-2 weeks)

### Task 5.1: Multiple Seeds

**Goal**: Statistical significance
**Runs**: 3-5 pilots with different random seeds

```bash
for seed in 42 43 44 45 46; do
  python scripts/run_pilot.py \
    --config config/pilot.yaml \
    --output results/pilot_seed_${seed} \
    --seed ${seed}
done

# Aggregate results
python scripts/aggregate_multiple_runs.py results/pilot_seed_*
```

### Task 5.2: Population Size Sweep

**Goal**: Answer RQ3

```bash
for n in 3 5 7 10; do
  python scripts/run_pilot.py \
    --n-agents ${n} \
    --output results/pilot_n${n}
done
```

### Task 5.3: Router Comparison

**Goal**: Validate affinity router vs baselines

```bash
for router in random round_robin greedy affinity; do
  python scripts/run_pilot.py \
    --router ${router} \
    --output results/pilot_router_${router}
done
```

### Task 5.4: Full HumanEval (164 tasks)

**Goal**: Scale up pilot

```bash
python scripts/run_pilot.py \
  --n-tasks 164 \
  --output results/full_humaneval
```

---

## Success Checklist

### Immediate Success (Can Run Pilot)
- [ ] Model interface fixed
- [ ] run_pilot.py fixed
- [ ] Smoke test passes
- [ ] 50-task pilot completes
- [ ] Analysis generates all plots
- [ ] Verdict is clear (GO/MODIFY/PIVOT)

### Short-term Success (Thesis Ready)
- [ ] Metrics validated with test data
- [ ] Multiple seeds run (n=3 minimum)
- [ ] Baseline comparison done
- [ ] At least one ablation study
- [ ] Results documented
- [ ] Figures publication-ready

### Long-term Success (Publishable)
- [ ] Full experimental matrix
- [ ] Statistical significance confirmed
- [ ] Comparison to MetaGPT (or documented gap)
- [ ] Robustness tests complete
- [ ] Code and data released
- [ ] Paper drafted

---

## Emergency Troubleshooting

### Issue: "Model file not found"
```bash
# Check path
ls -lh models/

# Re-download model
# [Instructions for specific model source]
```

### Issue: "CUDA out of memory"
```yaml
# In config/pilot.yaml, reduce:
model:
  context_length: 2048  # Instead of 4096

agents:
  n_agents: 2  # Instead of 3
```

### Issue: "All tests failing"
```python
# Test model output quality
from src.models.llama_cpp import LlamaCppModel

model = LlamaCppModel("test", "models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
model.load()
output = model.generate("def add(a, b):\n    ", max_tokens=50)
print(output)
# Should generate: "return a + b" or similar
```

### Issue: "Metrics give NaN or Inf"
- Check task_log has entries
- Check agent contexts not all empty
- Check task types are valid enum values

### Issue: "Plots don't generate"
```bash
# Install visualization deps
pip install matplotlib seaborn pandas

# Test individually
python -c "import matplotlib; import seaborn; print('OK')"
```

---

## Quick Reference Commands

```bash
# Run smoke test (3 tasks)
python scripts/smoke_test.py

# Run full pilot (50 tasks)
python scripts/run_pilot.py

# Analyze results
python scripts/analyze_results.py results/pilot_001

# Validate metrics
python scripts/validate_metrics.py

# Run tests (when implemented)
pytest tests/ -v

# Check model
python -c "from src.models.llama_cpp import LlamaCppModel; print('OK')"

# Check imports
python -c "from src.swarm import *; print('OK')"
```

---

## Contact & Support

If stuck:
1. Check `docs/IMPLEMENTATION_REVIEW.md` for detailed analysis
2. Review `SWARM_QUICKSTART.md` for setup
3. Check `docs/swarm_system.md` for architecture
4. Review pivot plans in `docs/plans/pivot/`

---

**Document Version**: 1.0
**Last Updated**: 2026-01-24
**Status**: Ready to Execute
