# Swarm System Quick Start Guide

## What is the Swarm System?

The swarm system is a **pivot** from the original explicitly-designed multi-agent architecture to a **context-accumulation swarm** where:

- ✅ Agents are **identical frozen models** (no weight updates)
- ✅ Differentiation happens through **accumulated few-shot examples** in context
- ✅ Routing is **learned** via affinity scores (success rates by task type)
- ✅ Emergence is **measured quantitatively** (S, D, F metrics)

This tests the hypothesis: **Can identical small LMs develop functional role specialization without explicit role assignment?**

## Installation

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download a model (example)
mkdir -p models
cd models
# Download qwen2.5-coder-1.5b-instruct-q4_k_m.gguf from HuggingFace
# URL: https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF
cd ..
```

## Running the Pilot Experiment

### Step 1: Configure

Edit `config/pilot.yaml`:

```yaml
agents:
  n_agents: 3  # Try 3, 5, or 7

router:
  type: "affinity"  # Or: random, round_robin, greedy
  temperature: 0.5

tasks:
  n_tasks: 50  # Pilot uses 50 tasks
```

### Step 2: Run

```bash
python scripts/run_pilot.py
```

**What this does:**
1. Loads 50 HumanEval problems (manually categorized into string/math/list/logic)
2. Initializes 3 identical agents with empty context buffers
3. Routes each task using affinity scores
4. Agents generate solutions with few-shot examples from their context
5. Tests are executed and agents update based on success/failure
6. Metrics (S, D, F) are computed every 10 tasks
7. Final analysis determines if emergence occurred

**Expected runtime:** 1-4 hours depending on hardware

### Step 3: Analyze

```bash
python scripts/analyze_results.py results/pilot_001
```

**Generated outputs:**
- `timeline.png` - S(t) and D(t) over time
- `agent_specialization.png` - Heatmap of agent success rates by task type
- `performance_by_type.png` - Success breakdown
- `contingency_table.png` - Chi-square test visualization
- `analysis_report.txt` - Full text report with verdict

## Understanding the Results

### Key Metrics

| Metric | What it means | Threshold |
|--------|---------------|-----------|
| **S** (Specialization Index) | How much agents specialize in different task types | S > 0.1 = emergence |
| **D** (Context Divergence) | How different agents' contexts become | D > 0.2 = meaningful |
| **F** (Functional Differentiation) | Are performance differences statistically real? | p < 0.05 = yes |

### Decision Criteria

✅ **GO** (S > 0.1): Specialization emerged! Proceed to full experiments

⚠️ **MODIFY** (S ≈ 0, D > 0.2): Contexts diverged but routing needs adjustment

❌ **PIVOT** (S ≈ 0, D ≈ 0): No emergence. Try:
- More tasks (100-164 instead of 50)
- Larger model (3B instead of 1.5B)
- Different routing mechanism

## Experiment Variants

### Test Different Routers

```bash
# Random baseline (control)
python scripts/run_pilot.py --router random --output results/pilot_random

# Round-robin baseline
python scripts/run_pilot.py --router round_robin --output results/pilot_rr

# Greedy (pure exploitation)
python scripts/run_pilot.py --router greedy --output results/pilot_greedy
```

### Test Population Size

```bash
# 5 agents
python scripts/run_pilot.py --n-agents 5 --output results/pilot_5agents

# 7 agents
python scripts/run_pilot.py --n-agents 7 --output results/pilot_7agents
```

### Use More Tasks

```bash
# All 164 HumanEval problems (requires updating humaneval.py categorization)
python scripts/run_pilot.py --n-tasks 164 --output results/full_humaneval
```

## File Guide

### Core Implementation

```
src/swarm/
├── agent.py         - SwarmAgent with context buffer
├── router.py        - AffinityRouter and variants
├── executor.py      - Safe code execution
├── metrics.py       - S, D, F calculations
├── logger.py        - JSONL logging
├── humaneval.py     - Benchmark integration
├── experiment.py    - Main orchestrator
└── types.py         - Data models
```

### Scripts

```
scripts/
├── run_pilot.py         - Run experiment
└── analyze_results.py   - Generate visualizations
```

### Configuration

```
config/
└── pilot.yaml          - Experiment parameters
```

### Documentation

```
docs/
├── swarm_system.md                      - Full system documentation
└── plans/pivot/
    ├── research-plan.md                 - Research questions
    ├── code-architecture.md             - System design
    ├── metrics-document.md              - Metric definitions
    └── pilot-experiment.md              - Pilot protocol
```

## Next Steps After Pilot

### If Emergence Detected (S > 0.1)

1. **Scale up**: Run with 5, 7, 10 agents
2. **Full HumanEval**: All 164 problems
3. **Model variants**: Test with 3B model
4. **Context strategies**: Try category-weighted examples
5. **Ablation studies**: Remove best agent, shuffle contexts

### If No Emergence (S ≈ 0)

1. **Diagnose**: Is D > 0? Then routing issue. D ≈ 0? Context not accumulating.
2. **More data**: Try 100-164 tasks instead of 50
3. **Stronger model**: Use 3B instead of 1.5B
4. **Different feedback**: Try graded metrics instead of binary
5. **Check categorization**: Verify task types are meaningful

## Comparison to Original System

| Aspect | Original System | Swarm System |
|--------|----------------|--------------|
| **Architecture** | Explicit specialist agents | Identical agents |
| **Differentiation** | Hardcoded roles | Emergent via context |
| **Coordination** | Coordinator agent | Affinity-based routing |
| **Learning** | None (frozen) | Context accumulation |
| **Metrics** | Task success | S, D, F emergence metrics |
| **Goal** | Software development | Research on emergence |

**The original system is still available** in `src/agents/` and `src/coordination/`.

## Troubleshooting

### "Model file not found"

```bash
# Check model path in config/pilot.yaml matches actual file
ls -lh models/*.gguf
```

### "datasets not found"

```bash
pip install datasets
```

### "Out of memory"

Reduce in `config/pilot.yaml`:
```yaml
agents:
  n_agents: 2  # Instead of 3
model:
  context_length: 2048  # Instead of 4096
```

### "All tests failing"

Check model output quality:
```python
from models.llama_cpp import LlamaCppModel

model = LlamaCppModel("models/your-model.gguf")
output = model.generate("def add(a, b):", max_tokens=50)
print(output)
```

## Citation

```bibtex
@mastersthesis{lofgren2026emergent,
  title={Emergent Role Specialization in Minimal Language Model Populations},
  author={Lofgren, Vinicius},
  year={2026}
}
```

## Support

- 📖 Full docs: [docs/swarm_system.md](docs/swarm_system.md)
- 🔬 Research plan: [docs/plans/pivot/research-plan.md](docs/plans/pivot/research-plan.md)
- 📊 Metrics guide: [docs/plans/pivot/metrics-document.md](docs/plans/pivot/metrics-document.md)

---

**Ready to test if swarms can self-organize? Run the pilot!** 🚀
