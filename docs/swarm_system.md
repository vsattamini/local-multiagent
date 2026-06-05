# Swarm-Based Multi-Agent System

## Overview

This system implements a **context-accumulation swarm** where identical frozen language models develop emergent role specialization through accumulated experience, rather than explicit role assignment.

## Architecture

### Core Components

1. **SwarmAgent** ([src/swarm/agent.py](../src/swarm/agent.py))
   - Context buffer with FIFO management
   - Task history tracking by type
   - Success rate computation
   - Few-shot prompt construction

2. **Router** ([src/swarm/router.py](../src/swarm/router.py))
   - **AffinityRouter**: Softmax-based selection using success rates
   - **RandomRouter**: Baseline uniform selection
   - **RoundRobinRouter**: Cyclic assignment
   - **GreedyRouter**: Pure exploitation

3. **Executor** ([src/swarm/executor.py](../src/swarm/executor.py))
   - Safe code execution with timeout
   - Subprocess isolation
   - HumanEval-specific handling

4. **MetricsEngine** ([src/swarm/metrics.py](../src/swarm/metrics.py))
   - **Specialization Index (S)**: Entropy-based measure
   - **Context Divergence (D)**: Embedding similarity
   - **Functional Differentiation (F)**: Chi-square test
   - Phase transition detection

5. **ExperimentLogger** ([src/swarm/logger.py](../src/swarm/logger.py))
   - Per-task JSONL logging
   - Periodic snapshots
   - Final metrics export

## Running Experiments

### Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download model (example with HuggingFace)
mkdir -p models
cd models
# Download qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
cd ..

# Run pilot experiment
python scripts/run_pilot.py --config config/pilot.yaml
```

### Configuration

Edit `config/pilot.yaml` to customize:

```yaml
agents:
  n_agents: 3  # Number of agents in swarm
  max_context_examples: 5  # Context buffer size

router:
  type: "affinity"  # affinity, random, round_robin, greedy
  temperature: 0.5  # Exploration vs exploitation

tasks:
  n_tasks: 50  # Number of tasks for pilot
```

### Analyzing Results

```bash
# Generate visualizations and report
python scripts/analyze_results.py results/pilot_001
```

## Metrics

### Specialization Index (S)

$$S = 1 - \frac{H(\text{task\_type} | \text{agent})}{H(\text{task\_type})}$$

- **S = 0**: Random assignment (no specialization)
- **S = 1**: Perfect specialization
- **Threshold**: S > 0.1 indicates emergence

### Context Divergence (D)

$$D = 1 - \text{mean}(\text{cosine\_sim}(\text{context}_i, \text{context}_j))$$

- **D = 0**: Identical contexts
- **D = 1**: Orthogonal contexts
- **Threshold**: D > 0.2 indicates meaningful divergence

### Functional Differentiation (F)

Chi-square test of independence on agent × task_type contingency table.

- **p < 0.05**: Agents perform differently on different task types
- **Cramér's V**: Effect size measure

## Pilot Experiment Decision Criteria

Based on [docs/plans/pivot/pilot-experiment.md](plans/pivot/pilot-experiment.md):

| Condition | Decision |
|-----------|----------|
| S > 0.1 | ✅ GO: Proceed to full experiments |
| S ≈ 0, D > 0.2 | ⚠️ GO with modifications: Context diverges but routing needs adjustment |
| S ≈ 0, D ≈ 0 | ❌ PIVOT: Need more tasks, larger model, or different mechanism |

## File Structure

```
src/swarm/
├── __init__.py           # Package exports
├── agent.py              # SwarmAgent with context accumulation
├── router.py             # Routing strategies
├── executor.py           # Safe code execution
├── metrics.py            # Emergence metrics
├── logger.py             # Experiment logging
├── humaneval.py          # HumanEval benchmark integration
├── experiment.py         # Main experiment orchestrator
└── types.py              # Type definitions

config/
└── pilot.yaml            # Pilot experiment configuration

scripts/
├── run_pilot.py          # Run pilot experiment
└── analyze_results.py    # Analyze and visualize results

results/
└── pilot_001/            # Experiment outputs
    ├── task_log.jsonl    # Per-task results
    ├── snapshots.jsonl   # Periodic system state
    ├── final_metrics.json # Final analysis
    ├── config.json       # Experiment configuration
    └── *.png             # Generated visualizations
```

## Research Questions

From [docs/plans/pivot/research-plan.md](plans/pivot/research-plan.md):

**RQ1**: Under what conditions can identical SLMs develop functional role specialization?

**RQ2**: What feedback mechanisms most effectively induce differentiation?

**RQ3**: How does population size affect emergence?

**RQ4**: Performance vs. explicitly-designed systems (MetaGPT, ChatDev)?

**RQ5**: What is the accessibility frontier (performance vs. hardware cost)?

## References

- **Research Plan**: [docs/plans/pivot/research-plan.md](plans/pivot/research-plan.md)
- **Architecture**: [docs/plans/pivot/code-architecture.md](plans/pivot/code-architecture.md)
- **Metrics**: [docs/plans/pivot/metrics-document.md](plans/pivot/metrics-document.md)
- **Pilot Protocol**: [docs/plans/pivot/pilot-experiment.md](plans/pivot/pilot-experiment.md)

## Contributing

When extending the system:

1. **Router variants**: Implement new routing strategies in `router.py`
2. **Context strategies**: Modify `agent.py` for different context management
3. **Metrics**: Add new emergence metrics to `metrics.py`
4. **Benchmarks**: Extend `humaneval.py` or create new task loaders

## Troubleshooting

### Model Loading Issues

```bash
# Check model file exists
ls -lh models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf

# Install llama-cpp-python with GPU support
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

### Memory Issues

- Reduce `n_agents` in config
- Use smaller model (1.5B instead of 3B)
- Reduce `context_length`

### Dataset Loading Issues

```bash
# Install datasets library
pip install datasets

# May need to set HuggingFace cache
export HF_HOME=/path/to/cache
```

## Citation

If you use this system in your research, please cite:

```bibtex
@mastersthesis{lofgren2026emergent,
  title={Emergent Role Specialization in Minimal Language Model Populations},
  author={Lofgren, Vinicius},
  year={2026},
  school={Your Institution}
}
```
