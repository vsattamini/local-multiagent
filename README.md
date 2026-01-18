# Local Multi-Agent Coding System

A swarm-adjacent multi-agent system using Small Language Models (SLMs) that can run on consumer GPUs (8-16GB) and perform coding tasks evaluated on SWEBench Lite.

## Features

- **Hybrid Swarm Architecture**: Lightweight coordinator with specialist agents
- **Memory Efficient**: Sequential agent execution within GPU constraints
- **Local Processing**: Runs entirely on local hardware
- **Self-Correction**: AST-based syntax validation with automatic retry loops
- **Reliable Generation**: Strictly constrained prompts to prevent hallucinations and verbose output
- **SWEBench Integration**: Evaluated on SWEBench Lite coding tasks
- **Extensible**: Easy to add new specialist agents

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run a single task
python -m src.cli --task "Write a function to add two numbers"

# Run SWEBench evaluation
python -m src.cli --evaluate --lite
```

## Architecture

- **Coordinator Agent**: Task decomposition and result synthesis
- **Specialist Agents**: Code generation, testing, debugging, review
- **Model Manager**: Memory-efficient SLM loading/unloading
- **Context Manager**: Token-aware context handling

## Requirements

- Python 3.9+
- GPU with 8-16GB VRAM
- 16GB+ system memory recommended

## Configuration

Edit `config/default.yaml` to customize:
- Memory limits
- Model paths
- Agent settings
- Evaluation parameters
