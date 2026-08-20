# Usage Guide

## Command Line Interface

### Single Task Processing

```bash
python -m src.cli --task "Your task description"
```

### Evaluation Mode

```bash
# SWEBench Lite evaluation
python -m src.cli --evaluate --lite

# Full SWEBench evaluation
python -m src.cli --evaluate
```

### Memory Configuration

```bash
python -m src.cli --task "Task" --memory 16
```

## Python API

```python
from src.main import MultiAgentSystem
import asyncio

async def main():
    system = MultiAgentSystem(max_memory_gb=8)
    await system.initialize()
    
    result = await system.process_task("Write a function")
    print(result)
    
    await system.shutdown()

asyncio.run(main())
```

## Configuration

The system uses YAML configuration files. Default config is in `config/default.yaml`.

Key settings:
- `system.max_memory_gb`: GPU memory limit
- `system.max_tokens`: Context window size
- `models.default_model`: Default SLM to use
