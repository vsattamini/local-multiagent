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
