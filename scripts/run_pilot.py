#!/usr/bin/env python3
"""
Run pilot experiment for swarm-based code generation.

Usage:
    python scripts/run_pilot.py [--config config/pilot.yaml]
"""

import argparse
import sys
from pathlib import Path
import yaml
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.llama_cpp import LlamaCppModel
from swarm.experiment import SwarmExperiment, ExperimentConfig
from swarm.types import TaskType


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def config_from_yaml(yaml_config: dict) -> ExperimentConfig:
    """Convert YAML config to ExperimentConfig."""
    exp_config = yaml_config.get('experiment', {})
    model_config = yaml_config.get('model', {})
    agent_config = yaml_config.get('agents', {})
    router_config = yaml_config.get('router', {})
    task_config = yaml_config.get('tasks', {})
    exec_config = yaml_config.get('execution', {})
    metrics_config = yaml_config.get('metrics', {})
    prompts = yaml_config.get('prompts', {})

    # Convert task types
    task_types = [
        TaskType(tt) for tt in task_config.get('task_types', ['string', 'math', 'list', 'logic'])
    ]

    return ExperimentConfig(
        model_path=model_config.get('path', 'models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf'),
        context_length=model_config.get('context_length', 4096),
        max_tokens=model_config.get('max_tokens', 512),
        n_agents=agent_config.get('n_agents', 3),
        max_context_examples=agent_config.get('max_context_examples', 5),
        router_type=router_config.get('type', 'affinity'),
        router_temperature=router_config.get('temperature', 0.5),
        n_tasks=task_config.get('n_tasks', 50),
        task_types=task_types,
        snapshot_interval=metrics_config.get('snapshot_interval', 10),
        output_dir=exp_config.get('output_dir', 'results/pilot_001'),
        timeout=exec_config.get('timeout', 5),
        system_prompt=prompts.get('system', 'You are a Python coding assistant.'),
        random_seed=yaml_config.get('random_seed', 42)
    )


def main():
    parser = argparse.ArgumentParser(description='Run swarm pilot experiment')
    parser.add_argument(
        '--config',
        type=str,
        default='config/pilot.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--model',
        type=str,
        help='Override model path from config'
    )
    parser.add_argument(
        '--n-agents',
        type=int,
        help='Override number of agents from config'
    )
    parser.add_argument(
        '--n-tasks',
        type=int,
        help='Override number of tasks from config'
    )
    parser.add_argument(
        '--router',
        type=str,
        choices=['affinity', 'random', 'round_robin', 'greedy'],
        help='Override router type from config'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Override output directory from config'
    )

    args = parser.parse_args()

    # Load config
    print(f"Loading config from: {args.config}")
    yaml_config = load_config(args.config)
    config = config_from_yaml(yaml_config)

    # Apply overrides
    if args.model:
        config.model_path = args.model
    if args.n_agents:
        config.n_agents = args.n_agents
    if args.n_tasks:
        config.n_tasks = args.n_tasks
    if args.router:
        config.router_type = args.router
    if args.output:
        config.output_dir = args.output

    # Set random seed
    if config.random_seed is not None:
        np.random.seed(config.random_seed)

    # Check if model file exists
    model_path = Path(config.model_path)
    if not model_path.exists():
        print(f"\n❌ Error: Model file not found: {config.model_path}")
        print(f"\nPlease download the model first:")
        print(f"  mkdir -p models")
        print(f"  cd models")
        print(f"  # Download from HuggingFace or other source")
        sys.exit(1)

    # Initialize model
    print(f"\nInitializing model: {config.model_path}")
    try:
        model = LlamaCppModel(
            model_name="qwen2.5-coder-1.5b",
            model_path=str(model_path),
            n_ctx=config.context_length,
            n_gpu_layers=-1
        )
        print("✓ Model instance created")

        print("Loading model into memory...")
        model.load()
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)

    # Create experiment
    print("\nInitializing experiment...")
    experiment = SwarmExperiment(model, config)

    # Load tasks
    print("Loading HumanEval tasks...")
    try:
        experiment.load_tasks()
    except Exception as e:
        print(f"❌ Error loading tasks: {e}")
        print("\nYou may need to install the datasets library:")
        print("  pip install datasets")
        sys.exit(1)

    # Run experiment
    try:
        metrics = experiment.run()
    except KeyboardInterrupt:
        print("\n\n⚠️  Experiment interrupted by user")
        print(f"Partial results saved to: {config.output_dir}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during experiment: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Print final verdict
    print(f"\n{'='*60}")
    print("PILOT VERDICT")
    print(f"{'='*60}")

    S = metrics['specialization_index']
    D = metrics['context_divergence']
    pattern = metrics['phase_transition']['pattern']
    pass_at_1 = metrics['summary_stats']['pass_at_1']

    print(f"\nSpecialization Index (S): {S:.3f}")
    print(f"Context Divergence (D): {D:.3f}")
    print(f"Pattern: {pattern}")
    print(f"Pass@1: {pass_at_1:.2%}")

    # Decision criteria from pilot-experiment.md
    if S > 0.1:
        print(f"\n✓ GO: S > 0.1, proceed to full experiments")
    elif S < 0.1 and D > 0.2:
        print(f"\n⚠️  GO WITH CAUTION: S ≈ 0 but D > 0.2, consider modified routing")
    else:
        print(f"\n✗ PIVOT: S ≈ 0 and D ≈ 0, need more tasks, larger model, or different mechanism")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
