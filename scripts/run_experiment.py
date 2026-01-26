#!/usr/bin/env python3
"""
Run full experiment for swarm-based code generation with multiple seeds.

Features:
- Multi-seed runs for statistical significance
- Full dataset mode (164 HumanEval tasks)
- Parameter sweeps via CLI overrides
- Resumption from interrupted runs
- Config file overrides

Usage:
    # Single run
    python scripts/run_experiment.py --config config/pilot.yaml --num-tasks 164
    
    # Multiple seeds
    python scripts/run_experiment.py --config config/pilot.yaml --seeds 42 123 456
    
    # Parameter sweep
    python scripts/run_experiment.py --config config/pilot.yaml --n-agents 5 --num-tasks 164
    
    # Resume interrupted run
    python scripts/run_experiment.py --config config/pilot.yaml --resume
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
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
        output_dir=exp_config.get('output_dir', 'results/experiment'),
        timeout=exec_config.get('timeout', 5),
        system_prompt=prompts.get('system', 'You are a Python coding assistant.'),
        random_seed=yaml_config.get('random_seed', 42)
    )


def get_completed_tasks(output_dir: Path) -> int:
    """Check how many tasks have been completed in an existing run."""
    task_log = output_dir / "task_log.jsonl"
    if not task_log.exists():
        return 0
    
    count = 0
    with open(task_log, 'r') as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def run_single_experiment(
    model: LlamaCppModel,
    config: ExperimentConfig,
    seed: int,
    resume: bool = False,
    use_full_categories: bool = True
) -> dict:
    """
    Run a single experiment with given seed.
    
    Args:
        model: Loaded LlamaCppModel instance
        config: Experiment configuration
        seed: Random seed for this run
        resume: Whether to resume from previous run
        use_full_categories: Whether to use full 164-task categorization
        
    Returns:
        Metrics dictionary
    """
    # Set seed
    np.random.seed(seed)
    config.random_seed = seed
    
    # Add seed to output directory
    original_output = Path(config.output_dir)
    config.output_dir = str(original_output / f"seed_{seed}")
    
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Check for resume
    start_task = 0
    if resume:
        start_task = get_completed_tasks(output_path)
        if start_task > 0:
            print(f"  Resuming from task {start_task}")
    
    # Create experiment
    experiment = SwarmExperiment(model, config)
    
    # Load tasks with full categorization if requested
    if use_full_categories:
        experiment.load_tasks(use_full_categories=True)
    else:
        experiment.load_tasks()
    
    # Run experiment
    metrics = experiment.run(start_task=start_task)
    
    # Save seed info
    with open(output_path / "run_info.json", "w") as f:
        json.dump({
            "seed": seed,
            "timestamp": datetime.now().isoformat(),
            "n_tasks": config.n_tasks,
            "n_agents": config.n_agents,
            "router_type": config.router_type,
            "model_path": config.model_path
        }, f, indent=2)
    
    return metrics


def aggregate_metrics(all_metrics: list[dict]) -> dict:
    """
    Aggregate metrics across multiple runs for confidence intervals.
    
    Args:
        all_metrics: List of metrics dicts from each run
        
    Returns:
        Aggregated metrics with means and CIs
    """
    if not all_metrics:
        return {}
    
    # Extract key metrics
    S_values = [m['specialization_index'] for m in all_metrics]
    D_values = [m['context_divergence'] for m in all_metrics]
    pass_at_1_values = [m['summary_stats']['pass_at_1'] for m in all_metrics]
    
    def compute_ci(values: list[float], confidence: float = 0.95) -> dict:
        """Compute mean and 95% confidence interval."""
        n = len(values)
        mean = np.mean(values)
        std = np.std(values, ddof=1) if n > 1 else 0
        se = std / np.sqrt(n) if n > 0 else 0
        
        # Use t-distribution for small samples
        from scipy import stats
        t_val = stats.t.ppf((1 + confidence) / 2, df=max(n-1, 1))
        
        return {
            "mean": float(mean),
            "std": float(std),
            "ci_lower": float(mean - t_val * se),
            "ci_upper": float(mean + t_val * se),
            "n": n
        }
    
    return {
        "specialization_index": compute_ci(S_values),
        "context_divergence": compute_ci(D_values),
        "pass_at_1": compute_ci(pass_at_1_values),
        "raw_values": {
            "S": [float(v) for v in S_values],
            "D": [float(v) for v in D_values],
            "pass_at_1": [float(v) for v in pass_at_1_values]
        }
    }


def main():
    parser = argparse.ArgumentParser(description='Run swarm experiment with multiple seeds')
    parser.add_argument('--config', type=str, default='config/pilot.yaml',
                        help='Path to configuration file')
    parser.add_argument('--model', type=str, help='Override model path')
    parser.add_argument('--n-agents', type=int, help='Override number of agents')
    parser.add_argument('--num-tasks', type=int, help='Number of tasks (use 164 for full HumanEval)')
    parser.add_argument('--router', type=str, 
                        choices=['affinity', 'random', 'round_robin', 'greedy'],
                        help='Override router type')
    parser.add_argument('--router-temp', type=float,
                        help='Override router temperature (for affinity router)')
    parser.add_argument('--output-dir', type=str, help='Override output directory')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42],
                        help='Random seeds to run (e.g., --seeds 42 123 456)')
    parser.add_argument('--num-seeds', type=int,
                        help='Generate N random seeds (alternative to --seeds)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume interrupted runs')
    parser.add_argument('--dry-run', action='store_true',
                        help='Validate configuration without running')
    parser.add_argument('--use-full-categories', action='store_true', default=True,
                        help='Use full 164-task categorization file')

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
    if args.num_tasks:
        config.n_tasks = args.num_tasks
    if args.router:
        config.router_type = args.router
    if args.router_temp is not None:
        config.router_temperature = args.router_temp
    if args.output_dir:
        config.output_dir = args.output_dir

    # Determine seeds
    if args.num_seeds:
        seeds = [np.random.randint(1, 100000) for _ in range(args.num_seeds)]
        print(f"Generated seeds: {seeds}")
    else:
        seeds = args.seeds

    # Check model exists
    model_path = Path(config.model_path)
    if not model_path.exists():
        print(f"\n❌ Error: Model file not found: {config.model_path}")
        sys.exit(1)

    # Print configuration
    print(f"\n{'='*60}")
    print("EXPERIMENT CONFIGURATION")
    print(f"{'='*60}")
    print(f"Model: {config.model_path}")
    print(f"Agents: {config.n_agents}")
    print(f"Tasks: {config.n_tasks}")
    print(f"Router: {config.router_type}")
    print(f"Seeds: {seeds}")
    print(f"Output: {config.output_dir}")
    print(f"{'='*60}\n")

    if args.dry_run:
        print("✓ Dry run complete - configuration is valid")
        return

    # Load model once (shared across all runs)
    print("Loading model...")
    model = LlamaCppModel(
        model_name="qwen2.5-coder",
        model_path=str(model_path),
        n_ctx=config.context_length,
        n_gpu_layers=-1
    )
    model.load()
    print("✓ Model loaded\n")

    # Run experiments for each seed
    all_metrics = []
    
    for i, seed in enumerate(seeds):
        print(f"\n{'='*40}")
        print(f"RUN {i+1}/{len(seeds)} - Seed {seed}")
        print(f"{'='*40}")
        
        try:
            metrics = run_single_experiment(
                model, config, seed,
                resume=args.resume,
                use_full_categories=args.use_full_categories
            )
            all_metrics.append(metrics)
            
            # Print run summary
            S = metrics['specialization_index']
            D = metrics['context_divergence']
            p1 = metrics['summary_stats']['pass_at_1']
            print(f"\nRun complete: S={S:.3f}, D={D:.3f}, Pass@1={p1:.2%}")
            
        except KeyboardInterrupt:
            print(f"\n\n⚠️  Interrupted at run {i+1}/{len(seeds)}")
            break
        except Exception as e:
            print(f"\n❌ Error in run {i+1}: {e}")
            import traceback
            traceback.print_exc()

    # Aggregate results if multiple runs
    if len(all_metrics) > 1:
        print(f"\n{'='*60}")
        print("AGGREGATED RESULTS")
        print(f"{'='*60}")
        
        aggregated = aggregate_metrics(all_metrics)
        
        S = aggregated['specialization_index']
        D = aggregated['context_divergence']
        P = aggregated['pass_at_1']
        
        print(f"\nSpecialization Index (S):")
        print(f"  Mean: {S['mean']:.3f} (95% CI: [{S['ci_lower']:.3f}, {S['ci_upper']:.3f}])")
        
        print(f"\nContext Divergence (D):")
        print(f"  Mean: {D['mean']:.3f} (95% CI: [{D['ci_lower']:.3f}, {D['ci_upper']:.3f}])")
        
        print(f"\nPass@1:")
        print(f"  Mean: {P['mean']:.2%} (95% CI: [{P['ci_lower']:.2%}, {P['ci_upper']:.2%}])")
        
        # Save aggregated results
        output_path = Path(config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        with open(output_path / "aggregated_results.json", "w") as f:
            json.dump(aggregated, f, indent=2)
        
        print(f"\n✓ Aggregated results saved to: {output_path}/aggregated_results.json")
    
    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
