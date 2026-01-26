"""Main experiment orchestrator for swarm system."""

from typing import List, Optional, Dict, Type
from dataclasses import dataclass, field
import time

from models.interface import ModelInterface
from .agent import SwarmAgent
from .router import Router, AffinityRouter
from .executor import HumanEvalExecutor
from .metrics import MetricsEngine
from .logger import ExperimentLogger
from .humaneval import HumanEvalLoader
from .types import SwarmTask, TaskType


@dataclass
class ExperimentConfig:
    """Configuration for swarm experiment."""

    # Model settings
    model_path: str = "models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
    context_length: int = 4096
    max_tokens: int = 512

    # Agent settings
    n_agents: int = 3
    max_context_examples: int = 5

    # Router settings
    router_type: str = "affinity"  # affinity, random, round_robin, greedy
    router_temperature: float = 0.5

    # Task settings
    n_tasks: int = 50  # Number of tasks for pilot
    task_types: List[TaskType] = field(
        default_factory=lambda: [TaskType.STRING, TaskType.MATH, TaskType.LIST, TaskType.LOGIC]
    )

    # Experiment settings
    snapshot_interval: int = 10  # Take snapshot every N tasks
    output_dir: str = "results/pilot_001"
    timeout: int = 5  # Code execution timeout in seconds

    # Prompt
    system_prompt: str = (
        "You are a Python coding assistant. Write clean, correct code that solves the problem. "
        "Only output the function implementation, no explanations."
    )

    # Random seed
    random_seed: Optional[int] = 42


class SwarmExperiment:
    """
    Main experiment orchestrator.

    Coordinates agents, router, executor, metrics, and logging.
    """

    def __init__(
        self,
        model: ModelInterface,
        config: ExperimentConfig
    ):
        """
        Initialize experiment.

        Args:
            model: Model interface for generation
            config: Experiment configuration
        """
        self.model = model
        self.config = config

        # Initialize components
        self.agents = [
            SwarmAgent(
                agent_id=i,
                max_context_examples=config.max_context_examples
            )
            for i in range(config.n_agents)
        ]

        self.router = self._create_router()
        self.executor = HumanEvalExecutor(timeout=config.timeout)
        self.metrics = MetricsEngine()
        self.logger = ExperimentLogger(config.output_dir)

        # Load tasks
        self.task_loader = HumanEvalLoader()
        self.tasks: List[SwarmTask] = []

        # Export config
        self.logger.export_config(self._config_to_dict())

    def _create_router(self) -> Router:
        """Create router based on config."""
        from .router import RandomRouter, RoundRobinRouter, GreedyRouter

        if self.config.router_type == "affinity":
            return AffinityRouter(temperature=self.config.router_temperature)
        elif self.config.router_type == "random":
            return RandomRouter()
        elif self.config.router_type == "round_robin":
            return RoundRobinRouter()
        elif self.config.router_type == "greedy":
            return GreedyRouter()
        else:
            raise ValueError(f"Unknown router type: {self.config.router_type}")

    def load_tasks(self, use_full_categories: bool = False) -> None:
        """Load tasks from HumanEval.
        
        Args:
            use_full_categories: If True, load categories from data/humaneval_categories_full.json
        """
        if use_full_categories:
            # Load from full categorization JSON
            self.tasks = self.task_loader.get_tasks_from_json(
                n_tasks=self.config.n_tasks
            )
        else:
            self.tasks = self.task_loader.get_pilot_subset(n_tasks=self.config.n_tasks)
        
        print(f"Loaded {len(self.tasks)} tasks")

        # Print task distribution
        stats = {}
        for task in self.tasks:
            stats[task.task_type.value] = stats.get(task.task_type.value, 0) + 1
        print(f"Task distribution: {stats}")

    def run(self, start_task: int = 0) -> Dict:
        """
        Run full experiment.

        Args:
            start_task: Task index to start from (for resumption)
            
        Returns:
            Final metrics dictionary
        """
        if not self.tasks:
            self.load_tasks()

        print(f"\n{'='*60}")
        print(f"Starting Swarm Experiment")
        print(f"  Agents: {self.config.n_agents}")
        print(f"  Router: {self.config.router_type}")
        print(f"  Tasks: {len(self.tasks)}")
        if start_task > 0:
            print(f"  Resuming from task: {start_task}")
        print(f"  Output: {self.config.output_dir}")
        print(f"{'='*60}\n")

        for i, task in enumerate(self.tasks[start_task:], start=start_task):
            print(f"[{i+1}/{len(self.tasks)}] Processing {task.id} ({task.task_type.value})...")

            # 1. Route task to agent
            agent = self.router.select_agent(self.agents, task.task_type)
            print(f"  → Routed to Agent {agent.agent_id}")

            # 2. Generate solution
            prompt = agent.build_prompt(self.config.system_prompt, task.problem)
            solution = self.model.generate(prompt, max_tokens=self.config.max_tokens)

            # 3. Execute tests
            result = self.executor.execute_humaneval(
                solution,
                task.test_code,
                task.entry_point
            )

            # 4. Update agent state
            if result.success:
                agent.add_success(task.problem, solution, task.task_type)
                print(f"  ✓ Success (exec time: {result.execution_time:.2f}s)")
            else:
                agent.add_failure(task.task_type)
                print(f"  ✗ Failed: {result.error_message[:100] if result.error_message else 'Unknown'}")

            # 5. Update router
            self.router.update(agent, task.task_type, result.success)

            # 6. Log task
            self.logger.log_task(task, agent, result, solution)

            # 7. Periodic snapshot
            if (i + 1) % self.config.snapshot_interval == 0:
                print(f"  📸 Taking snapshot at task {i+1}...")
                snapshot = self.metrics.snapshot(
                    self.agents,
                    self.logger.get_task_log(),
                    i + 1
                )
                self.logger.log_snapshot(snapshot)
                print(f"     S = {snapshot['specialization_index']:.3f}, "
                      f"D = {snapshot['context_divergence']:.3f}")

        # Final analysis
        print(f"\n{'='*60}")
        print("Computing final metrics...")
        final_metrics = self._compute_final_metrics()

        # Log final metrics
        self.logger.log_final_metrics(final_metrics)
        self.logger.finalize()

        print(f"\n{'='*60}")
        print("Experiment Complete!")
        print(f"{'='*60}")
        print(f"Results saved to: {self.config.output_dir}")
        self._print_summary(final_metrics)

        return final_metrics

    def _compute_final_metrics(self) -> Dict:
        """Compute all final metrics."""
        task_log = self.logger.get_task_log()

        # Compute core metrics
        S = self.metrics.specialization_index(task_log)
        D = self.metrics.context_divergence(self.agents)
        F = self.metrics.functional_differentiation(task_log)

        # Significance test
        S_sig = self.metrics.specialization_significance(task_log)

        # Summary stats
        summary = self.logger.get_summary_stats()

        # Get snapshots for timeline
        snapshots = self.logger.load_snapshots()
        S_timeline = [s["specialization_index"] for s in snapshots]

        # Phase transition analysis
        transition = self.metrics.detect_phase_transition(S_timeline)

        # Router affinity matrix (if applicable)
        affinity_matrix = None
        if hasattr(self.router, 'get_affinity_matrix'):
            affinity_matrix = self.router.get_affinity_matrix(self.agents)

        return {
            "specialization_index": S,
            "specialization_significance": S_sig,
            "context_divergence": D,
            "functional_differentiation": F,
            "phase_transition": transition,
            "summary_stats": summary,
            "affinity_matrix": affinity_matrix,
            "agent_states": {
                agent.agent_id: agent.get_state_summary()
                for agent in self.agents
            }
        }

    def _print_summary(self, metrics: Dict) -> None:
        """Print experiment summary."""
        summary = metrics["summary_stats"]

        print(f"\nPerformance:")
        print(f"  Pass@1: {summary['pass_at_1']:.2%}")
        print(f"  Successful: {summary['successful_tasks']}/{summary['total_tasks']}")

        print(f"\nEmergence Metrics:")
        print(f"  Specialization Index (S): {metrics['specialization_index']:.3f}")
        print(f"  Context Divergence (D): {metrics['context_divergence']:.3f}")
        print(f"  S p-value: {metrics['specialization_significance']['p_value']:.4f} "
              f"({'significant' if metrics['specialization_significance']['significant'] else 'not significant'})")

        print(f"\nFunctional Differentiation:")
        F = metrics['functional_differentiation']
        print(f"  Chi-square: {F['chi2']:.2f}")
        print(f"  p-value: {F['p_value']:.4f} ({'significant' if F['significant'] else 'not significant'})")
        print(f"  Effect size (Cramér's V): {F['effect_size']:.3f}")

        print(f"\nPattern: {metrics['phase_transition']['pattern']}")

        if metrics['affinity_matrix']:
            print(f"\nAgent Specialization:")
            for agent_id, affinities in metrics['affinity_matrix'].items():
                print(f"  Agent {agent_id}:")
                for task_type, score in affinities.items():
                    print(f"    {task_type}: {score:.3f}")

    def _config_to_dict(self) -> Dict:
        """Convert config to dictionary for logging."""
        return {
            "model_path": self.config.model_path,
            "context_length": self.config.context_length,
            "max_tokens": self.config.max_tokens,
            "n_agents": self.config.n_agents,
            "max_context_examples": self.config.max_context_examples,
            "router_type": self.config.router_type,
            "router_temperature": self.config.router_temperature,
            "n_tasks": self.config.n_tasks,
            "task_types": [t.value for t in self.config.task_types],
            "snapshot_interval": self.config.snapshot_interval,
            "timeout": self.config.timeout,
            "random_seed": self.config.random_seed
        }
