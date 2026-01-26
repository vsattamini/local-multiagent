"""Swarm-based multi-agent system for emergent role specialization."""

from .agent import SwarmAgent
from .router import AffinityRouter, RandomRouter, RoundRobinRouter, GreedyRouter
from .executor import CodeExecutor, HumanEvalExecutor
from .metrics import MetricsEngine, RobustnessMetrics
from .logger import ExperimentLogger
from .types import SwarmTask, TaskType, ExecutionResult, FewShotExample

__all__ = [
    "SwarmAgent",
    "AffinityRouter",
    "RandomRouter",
    "RoundRobinRouter",
    "GreedyRouter",
    "CodeExecutor",
    "HumanEvalExecutor",
    "MetricsEngine",
    "RobustnessMetrics",
    "ExperimentLogger",
    "SwarmTask",
    "TaskType",
    "ExecutionResult",
    "FewShotExample",
]
