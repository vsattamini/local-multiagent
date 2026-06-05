"""Task routing strategies for swarm agents."""

import numpy as np
from abc import ABC, abstractmethod
from typing import List
from .agent import SwarmAgent
from .types import TaskType


class Router(ABC):
    """Base class for task routing strategies."""

    @abstractmethod
    def select_agent(self, agents: List[SwarmAgent], task_type: TaskType) -> SwarmAgent:
        """Select an agent for the given task type."""
        pass

    @abstractmethod
    def update(self, agent: SwarmAgent, task_type: TaskType, success: bool) -> None:
        """Update routing strategy based on task outcome."""
        pass


class RandomRouter(Router):
    """Baseline router that selects agents uniformly at random."""

    def select_agent(self, agents: List[SwarmAgent], task_type: TaskType) -> SwarmAgent:
        """Select random agent."""
        idx = np.random.choice(len(agents))
        return agents[idx]

    def update(self, agent: SwarmAgent, task_type: TaskType, success: bool) -> None:
        """No-op for random router."""
        pass


class AffinityRouter(Router):
    """
    Affinity-based router using softmax over success rates.

    Routes tasks to agents based on their historical success rates
    for each task type, using softmax to balance exploration/exploitation.
    """

    def __init__(self, temperature: float = 0.5):
        """
        Initialize affinity router.

        Args:
            temperature: Controls exploration vs exploitation.
                        Higher = more exploration (more uniform)
                        Lower = more exploitation (greedier)
        """
        self.temperature = temperature

    def select_agent(self, agents: List[SwarmAgent], task_type: TaskType) -> SwarmAgent:
        """
        Select agent using softmax over success rates.

        Args:
            agents: List of available agents
            task_type: Type of task to route

        Returns:
            Selected agent
        """
        # Get success rates for this task type
        scores = np.array([agent.success_rate(task_type) for agent in agents])

        # Apply softmax with temperature
        probs = self._softmax(scores)

        # Sample from distribution
        idx = np.random.choice(len(agents), p=probs)
        return agents[idx]

    def update(self, agent: SwarmAgent, task_type: TaskType, success: bool) -> None:
        """
        Update is handled by agent itself in add_success/add_failure.
        This method is here for interface compatibility.
        """
        pass

    def _softmax(self, scores: np.ndarray) -> np.ndarray:
        """
        Compute softmax with temperature scaling.

        Args:
            scores: Array of scores

        Returns:
            Probability distribution
        """
        # Temperature scaling
        scaled_scores = scores / self.temperature

        # Numerical stability: subtract max
        exp_scores = np.exp(scaled_scores - np.max(scaled_scores))

        # Normalize
        return exp_scores / exp_scores.sum()

    def get_affinity_matrix(self, agents: List[SwarmAgent]) -> dict:
        """
        Get current affinity matrix for all agents and task types.

        Returns:
            Dictionary mapping agent_id -> task_type -> success_rate
        """
        return {
            agent.agent_id: {
                task_type.value: agent.success_rate(task_type)
                for task_type in TaskType
            }
            for agent in agents
        }


class RoundRobinRouter(Router):
    """Router that cycles through agents in order (control baseline)."""

    def __init__(self):
        self.current_idx = 0

    def select_agent(self, agents: List[SwarmAgent], task_type: TaskType) -> SwarmAgent:
        """Select next agent in round-robin order."""
        agent = agents[self.current_idx]
        self.current_idx = (self.current_idx + 1) % len(agents)
        return agent

    def update(self, agent: SwarmAgent, task_type: TaskType, success: bool) -> None:
        """No-op for round-robin router."""
        pass


class GreedyRouter(Router):
    """Router that always selects agent with highest success rate (pure exploitation)."""

    def select_agent(self, agents: List[SwarmAgent], task_type: TaskType) -> SwarmAgent:
        """Select agent with highest success rate for task type."""
        scores = [agent.success_rate(task_type) for agent in agents]
        best_idx = np.argmax(scores)
        return agents[best_idx]

    def update(self, agent: SwarmAgent, task_type: TaskType, success: bool) -> None:
        """No-op for greedy router."""
        pass
