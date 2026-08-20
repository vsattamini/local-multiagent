"""Swarm agent with context-based differentiation."""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
from .types import FewShotExample, TaskType


@dataclass
class SwarmAgent:
    """
    A swarm agent is just a context window + metadata.
    No separate model - all agents share the same frozen model.
    """

    agent_id: int
    max_context_examples: int = 5
    context_buffer: List[FewShotExample] = field(default_factory=list)
    task_history: Dict[TaskType, List[bool]] = field(default_factory=dict)
    # Optional per-agent system prompt (persona positive-control condition).
    # None => use the shared system prompt passed to build_prompt.
    system_prompt_override: Optional[str] = None

    def add_success(self, problem: str, solution: str, task_type: TaskType) -> None:
        """Add successful example to context buffer."""
        example = FewShotExample(
            problem=problem,
            solution=solution,
            task_type=task_type
        )

        self.context_buffer.append(example)

        # FIFO: Keep only most recent examples
        if len(self.context_buffer) > self.max_context_examples:
            self.context_buffer.pop(0)

        # Track history
        if task_type not in self.task_history:
            self.task_history[task_type] = []
        self.task_history[task_type].append(True)

    def add_failure(self, task_type: TaskType) -> None:
        """Record failure (no context addition)."""
        if task_type not in self.task_history:
            self.task_history[task_type] = []
        self.task_history[task_type].append(False)

    def build_prompt(
        self,
        system_prompt: str,
        new_problem: str,
        task_type: Optional[TaskType] = None,
        type_filtered: bool = False,
        max_show: Optional[int] = None,
    ) -> str:
        """
        Construct full prompt with few-shot examples from context.

        Args:
            system_prompt: Base instruction (overridden by self.system_prompt_override
                if set — the persona positive-control condition).
            new_problem: New problem to solve
            task_type: Type of the current task (used for type-filtered retrieval)
            type_filtered: If True, prefer accumulated examples of the SAME task_type
                (the mechanism for coupling accumulated experience to per-type skill).
            max_show: Cap on number of examples shown (None => all selected).

        Returns:
            Complete prompt with few-shot examples

        NOTE: with defaults (type_filtered=False, max_show=None, no override) this
        reproduces the original behaviour exactly.
        """
        base = self.system_prompt_override or system_prompt
        prompt = base + "\n\n"

        # Select few-shot examples
        if type_filtered and task_type is not None:
            same = [ex for ex in self.context_buffer if ex.task_type == task_type]
            examples = same if same else list(self.context_buffer)
        else:
            examples = list(self.context_buffer)

        if max_show is not None:
            examples = examples[-max_show:]

        for example in examples:
            prompt += f"### Problem:\n{example.problem}\n\n"
            prompt += f"### Solution:\n```python\n{example.solution}\n```\n\n"

        # Add new problem
        prompt += f"### Problem:\n{new_problem}\n\n### Solution:\n```python\n"

        return prompt

    def success_rate(self, task_type: TaskType) -> float:
        """
        Get success rate for a specific task type.

        Args:
            task_type: Type of task

        Returns:
            Success rate [0, 1], or 0.5 (prior) if no history
        """
        history = self.task_history.get(task_type, [])
        if not history:
            return 0.5  # Uniform prior
        return sum(history) / len(history)

    def overall_success_rate(self) -> float:
        """Get overall success rate across all task types."""
        all_results = []
        for results in self.task_history.values():
            all_results.extend(results)

        if not all_results:
            return 0.0
        return sum(all_results) / len(all_results)

    def total_tasks(self) -> int:
        """Get total number of tasks attempted."""
        return sum(len(results) for results in self.task_history.values())

    def successful_tasks(self) -> int:
        """Get total number of successful tasks."""
        return sum(sum(results) for results in self.task_history.values())

    def get_state_summary(self) -> Dict:
        """Get summary of agent state for logging."""
        return {
            "agent_id": self.agent_id,
            "total_tasks": self.total_tasks(),
            "successful_tasks": self.successful_tasks(),
            "success_rate": self.overall_success_rate(),
            "context_size": len(self.context_buffer),
            "task_type_performance": {
                task_type.value: self.success_rate(task_type)
                for task_type in TaskType
            }
        }
