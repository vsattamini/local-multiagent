"""Type definitions for swarm-based multi-agent system."""

from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Tuple
from enum import Enum


class TaskType(str, Enum):
    """Task categories for HumanEval problems."""
    STRING = "string"
    MATH = "math"
    LIST = "list"
    LOGIC = "logic"


class SwarmTask(BaseModel):
    """A coding task for the swarm to solve."""
    id: str
    task_type: TaskType
    problem: str
    test_code: str
    entry_point: str
    canonical_solution: Optional[str] = None


class ExecutionResult(BaseModel):
    """Result of code execution."""
    success: bool
    stdout: str = ""
    stderr: str = ""
    execution_time: float = 0.0
    error_message: Optional[str] = None


class FewShotExample(BaseModel):
    """A few-shot example stored in agent context."""
    problem: str
    solution: str
    task_type: TaskType


class AgentState(BaseModel):
    """Complete state of a swarm agent."""
    agent_id: int
    context_buffer: List[FewShotExample] = Field(default_factory=list)
    task_history: Dict[TaskType, List[bool]] = Field(default_factory=dict)
    total_tasks: int = 0
    successful_tasks: int = 0
