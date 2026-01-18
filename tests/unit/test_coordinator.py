import pytest
from src.coordination.coordinator import Coordinator
from src.agents.types import Task, AgentType

@pytest.mark.asyncio
async def test_coordinator_task_decomposition():
    coordinator = Coordinator("coordinator", "test-model")
    task = Task(
        id="main-task",
        type=AgentType.COORDINATOR,
        description="Fix bug in authentication module"
    )
    subtasks = await coordinator.decompose_task(task)
    assert len(subtasks) > 0
    assert all(subtask.status.value == "pending" for subtask in subtasks)
