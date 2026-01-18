# tests/unit/test_agent_base.py
import pytest
from src.agents.base import BaseAgent
from src.agents.types import AgentMessage, Task

class TestAgent(BaseAgent):
    async def process_task(self, task: Task) -> str:
        return "processed"
    
    async def handle_message(self, message: AgentMessage) -> None:
        return None

def test_base_agent_initialization():
    agent = TestAgent("test-agent", "test-model")
    assert agent.name == "test-agent"
    assert agent.model_name == "test-model"

def test_agent_message_creation():
    message = AgentMessage(
        sender="test-agent",
        content="test message",
        task_id="task-123"
    )
    assert message.sender == "test-agent"
    assert message.task_id == "task-123"

