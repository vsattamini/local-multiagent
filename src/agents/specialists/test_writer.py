from src.agents.base import BaseAgent
from src.agents.types import Task, AgentMessage
from typing import Optional

class TestWriter(BaseAgent):
    async def process_task(self, task: Task) -> str:
        """Write tests for the given task"""
        self.current_task = task
        
        prompt = self._build_test_prompt(task)
        tests = await self._generate_tests(prompt)
        
        return tests
        
    def _build_test_prompt(self, task: Task) -> str:
        """Build prompt for test generation"""
        context = task.context or {}
        return f"""
Write comprehensive tests for:
{task.description}

Context: {context}

Requirements:
- Use pytest framework
- Include edge cases
- Test both success and failure scenarios
- Follow test naming conventions
"""
        
    async def _generate_tests(self, prompt: str) -> str:
        """Generate tests using the model (placeholder)"""
        return f'''import pytest

def test_add_numbers_positive():
    """Test adding two positive numbers."""
    result = add_numbers(2, 3)
    assert result == 5

def test_add_numbers_negative():
    """Test adding two negative numbers."""
    result = add_numbers(-2, -3)
    assert result == -5

def test_add_numbers_zero():
    """Test adding with zero."""
    result = add_numbers(5, 0)
    assert result == 5
'''
        
    async def handle_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle messages related to test writing"""
        return None
