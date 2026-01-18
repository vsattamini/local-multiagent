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
        
        return f"""
<|im_start|>system
You are an expert QA Engineer specialized in writing robust Python tests using pytest.
Your goal is to write a concise, strictly typed, and correct test suite for the provided code/task.

Constraints:
1. Output ONLY the Python code for the tests.
2. Do NOT add any conversational text, explanations, or "Here is the code".
3. Wrap code in ```python ... ``` blocks.
4. Keep the tests focused and atomic.
5. Limit the output to the most critical test cases (Success, Edge Case, Failure).
<|im_end|>
<|im_start|>user
Write comprehensive tests for:
{task.description}

Context: {context}
<|im_end|>
<|im_start|>assistant
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
