import pytest
from src.agents.specialists.code_generator import CodeGenerator
from src.agents.specialists.test_writer import TestWriter
from src.agents.types import Task, AgentType

@pytest.mark.asyncio
async def test_code_generator():
    agent = CodeGenerator("code-gen", "test-model")
    task = Task(
        id="test-task",
        type=AgentType.CODE_GENERATOR,
        description="Write a function to add two numbers"
    )
    result = await agent.process_task(task)
    assert "def" in result
    assert "add" in result.lower()

@pytest.mark.asyncio
async def test_test_writer():
    agent = TestWriter("test-writer", "test-model")
    task = Task(
        id="test-task",
        type=AgentType.TEST_WRITER,
        description="Write tests for add function"
    )
    result = await agent.process_task(task)
    assert "test" in result.lower()
    assert "assert" in result.lower()
