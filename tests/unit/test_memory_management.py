import pytest
from src.utils.memory import MemoryManager
from src.utils.context import ContextManager

@pytest.mark.asyncio
async def test_memory_manager():
    manager = MemoryManager(max_memory_gb=8)
    # The get_available_memory function returns usage, not simple boolean, but we check if it runs
    assert manager.get_available_memory() >= 0
    
    result = await manager.allocate_memory("test-model", 2)
    assert result is True
    assert manager.get_allocated_memory("test-model") == 2

@pytest.mark.asyncio
async def test_context_manager():
    manager = ContextManager(max_tokens=4000)
    await manager.add_context("task-1", "Initial context")
    context = await manager.get_context("task-1")
    assert "Initial context" in context
