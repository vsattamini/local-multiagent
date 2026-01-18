import pytest
from src.main import MultiAgentSystem

@pytest.mark.asyncio
async def test_full_system_workflow():
    """Test complete system workflow"""
    system = MultiAgentSystem()
    await system.initialize()
    
    # Process a task
    result = await system.process_task("Write a function to calculate factorial")
    
    # Verify result contains expected elements
    assert "def" in result
    assert "factorial" in result.lower()
    
    # Run evaluation
    eval_results = await system.run_evaluation(lite=True)
    assert "average_score" in eval_results
    
    await system.shutdown()

@pytest.mark.asyncio
async def test_memory_constraints():
    """Test system respects memory constraints"""
    system = MultiAgentSystem(max_memory_gb=4)
    await system.initialize()
    
    # Verify memory manager is working
    available = system.memory_manager.get_available_memory()
    assert available > 0
    
    await system.shutdown()
