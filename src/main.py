import asyncio
from typing import Optional, Dict, Any, List
import logging

from .coordination.coordinator import Coordinator
from .models.manager import ModelManager
from .utils.memory import MemoryManager
from .utils.context import ContextManager
from .evaluation.runner import BenchmarkRunner
from .evaluation.swebench import SWEBenchLoader

logger = logging.getLogger(__name__)

class MultiAgentSystem:
    def __init__(self, max_memory_gb: int = 8, max_tokens: int = 4000):
        self.memory_manager = MemoryManager(max_memory_gb)
        self.context_manager = ContextManager(max_tokens)
        self.model_manager = ModelManager(max_memory_gb)
        self.coordinator = Coordinator("main-coordinator", "default-model")
        self.evaluation_runner = BenchmarkRunner(self.coordinator)
        
    async def initialize(self) -> None:
        """Initialize the system"""
        # Load default model
        # Placeholder: await self.model_manager.load_model("default-model", "path/to/model")
        logger.info("System initialized")
        
    async def process_task(self, task_description: str) -> str:
        """Process a single task through the multi-agent system"""
        from .agents.types import Task, AgentType
        
        task = Task(
            id="main-task",
            type=AgentType.COORDINATOR,
            description=task_description
        )
        
        result = await self.coordinator.process_task(task)
        return result
        
    async def run_evaluation(self, lite: bool = True) -> dict:
        """Run SWEBench evaluation"""
        loader = SWEBenchLoader()
        # For this implementation, lite means a small subset
        split = "test"
        tasks = loader.load_tasks(split=split)
        
        if lite:
             tasks = tasks[:2] # Limit to 2 tasks for lite/quick run
        
        results = await self.evaluation_runner.run_batch(tasks)
        
        # Calculate basic metrics (success rate)
        completed = len([r for r in results if r.get("status") == "success"])
        avg_score = completed / len(results) if results else 0.0
        
        return {
            "total_tasks": len(tasks),
            "completed_tasks": completed,
            "average_score": avg_score,
            "results": results
        }
        
    async def shutdown(self) -> None:
        """Shutdown the system and cleanup resources"""
        # Unload all models
        for model_name in list(self.model_manager.loaded_models.keys()):
            await self.model_manager.unload_model(model_name)
        logger.info("System shutdown complete")
