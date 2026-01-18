import logging
import asyncio
from typing import List, Dict, Any
from src.evaluation.swebench import TaskInstance
from src.agents.types import Task, AgentType, TaskStatus
from src.coordination.coordinator import Coordinator

logger = logging.getLogger(__name__)

class BenchmarkRunner:
    def __init__(self, coordinator: Coordinator, output_dir: str = "evaluation_results"):
        self.coordinator = coordinator
        self.output_dir = output_dir

    async def run_task(self, instance: TaskInstance) -> Dict[str, Any]:
        """
        Run a single SWE-bench task through the coordinator.
        """
        logger.info(f"Starting task {instance.instance_id}")
        
        # Create internal Task object
        task = Task(
            id=instance.instance_id,
            type=AgentType.COORDINATOR,
            description=instance.problem_statement,
            context={
                "repo": instance.repo,
                "base_commit": instance.base_commit,
                "environment_setup_commit": instance.environment_setup_commit,
                "version": instance.version
            }
        )

        try:
            # Execute
            result = await self.coordinator.process_task(task)
            
            # The result from coordinator currently is a string (final output)
            # In a real scenario, we expect this string to be the git patch or similar.
            
            return {
                "instance_id": instance.instance_id,
                "patch": result,
                "status": "success"
            }
        except Exception as e:
            logger.error(f"Task {instance.instance_id} failed: {e}", exc_info=True)
            return {
                "instance_id": instance.instance_id,
                "error": str(e),
                "status": "failed"
            }

    async def run_batch(self, tasks: List[TaskInstance]) -> List[Dict[str, Any]]:
        """Run a batch of tasks sequentially."""
        results = []
        for task_instance in tasks:
            result = await self.run_task(task_instance)
            results.append(result)
        return results
