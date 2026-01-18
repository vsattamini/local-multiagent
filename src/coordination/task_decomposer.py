from typing import List
from src.agents.types import Task, AgentType

class TaskDecomposer:
    @staticmethod
    async def decompose_coding_task(task: Task) -> List[Task]:
        """Decompose a coding task into specialist subtasks"""
        subtasks = []
        
        # Code generation subtask
        code_task = Task(
            id=f"{task.id}-code",
            type=AgentType.CODE_GENERATOR,
            description=f"Write code for: {task.description}",
            context={"parent_task": task.id}
        )
        subtasks.append(code_task)
        
        # Test writing subtask
        test_task = Task(
            id=f"{task.id}-test",
            type=AgentType.TEST_WRITER,
            description=f"Write tests for: {task.description}",
            context={"parent_task": task.id}
        )
        subtasks.append(test_task)
        
        # Code review subtask
        review_task = Task(
            id=f"{task.id}-review",
            type=AgentType.REVIEWER,
            description=f"Review code and tests for: {task.description}",
            context={"parent_task": task.id}
        )
        subtasks.append(review_task)
        
        return subtasks
