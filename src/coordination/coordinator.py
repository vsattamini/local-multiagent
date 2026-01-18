from typing import List, Dict, Optional
from src.agents.base import BaseAgent
from src.agents.types import Task, AgentMessage, AgentType, TaskStatus
from .task_decomposer import TaskDecomposer

from src.agents.specialists.code_generator import CodeGenerator
from src.agents.specialists.test_writer import TestWriter
from src.agents.specialists.debugger import Debugger
from src.agents.specialists.reviewer import Reviewer
from src.utils.validator import validate_python_code

class Coordinator(BaseAgent):
    def __init__(self, name: str, model_name: str):
        super().__init__(name, model_name)
        self.active_tasks: Dict[str, Task] = {}
        self.task_queue: List[Task] = []
        self.decomposer = TaskDecomposer()
        
    async def process_task(self, task: Task) -> str:
        """Process a task by decomposing and coordinating"""
        self.current_task = task
        self.active_tasks[task.id] = task
        
        # Decompose task into subtasks
        subtasks = await self.decompose_task(task)
        
        # Add subtasks to queue
        self.task_queue.extend(subtasks)
        
        # Process subtasks sequentially (for memory constraints)
        results = []
        for subtask in subtasks:
            result = await self.execute_subtask(subtask)
            results.append(result)
            
        # Synthesize final result
        final_result = await self.synthesize_results(results)
        task.result = final_result
        task.status = TaskStatus.COMPLETED
        
        return final_result
        
    async def decompose_task(self, task: Task) -> List[Task]:
        """Decompose a task into specialist subtasks"""
        if task.type == AgentType.COORDINATOR:
            return await self.decomposer.decompose_coding_task(task)
        else:
            return [task]
            
    async def execute_subtask(self, subtask: Task) -> str:
        """Execute a single subtask by dispatching to specialist agent"""
        subtask.status = TaskStatus.IN_PROGRESS
        
        agent = self._get_agent_for_task(subtask)
        if agent:
             # Validation Loop
             max_retries = 2
             attempt = 0
             original_description = subtask.description
             
             while attempt <= max_retries:
                 # Process task
                 result = await agent.process_task(subtask)
                 
                 # Only validate for code-generating agents
                 if subtask.type in [AgentType.CODE_GENERATOR, AgentType.TEST_WRITER]:
                     validation = validate_python_code(result)
                     if validation["valid"]:
                         subtask.result = result
                         subtask.status = TaskStatus.COMPLETED
                         return result
                     else:
                         # Invalid code. Retry if possible
                         attempt += 1
                         if attempt <= max_retries:
                             print(f"Validation failed (Attempt {attempt}): {validation['error']}. Retrying...")
                             # Update task description with feedback
                             subtask.description = (
                                 f"{original_description}\n\n"
                                 f"PREVIOUS ATTEMPT FAILED WITH SYNTAX ERROR:\n"
                                 f"{validation['error']}\n"
                                 f"Please fix the syntax and regenerate the code."
                             )
                         else:
                             # Retries exhausted
                             subtask.status = TaskStatus.FAILED
                             return f"Generation failed after {max_retries} retries. Last error: {validation['error']}\nCode:\n{result}"
                 else:
                     # Non-code agents don't need syntax validation
                     subtask.result = result
                     subtask.status = TaskStatus.COMPLETED
                     return result
             
        # Fallback
        subtask.status = TaskStatus.FAILED
        return f"Error: No agent found for task type {subtask.type}"

    def _get_agent_for_task(self, task: Task) -> Optional[BaseAgent]:
        if task.type == AgentType.CODE_GENERATOR:
            return CodeGenerator("code-gen", self.model_name)
        elif task.type == AgentType.TEST_WRITER:
            return TestWriter("test-writer", self.model_name)
        elif task.type == AgentType.DEBUGGER:
            return Debugger("debugger", self.model_name)
        elif task.type == AgentType.REVIEWER:
            return Reviewer("reviewer", self.model_name)
        return None
        
    async def synthesize_results(self, results: List[str]) -> str:
        """Synthesize subtask results into final output"""
        return "\n".join(results)
        
    async def handle_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle messages from other agents"""
        if message.task_id and message.task_id in self.active_tasks:
            task = self.active_tasks[message.task_id]
            # Update task status based on message
            return await self.send_message(
                message.sender,
                f"Task {message.task_id} updated",
                message.task_id
            )
        return None
