from src.agents.base import BaseAgent
from src.agents.types import Task, AgentMessage
from typing import Optional

class CodeGenerator(BaseAgent):
    async def process_task(self, task: Task) -> str:
        """Generate code for the given task"""
        self.current_task = task
        
        # Build prompt for code generation
        prompt = self._build_code_prompt(task)
        
        # Generate code (placeholder for actual model call)
        code = await self._generate_code(prompt)
        
        return code
        
    def _build_code_prompt(self, task: Task) -> str:
        """Build prompt for code generation"""
        context = task.context or {}
        return f"""
Write code for the following task:
{task.description}

Context: {context}

Requirements:
- Write clean, readable code
- Follow Python best practices
- Include necessary imports
- Add docstrings
"""
        
    async def _generate_code(self, prompt: str) -> str:
        """Generate code using the model (placeholder)"""
        # This would call the actual SLM
        if "factorial" in prompt.lower():
            return '''def factorial(n):
    """Calculate factorial of n."""
    if n == 0:
        return 1
    return n * factorial(n-1)
'''
        return f'''def add_numbers(a, b):
    """Add two numbers and return the result."""
    return a + b
'''
        
    async def handle_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle messages related to code generation"""
        if "modify" in message.content.lower():
            # Handle code modification requests
            return await self.send_message(
                message.sender,
                "Code modification request received",
                message.task_id
            )
        return None
