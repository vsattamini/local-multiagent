from src.agents.base import BaseAgent
from src.agents.types import Task, AgentMessage
from typing import Optional

class Debugger(BaseAgent):
    async def process_task(self, task: Task) -> str:
        """Debug and fix issues in code"""
        self.current_task = task
        
        prompt = self._build_debug_prompt(task)
        fix = await self._generate_fix(prompt)
        
        return fix
        
    def _build_debug_prompt(self, task: Task) -> str:
        """Build prompt for debugging"""
        context = task.context or {}
        return f"""
Debug and fix the following issue:
{task.description}

Context: {context}

Requirements:
- Identify the root cause
- Provide a clear fix
- Explain the reasoning
- Suggest prevention strategies
"""
        
    async def _generate_fix(self, prompt: str) -> str:
        """Generate fix using the model (placeholder)"""
        return "Issue identified and fixed with proper error handling."
        
    async def handle_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle debugging-related messages"""
        return None
