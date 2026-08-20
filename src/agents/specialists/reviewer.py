from src.agents.base import BaseAgent
from src.agents.types import Task, AgentMessage
from typing import Optional

class Reviewer(BaseAgent):
    async def process_task(self, task: Task) -> str:
        """Review code and provide feedback"""
        self.current_task = task
        
        prompt = self._build_review_prompt(task)
        review = await self._generate_review(prompt)
        
        return review
        
    def _build_review_prompt(self, task: Task) -> str:
        """Build prompt for code review"""
        context = task.context or {}
        return f"""
Review the following code:
{task.description}

Context: {context}

Review criteria:
- Code quality and readability
- Adherence to best practices
- Performance considerations
- Security implications
- Test coverage
"""
        
    async def _generate_review(self, prompt: str) -> str:
        """Generate review using the model (placeholder)"""
        return "Code review completed. All checks passed."
        
    async def handle_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle review-related messages"""
        return None
