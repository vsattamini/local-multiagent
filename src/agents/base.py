from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from .types import AgentMessage, Task, TaskStatus

class BaseAgent(ABC):
    def __init__(self, name: str, model_name: str):
        self.name = name
        self.model_name = model_name
        self.current_task: Optional[Task] = None
        
    @abstractmethod
    async def process_task(self, task: Task) -> str:
        """Process a task and return the result"""
        pass
        
    @abstractmethod
    async def handle_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle incoming message and optionally return a response"""
        pass
        
    async def send_message(self, receiver: str, content: str, task_id: Optional[str] = None) -> AgentMessage:
        """Create a message to send to another agent"""
        return AgentMessage(
            sender=self.name,
            receiver=receiver,
            content=content,
            task_id=task_id
        )
