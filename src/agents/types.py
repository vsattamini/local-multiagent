from pydantic import BaseModel
from typing import Optional, Dict, Any
from enum import Enum

class AgentType(str, Enum):
    COORDINATOR = "coordinator"
    CODE_GENERATOR = "code_generator"
    TEST_WRITER = "test_writer"
    DEBUGGER = "debugger"
    REVIEWER = "reviewer"

class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class Task(BaseModel):
    id: str
    type: AgentType
    description: str
    context: Optional[Dict[str, Any]] = None
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[str] = None

class AgentMessage(BaseModel):
    sender: str
    receiver: Optional[str] = None
    content: str
    task_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
