from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class ContextItem:
    task_id: str
    content: str
    tokens: int
    timestamp: float
    priority: int = 1

class ContextManager:
    def __init__(self, max_tokens: int = 4000):
        self.max_tokens = max_tokens
        self.context_items: Dict[str, List[ContextItem]] = {}
        self.current_tokens = 0
        
    async def add_context(self, task_id: str, content: str, priority: int = 1) -> bool:
        """Add context item if within token limits"""
        tokens = self._estimate_tokens(content)
        
        if self.current_tokens + tokens > self.max_tokens:
            # Try to prune old context
            await self._prune_context(tokens)
            
        if self.current_tokens + tokens > self.max_tokens:
            return False
            
        item = ContextItem(
            task_id=task_id,
            content=content,
            tokens=tokens,
            timestamp=self._get_timestamp(),
            priority=priority
        )
        
        if task_id not in self.context_items:
            self.context_items[task_id] = []
            
        self.context_items[task_id].append(item)
        self.current_tokens += tokens
        
        return True
        
    async def get_context(self, task_id: str) -> str:
        """Get context for a specific task"""
        if task_id not in self.context_items:
            return ""
            
        items = sorted(self.context_items[task_id], key=lambda x: x.timestamp)
        return "\n".join(item.content for item in items)
        
    async def _prune_context(self, needed_tokens: int) -> None:
        """Prune old context to make room"""
        # Sort by priority and timestamp
        all_items = []
        for task_items in self.context_items.values():
            all_items.extend(task_items)
            
        all_items.sort(key=lambda x: (x.priority, x.timestamp))
        
        # Remove oldest items until we have enough space
        for item in all_items:
            if self.current_tokens - item.tokens >= needed_tokens:
                self.context_items[item.task_id].remove(item)
                self.current_tokens -= item.tokens
            else:
                break
                
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (1 token ≈ 4 characters)"""
        return len(text) // 4
        
    def _get_timestamp(self) -> float:
        """Get current timestamp"""
        import time
        return time.time()
