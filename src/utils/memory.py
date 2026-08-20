import psutil
from typing import Dict, Optional

class MemoryManager:
    def __init__(self, max_memory_gb: int = 8):
        self.max_memory_gb = max_memory_gb
        self.allocated_memory: Dict[str, float] = {}
        
    def get_total_memory(self) -> float:
        """Get total system memory in GB"""
        return psutil.virtual_memory().total / (1024**3)
        
    def get_available_memory(self) -> float:
        """Get available memory in GB"""
        return psutil.virtual_memory().available / (1024**3)
        
    def get_used_memory(self) -> float:
        """Get used memory in GB"""
        return psutil.virtual_memory().used / (1024**3)
        
    async def allocate_memory(self, model_name: str, memory_gb: float) -> bool:
        """Allocate memory for a model"""
        available = self.get_available_memory()
        
        if memory_gb > available:
            return False
            
        if model_name in self.allocated_memory:
            self.allocated_memory[model_name] += memory_gb
        else:
            self.allocated_memory[model_name] = memory_gb
            
        return True
        
    async def deallocate_memory(self, model_name: str) -> None:
        """Deallocate memory for a model"""
        if model_name in self.allocated_memory:
            del self.allocated_memory[model_name]
            
    def get_allocated_memory(self, model_name: str) -> Optional[float]:
        """Get allocated memory for a specific model"""
        return self.allocated_memory.get(model_name)
        
    def get_total_allocated(self) -> float:
        """Get total allocated memory"""
        return sum(self.allocated_memory.values())
