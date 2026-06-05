from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class ModelInterface(ABC):
    def __init__(self, model_name: str, model_path: str):
        self.model_name = model_name
        self.model_path = model_path
        self.is_loaded = False
        
    @abstractmethod
    def load(self) -> None:
        """Load the model into memory"""
        pass

    @abstractmethod
    def unload(self) -> None:
        """Unload the model from memory"""
        pass

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from the model"""
        pass
        
    @abstractmethod
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage information"""
        pass
