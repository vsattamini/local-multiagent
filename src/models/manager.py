from typing import Dict, Optional, Any
from .interface import ModelInterface

class LocalSLMModel(ModelInterface):
    async def load(self) -> None:
        # Placeholder for actual model loading
        self.is_loaded = True
        
    async def unload(self) -> None:
        # Placeholder for actual model unloading
        self.is_loaded = False
        
    async def generate(self, prompt: str, **kwargs) -> str:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        # Placeholder for actual generation
        return f"Generated response for: {prompt[:50]}..."
        
    def get_memory_usage(self) -> Dict[str, Any]:
        return {"model": self.model_name, "loaded": self.is_loaded}

class ModelManager:
    def __init__(self, max_memory_gb: int = 8):
        self.loaded_models: Dict[str, ModelInterface] = {}
        self.max_memory_gb = max_memory_gb
        
    async def load_model(self, model_name: str, model_path: str) -> ModelInterface:
        """Load a model with memory management"""
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]
            
        # Check memory constraints
        if len(self.loaded_models) > 0:
            await self.unload_oldest_model()
            
        # Create and load model
        if model_path.endswith(".gguf"):
            try:
                from .llama_cpp import LlamaCppModel
                model = LlamaCppModel(model_name, model_path)
            except ImportError:
                print("Warning: llama-cpp-python not found, falling back to mock")
                model = LocalSLMModel(model_name, model_path)
        else:
            model = LocalSLMModel(model_name, model_path)
            
        await model.load()
        self.loaded_models[model_name] = model
        return model
        
    async def unload_model(self, model_name: str) -> None:
        """Unload a specific model"""
        if model_name in self.loaded_models:
            await self.loaded_models[model_name].unload()
            del self.loaded_models[model_name]
            
    async def unload_oldest_model(self) -> None:
        """Unload the oldest loaded model"""
        if self.loaded_models:
            oldest_model = next(iter(self.loaded_models))
            await self.unload_model(oldest_model)
