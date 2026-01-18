from typing import Dict, Any, Optional
import os
import asyncio
from .interface import ModelInterface

try:
    from llama_cpp import Llama
    HAS_LLAMA_CPP = True
except ImportError:
    HAS_LLAMA_CPP = False

class LlamaCppModel(ModelInterface):
    def __init__(self, model_name: str, model_path: str, n_ctx: int = 4096, n_gpu_layers: int = -1):
        super().__init__(model_name, model_path)
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self._llm: Optional[Llama] = None

    async def load(self) -> None:
        """Load the model into memory using llama.cpp"""
        if not HAS_LLAMA_CPP:
            raise ImportError("llama-cpp-python is not installed. Please install it to use this model.")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")

        # Llama loading is blocking, so ideally we'd run this in an executor 
        # to avoid blocking the async event loop, but for simplicity here we'll keep it direct
        # or wrapping in to_thread is better practice for "async def"
        print(f"Loading model {self.model_name} from {self.model_path}...")
        print(f"Config: n_ctx={self.n_ctx}, n_gpu_layers={self.n_gpu_layers} (CUDA Enabled: {HAS_LLAMA_CPP})")
        
        self._llm = await asyncio.to_thread(
            Llama,
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
            verbose=True # Enable verbose to see Llama.cpp internal GPU usage logs
        )
        self.is_loaded = True
        print(f"Model {self.model_name} loaded successfully.")

    async def unload(self) -> None:
        """Unload the model from memory"""
        if self._llm:
            del self._llm
            self._llm = None
        self.is_loaded = False
        print(f"Model {self.model_name} unloaded.")

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from the model"""
        if not self.is_loaded or not self._llm:
            raise RuntimeError("Model not loaded")

        # Default generation parameters suitable for coding
        max_tokens = kwargs.get("max_tokens", 512)
        temperature = kwargs.get("temperature", 0.2)
        stop = kwargs.get("stop", ["```\n", "User:", "<|endoftext|>"])

        output = await asyncio.to_thread(
            self._llm.create_completion,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop
        )
        
        return output["choices"][0]["text"]

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage information"""
        # Llama.cpp doesn't expose easy python-side memory stats directly 
        # without parsing logs or checking process memory. 
        # Returning basic state for now.
        return {
            "model": self.model_name,
            "loaded": self.is_loaded,
            "backend": "llama.cpp"
        }
