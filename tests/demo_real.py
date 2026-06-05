
import asyncio
import sys
import os
import argparse

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.main import MultiAgentSystem
from src.models.llama_cpp import LlamaCppModel
from src.agents.specialists.code_generator import CodeGenerator
from src.agents.specialists.test_writer import TestWriter
from src.agents.specialists.reviewer import Reviewer

# Global model reference for the patches
SHARED_MODEL = None

# ---------------------------------------------------------
# Dynamic Patches to inject Real Model usage
# ---------------------------------------------------------
async def patched_generate_code(self, prompt: str) -> str:
    print(f"\n[CodeGenerator] Prompting model ({len(prompt)} chars)...")
    if not SHARED_MODEL:
        return "Error: Model not loaded"
    return await SHARED_MODEL.generate(prompt, max_tokens=256)

async def patched_generate_tests(self, prompt: str) -> str:
    print(f"\n[TestWriter] Prompting model ({len(prompt)} chars)...")
    if not SHARED_MODEL:
        return "Error: Model not loaded"
    return await SHARED_MODEL.generate(prompt, max_tokens=256)

async def patched_generate_review(self, prompt: str) -> str:
    print(f"\n[Reviewer] Prompting model ({len(prompt)} chars)...")
    if not SHARED_MODEL:
        return "Error: Model not loaded"
    return await SHARED_MODEL.generate(prompt, max_tokens=256)

# ---------------------------------------------------------

async def main():
    global SHARED_MODEL
    
    parser = argparse.ArgumentParser(description="Run Multi-Agent System with Real Local Model")
    parser.add_argument("--model_path", required=True, help="Path to .gguf model file")
    parser.add_argument("--n_gpu_layers", type=int, default=-1, help="Number of layers to offload to GPU (-1 for all)")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        return

    print("=== Demo: Real Model Integration ===")
    
    # 1. Load the Model manually first
    try:
        SHARED_MODEL = LlamaCppModel("real-model", args.model_path, n_gpu_layers=args.n_gpu_layers)
        await SHARED_MODEL.load()
    except ImportError:
        print("Error: llama-cpp-python not installed. Cannot run real demo.")
        print("Install with: pip install llama-cpp-python")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Patch the agents to use this model
    print("Patching agents to use loaded model...")
    CodeGenerator._generate_code = patched_generate_code
    TestWriter._generate_tests = patched_generate_tests
    Reviewer._generate_review = patched_generate_review

    # 3. Initialize System
    system = MultiAgentSystem(max_memory_gb=8)
    await system.initialize()
    
    task = "Write a Python function called 'is_palindrome' that checks if a string is a palindrome."
    print(f"\nProcessing Task: {task}\n")
    
    try:
        result = await system.process_task(task)
        print("\n=== Final Result ===")
        print(result)
    finally:
        # Cleanup
        await SHARED_MODEL.unload()
        await system.shutdown()
        print("\n=== Demo Complete ===")

if __name__ == "__main__":
    asyncio.run(main())
