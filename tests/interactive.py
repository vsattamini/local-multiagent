
import asyncio
import sys
import os
import argparse
import time
import logging
from typing import Optional

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.main import MultiAgentSystem
from src.models.llama_cpp import LlamaCppModel
from src.agents.specialists.code_generator import CodeGenerator
from src.agents.specialists.test_writer import TestWriter
from src.agents.specialists.reviewer import Reviewer

# Configure Logging to console only
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("Interactive")
# Silence other loggers
logging.getLogger("src").setLevel(logging.WARNING)

SHARED_MODEL = None

async def patched_generate_code(self, prompt: str) -> str:
    print(f"\n[CodeGenerator] Generating...", end="", flush=True)
    start = time.perf_counter()
    response = await asyncio.to_thread(
        SHARED_MODEL._llm.create_completion,
        prompt=prompt,
        max_tokens=1024,
        temperature=0.2,
        stop=["```\n", "User:", "<|im_end|>"]
    )
    duration = time.perf_counter() - start
    text = response["choices"][0]["text"]
    print(f" Done ({duration:.1f}s)")
    return text

async def patched_generate_tests(self, prompt: str) -> str:
    print(f"\n[TestWriter] Generating tests...", end="", flush=True)
    start = time.perf_counter()
    response = await asyncio.to_thread(
        SHARED_MODEL._llm.create_completion,
        prompt=prompt,
        max_tokens=1024,
        temperature=0.2,
        stop=["```\n", "User:", "<|im_end|>"]
    )
    duration = time.perf_counter() - start
    text = response["choices"][0]["text"]
    print(f" Done ({duration:.1f}s)")
    return text

async def patched_generate_review(self, prompt: str) -> str:
    print(f"\n[Reviewer] Reviewing...", end="", flush=True)
    start = time.perf_counter()
    response = await asyncio.to_thread(
        SHARED_MODEL._llm.create_completion,
        prompt=prompt,
        max_tokens=1024,
        temperature=0.2,
        stop=["```\n", "User:", "<|im_end|>"]
    )
    duration = time.perf_counter() - start
    text = response["choices"][0]["text"]
    print(f" Done ({duration:.1f}s)")
    return text

async def interactive_loop(system):
    print("\n" + "="*50)
    print("🤖 Local Multi-Agent System (Interactive Mode)")
    print("Context: 8GB VRAM | Qwen2.5-Coder-7B | GPU Accelerated")
    print("Type 'exit', 'quit', or Ctrl+C to stop.")
    print("="*50 + "\n")

    history = []
    
    while True:
        try:
            print(f"\n>> Enter task (History: {len(history)} turns)")
            print("(Type 'END' on a new line to submit, or 'exit' to quit):")
            lines = []
            while True:
                line = input()
                if line.strip() == "END":
                    break
                if line.strip().lower() in ["exit", "quit"] and not lines:
                    return # Exit main loop
                lines.append(line)
            
            task_desc = "\n".join(lines).strip()
            
            if not task_desc:
                continue
            
            # Context construction
            context = {
                "previous_turns": history[-3:] # Keep last 3 turns to fit context
            }
            
            # Tool Injection Logic Check
            if "tool" in task_desc.lower() and "create" in task_desc.lower():
                context["instruction"] = "The user wants to create a tool. Output a valid Python function that performs the requested action."
                
            print(f"\n[Coordinator] Processing Task ({len(task_desc)} chars)...")
            result = await system.process_task(task_desc, history=context)
            
            # Update history
            history.append({"user": task_desc, "agent": result})
            
            print("\n" + "-"*20 + " Final Output " + "-"*20)
            print(result)
            print("-" * 54 + "\n")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Error processing task: {e}")

async def main():
    global SHARED_MODEL
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="models/qwen2.5-coder-7b-instruct-q4_k_m.gguf")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at {args.model_path}")
        print("Please download the model or specify valid path with --model_path")
        return

    print("Initializing system... (this may take a few seconds)")
    
    try:
        # Load Model
        SHARED_MODEL = LlamaCppModel("shared-qwen", args.model_path, n_gpu_layers=-1, n_ctx=8192)
        await SHARED_MODEL.load()
        print("Model loaded successfully.")
        
        # Patch Agents
        CodeGenerator._generate_code = patched_generate_code
        TestWriter._generate_tests = patched_generate_tests
        Reviewer._generate_review = patched_generate_review
        
        # Initialize System
        system = MultiAgentSystem(max_memory_gb=8)
        await system.initialize()

        # Start Loop
        await interactive_loop(system)
        
    except Exception as e:
        logger.error(f"Fatal Error: {e}")
    finally:
        if SHARED_MODEL:
            print("Unloading model...")
            await SHARED_MODEL.unload()
        if 'system' in locals():
            await system.shutdown()
        print("Goodbye!")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
