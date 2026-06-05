
import asyncio
import sys
import os
import argparse
import time
import logging
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.main import MultiAgentSystem
from src.models.llama_cpp import LlamaCppModel
from src.agents.specialists.code_generator import CodeGenerator
from src.agents.specialists.test_writer import TestWriter
from src.agents.specialists.reviewer import Reviewer
from src.coordination.coordinator import Coordinator

# Setup Logging
log_filename = f"execution_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("System")

SHARED_MODEL = None

async def patched_generate_code(self, prompt: str) -> str:
    agent_name = "CodeGenerator"
    logger.info(f"[{agent_name}] Starting Code Generation...")
    logger.debug(f"[{agent_name}] Prompt:\n{prompt}")
    
    start_time = time.perf_counter()
    
    # Direct call to access usage stats
    response = await asyncio.to_thread(
        SHARED_MODEL._llm.create_completion,
        prompt=prompt,
        max_tokens=1024,
        temperature=0.2,
        stop=["```\n", "User:", "<|endoftext|>"]
    )
    
    end_time = time.perf_counter()
    duration = end_time - start_time
    
    text = response["choices"][0]["text"]
    usage = response.get("usage", {})
    completion_tokens = usage.get("completion_tokens", 0)
    prompt_tokens = usage.get("prompt_tokens", 0)
    tps = completion_tokens / duration if duration > 0 else 0
    
    logger.info(f"[{agent_name}] Generation Complete in {duration:.2f}s")
    logger.info(f"[{agent_name}] Performance: {completion_tokens} tokens generated ({tps:.2f} T/s)")
    logger.info(f"[{agent_name}] Output:\n{text}")
    
    return text

async def patched_generate_tests(self, prompt: str) -> str:
    agent_name = "TestWriter"
    logger.info(f"[{agent_name}] Starting Test Generation...")
    
    start_time = time.perf_counter()
    response = await asyncio.to_thread(
        SHARED_MODEL._llm.create_completion,
        prompt=prompt,
        max_tokens=1024,
        temperature=0.2,
         stop=["```\n", "User:", "<|endoftext|>"]
    )
    end_time = time.perf_counter()
    duration = end_time - start_time
    
    text = response["choices"][0]["text"]
    usage = response.get("usage", {})
    tps = usage.get("completion_tokens", 0) / duration if duration > 0 else 0
    
    logger.info(f"[{agent_name}] Generation Complete in {duration:.2f}s ({tps:.2f} T/s)")
    logger.info(f"[{agent_name}] Output:\n{text}")
    return text

async def patched_generate_review(self, prompt: str) -> str:
    agent_name = "Reviewer"
    logger.info(f"[{agent_name}] Starting Review...")
    
    start_time = time.perf_counter()
    response = await asyncio.to_thread(
        SHARED_MODEL._llm.create_completion,
        prompt=prompt,
        max_tokens=1024,
        temperature=0.2,
         stop=["```\n", "User:", "<|endoftext|>"]
    )
    end_time = time.perf_counter()
    duration = end_time - start_time
    
    text = response["choices"][0]["text"]
    usage = response.get("usage", {})
    tps = usage.get("completion_tokens", 0) / duration if duration > 0 else 0
    
    logger.info(f"[{agent_name}] Review Complete in {duration:.2f}s ({tps:.2f} T/s)")
    logger.info(f"[{agent_name}] Output:\n{text}")
    return text

# We also wrap the coordinator to log its high-level actions
class LoggingCoordinator(Coordinator):
    async def process_task(self, task):
        logger.info(f"[Coordinator] Received Task: {task.description}")
        start_time = time.perf_counter()
        
        # Call original (which uses decomposition logic)
        results = await super().process_task(task)
        
        duration = time.perf_counter() - start_time
        logger.info(f"[Coordinator] Workflow Complete in {duration:.2f}s")
        return results

async def main():
    global SHARED_MODEL
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="models/qwen2.5-coder-7b-instruct-q4_k_m.gguf")
    parser.add_argument("--task", default="Write a Python class for a Binary Search Tree with insert and search methods.")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        logger.error(f"Model not found at {args.model_path}")
        return

    logger.info("=== System Initialization ===")
    logger.info(f"Model Path: {args.model_path}")
    
    try:
        # Load Model
        logger.info("Loading LlamaCppModel...")
        load_start = time.perf_counter()
        SHARED_MODEL = LlamaCppModel("shared-qwen", args.model_path, n_gpu_layers=-1, n_ctx=8192)
        await SHARED_MODEL.load()
        logger.info(f"Model Loaded in {time.perf_counter() - load_start:.2f}s")
        
        # Patch Agents
        logger.info("Patching Agents with Monitoring Hooks...")
        CodeGenerator._generate_code = patched_generate_code
        TestWriter._generate_tests = patched_generate_tests
        Reviewer._generate_review = patched_generate_review
        
        # Initialize System with Logging Coordinator
        system = MultiAgentSystem(max_memory_gb=8)
        # Swap out the default coordinator for our logging one
        system.coordinator = LoggingCoordinator("logging-coordinator", "shared-qwen")
        
        await system.initialize()

        logger.info("\n=== Starting Execution Loop ===\n")
        
        result = await system.process_task(args.task)
        
        logger.info("\n=== Final Result ===\n")
        print(result)
        
    except Exception as e:
        logger.exception("Fatal Error during execution")
    finally:
        if SHARED_MODEL:
            logger.info("Unloading Model...")
            await SHARED_MODEL.unload()
        if 'system' in locals():
            await system.shutdown()
        logger.info("Shutdown Complete")

if __name__ == "__main__":
    asyncio.run(main())
