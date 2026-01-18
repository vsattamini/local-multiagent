
import asyncio
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.main import MultiAgentSystem

async def main():
    print("=== Demo: Architecture Verification (Mock) ===")
    
    # Initialize system with defaults (uses mock models)
    system = MultiAgentSystem(max_memory_gb=8)
    await system.initialize()
    
    task = "Write a Python function to calculate the factorial of a number."
    print(f"\nProcessing Task: {task}\n")
    
    # This will run Task -> Coordinator -> Decompose -> CodeGen -> TestWriter -> Reviewer
    # The agents use placeholder logic (Mock) as defined in src/agents/specialists/
    result = await system.process_task(task)
    
    print("\n=== Result ===")
    print(result)
    
    await system.shutdown()
    print("\n=== Demo Complete ===")

if __name__ == "__main__":
    asyncio.run(main())
