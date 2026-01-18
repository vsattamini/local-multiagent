import argparse
import asyncio
import logging
import sys
from src.main import MultiAgentSystem

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

async def run_task(task: str, memory_gb: int = 8) -> None:
    """Run a single task"""
    system = MultiAgentSystem(max_memory_gb=memory_gb)
    await system.initialize()
    
    try:
        result = await system.process_task(task)
        print(f"Task Result:\n{result}")
    finally:
        await system.shutdown()

async def run_evaluation(args) -> None:
    """Run SWEBench evaluation"""
    lite = args.lite
    memory_gb = args.memory
    
    system = MultiAgentSystem(max_memory_gb=memory_gb)
    await system.initialize()
    
    try:
        results = await system.run_evaluation(lite=lite)
        print(f"Evaluation Results:")
        print(f"Total tasks: {results['total_tasks']}")
        print(f"Completed: {results['completed_tasks']}")
        print(f"Average score: {results['average_score']:.2f}")
    finally:
        await system.shutdown()

def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Local Multi-Agent Coding System")
    subparsers = parser.add_subparsers(dest="command")
    
    # Task command
    task_parser = subparsers.add_parser("task", help="Run a single task")
    task_parser.add_argument("description", type=str, help="Task description")
    task_parser.add_argument("--memory", type=int, default=8, help="Maximum memory in GB")

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Run SWE-bench evaluation")
    eval_parser.add_argument("--lite", action="store_true", default=True, help="Use SWEBench Lite (subset)")
    eval_parser.add_argument("--full", action="store_true", help="Use full dataset (overrides --lite)")
    eval_parser.add_argument("--memory", type=int, default=8, help="Maximum memory in GB")
    
    args = parser.parse_args()
    
    if args.command == "task":
        asyncio.run(run_task(args.description, args.memory))
    elif args.command == "evaluate":
        if args.full:
            args.lite = False
        asyncio.run(run_evaluation(args))
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
