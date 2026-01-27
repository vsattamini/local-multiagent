#!/usr/bin/env python3
"""
Run comprehensive SWE-bench prompt sweep.
Automates the execution of different prompt versions on appropriate models.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.swebench.prompts import get_recommended_config, PROMPT_VERSIONS, list_versions_by_model


def run_experiment(
    prompt_version: str,
    model_path: str,
    output_base: str,
    config: Dict,
    dry_run: bool = False
):
    """Run a single experiment configuration."""
    output_dir = Path(output_base) / f"swebench_sweep_{prompt_version}"
    
    cmd = [
        "python", "scripts/run_swebench_experiment.py",
        "--condition", "experimental",
        "--model-path", model_path,
        "--n-agents", str(config["num_agents"]),
        "--n-issues", "50",  # Standard sweep size
        "--router-temp", str(config["temperature"]),
        "--prompt-version", prompt_version,
        "--output-dir", str(output_dir)
    ]
    
    # Passes not yet implemented in run script, ignoring passes_per_agent for now 
    # and relying on standard one-shot per agent unless we modify the main loop.
    # The current run_swebench_experiment.py runs ONE generation per agent.
    # "passes" in the user note probably referred to total attempts or strictly 
    # how many times we retry. The current script does n-agents parallel attempts.
    
    print(f"[{prompt_version}] Starting run with {config['target_model']} model...")
    print(f"[{prompt_version}] Config: Temp={config['temperature']}, Agents={config['num_agents']}")
    print(f"[{prompt_version}] Command: {' '.join(cmd)}")
    
    if not dry_run:
        # Ensure output directory exists for logging
        output_dir.mkdir(parents=True, exist_ok=True)
        
        start_time = time.time()
        try:
            # Run with unbuffered output
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Streaming output to keep user informed (simplistic)
            with open(output_dir / "run.log", "w") as log:
                for line in process.stdout:
                    # Print only key milestones to stdout to avoid clutter
                    if "Solved" in line or "Generated" in line or "Error" in line:
                        print(f"[{prompt_version}] {line.strip()}")
                    log.write(line)
            
            process.wait()
            duration = time.time() - start_time
            
            if process.returncode == 0:
                print(f"[{prompt_version}] COMPLETED in {duration:.1f}s")
            else:
                print(f"[{prompt_version}] FAILED (code {process.returncode})")
                
        except Exception as e:
            print(f"[{prompt_version}] EXCEPTION: {e}")
    else:
        print(f"[{prompt_version}] (Dry Run) Would execute now.")


def main():
    parser = argparse.ArgumentParser(description="Run comprehensive prompt sweep")
    parser.add_argument("--model-1.5b", dest="model_1_5b", help="Path to 1.5B model")
    parser.add_argument("--model-7b", dest="model_7b", help="Path to 7B model")
    parser.add_argument("--base-dir", default="results", help="Base directory for results")
    parser.add_argument("--parallel-1.5b", dest="parallel_1_5b", type=int, default=3, help="Parallel jobs for 1.5B")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")
    args = parser.parse_args()

    # create base dir
    if not args.dry_run:
        Path(args.base_dir).mkdir(exist_ok=True, parents=True)

    # filter v1, v2 as they are baselines
    all_v3_to_v15 = [f"v{i}" for i in range(3, 16)]
    
    # 1.5B Prompts
    prompts_1_5b = [v for v in all_v3_to_v15 if v in list_versions_by_model("1.5B")]
    
    if args.model_1_5b:
        print("\n" + "="*50)
        print(f"LAUNCHING 1.5B SWEEP ({len(prompts_1_5b)} prompts, Parallel={args.parallel_1_5b})")
        print("Prompts: " + ", ".join(prompts_1_5b))
        print("="*50)
        
        with ThreadPoolExecutor(max_workers=args.parallel_1_5b) as executor:
            futures = []
            for version in prompts_1_5b:
                config = get_recommended_config(version)
                futures.append(
                    executor.submit(
                        run_experiment,
                        version,
                        args.model_1_5b,
                        args.base_dir,
                        config,
                        args.dry_run
                    )
                )
            
            # Wait for all 1.5B runs
            for f in futures:
                f.result()
    else:
        print("Skipping 1.5B sweep (no model path provided)")

    # 7B Prompts
    prompts_7b = [v for v in all_v3_to_v15 if v in list_versions_by_model("7B")]
    
    if args.model_7b:
        print("\n" + "="*50)
        print(f"LAUNCHING 7B SWEEP ({len(prompts_7b)} prompts, Sequential)")
        print("Prompts: " + ", ".join(prompts_7b))
        print("="*50)
        
        for version in prompts_7b:
            config = get_recommended_config(version)
            run_experiment(
                version,
                args.model_7b,
                args.base_dir,
                config,
                args.dry_run
            )
    else:
        print("Skipping 7B sweep (no model path provided)")

    print("\nAll requested sweeps completed.")


if __name__ == "__main__":
    main()
