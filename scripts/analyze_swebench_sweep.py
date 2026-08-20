#!/usr/bin/env python3
"""
Analyze SWE-bench combination sweep results.

Loads results from results/swebench_sweep/ and produces:
- Summary table of S, D, success rate by (temp, n_agents, model)
- Optimal configuration identification
- Model output analysis statistics
"""

import json
from pathlib import Path
from typing import Dict, List
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_sweep_results(base_dir: str = "results/swebench_sweep") -> List[Dict]:
    """Load all sweep results from the directory structure."""
    base_path = Path(base_dir)
    results = []
    
    if not base_path.exists():
        print(f"No results found at {base_dir}")
        return []
    
    for model_dir in base_path.iterdir():
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name  # "1.5b" or "7b"
        
        for config_dir in model_dir.iterdir():
            if not config_dir.is_dir():
                continue
            
            # Parse config from folder name (e.g., "temp0_1_n5")
            config_name = config_dir.name
            parts = config_name.split("_")
            
            try:
                temp = float(parts[0].replace("temp", "") + "." + parts[1])
                n_agents = int(parts[2].replace("n", ""))
            except (IndexError, ValueError):
                print(f"  Skipping unparseable config: {config_name}")
                continue
            
            # Load results.json
            results_file = config_dir / "results.json"
            if not results_file.exists():
                # Try experimental subfolder
                results_file = config_dir / "experimental" / "results.json"
            
            if results_file.exists():
                with open(results_file) as f:
                    data = json.load(f)
                
                issues = data.get("results", [])
                
                # Compute metrics - distinguish generated from validated
                total = len(issues)
                generated = sum(1 for r in issues if r.get("generated", r.get("success", False)))
                valid_patches = sum(1 for r in issues if r.get("valid_patch", False))
                
                # Backwards compat: old results used "success" for generation
                successful = sum(1 for r in issues if r.get("success", False))
                
                # Agent distribution
                agent_counts = {}
                for r in issues:
                    aid = r.get("agent_id", 0)
                    agent_counts[aid] = agent_counts.get(aid, 0) + 1
                
                # Avg output length
                output_lengths = [
                    len(r.get("model_output", "")) 
                    for r in issues 
                    if r.get("model_output")
                ]
                avg_output_len = sum(output_lengths) / len(output_lengths) if output_lengths else 0
                
                # Validation stats
                has_diff = sum(1 for r in issues if r.get("validation", {}).get("has_diff_header", False))
                has_hunks = sum(1 for r in issues if r.get("validation", {}).get("has_hunks", False))
                
                results.append({
                    "model": model_name,
                    "temp": temp,
                    "n_agents": n_agents,
                    "total_issues": total,
                    "generated": generated,
                    "valid_patches": valid_patches,
                    "has_diff_header": has_diff,
                    "has_hunks": has_hunks,
                    "generation_rate": generated / total if total > 0 else 0,
                    "validation_rate": valid_patches / total if total > 0 else 0,
                    "agent_distribution": agent_counts,
                    "avg_output_length": avg_output_len,
                    "path": str(config_dir)
                })
    
    return results


def print_results_table(results: List[Dict]):
    """Print a formatted results table."""
    if not results:
        print("No results to display.")
        return
    
    # Sort by model, temp, n_agents
    results = sorted(results, key=lambda x: (x["model"], x["temp"], x["n_agents"]))
    
    print("\n" + "="*90)
    print("SWE-BENCH COMBINATION SWEEP RESULTS")
    print("="*90)
    print("\nMetric Clarification:")
    print("  - Generated: Model produced output without exception")
    print("  - Valid: Output is a valid unified diff format with hunks")
    print()
    
    print("| Model | Temp | Agents | Total | Generated | Valid | Valid Rate | Avg Len |")
    print("|-------|------|--------|-------|-----------|-------|------------|---------|")
    
    for r in results:
        valid = r.get("valid_patches", 0)
        gen = r.get("generated", r.get("total_issues", 0))
        valid_rate = r.get("validation_rate", 0)
        print(f"| {r['model']:5} | {r['temp']:.1f} | {r['n_agents']:6} | "
              f"{r['total_issues']:5} | {gen:9} | {valid:5} | "
              f"{valid_rate:10.1%} | {r['avg_output_length']:7.0f} |")
    
    # Find best config by validation rate
    if results:
        best = max(results, key=lambda x: x.get("validation_rate", 0))
        print(f"\nBest config by validation rate: model={best['model']}, temp={best['temp']}, "
              f"n_agents={best['n_agents']} (valid={best.get('valid_patches', 0)}/{best['total_issues']})")
        
        # Check if any old-format results (no valid_patches field)
        old_format = [r for r in results if "valid_patches" not in r or r.get("valid_patches") == 0]
        if old_format and len(old_format) == len(results):
            print("\n⚠️  Note: These results are from OLD format (before validation was added).")
            print("   Re-run experiments with updated script to get validation metrics.")


def print_model_comparison(results: List[Dict]):
    """Compare 1.5B vs 7B models."""
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    
    for model in ["1.5b", "7b"]:
        model_results = [r for r in results if r["model"] == model]
        if not model_results:
            print(f"\n{model.upper()}: No results")
            continue
        
        avg_success = sum(r["success_rate"] for r in model_results) / len(model_results)
        best = max(model_results, key=lambda x: x["success_rate"])
        
        print(f"\n{model.upper()}:")
        print(f"  Configs tested: {len(model_results)}")
        print(f"  Avg success rate: {avg_success:.1%}")
        print(f"  Best config: temp={best['temp']}, n_agents={best['n_agents']} "
              f"({best['success_rate']:.1%})")


def print_agent_analysis(results: List[Dict]):
    """Analyze agent distribution to check for specialization."""
    print("\n" + "="*80)
    print("AGENT DISTRIBUTION ANALYSIS")
    print("="*80)
    
    for r in results:
        dist = r.get("agent_distribution", {})
        if len(dist) > 1:
            total = sum(dist.values())
            entropy = 0
            for count in dist.values():
                p = count / total
                if p > 0:
                    import math
                    entropy -= p * math.log2(p)
            max_entropy = math.log2(len(dist))
            specialization = 1 - (entropy / max_entropy) if max_entropy > 0 else 0
            
            print(f"\n{r['model']} temp={r['temp']} n={r['n_agents']}:")
            print(f"  Distribution: {dist}")
            print(f"  Specialization proxy: {specialization:.3f}")


def main():
    print("Loading SWE-bench sweep results...")
    results = load_sweep_results()
    
    if not results:
        print("\nNo results found. Run the sweep first:")
        print("  ./scripts/run_swebench_sweep.sh")
        return
    
    print(f"Loaded {len(results)} experiment configurations")
    
    print_results_table(results)
    print_model_comparison(results)
    print_agent_analysis(results)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
