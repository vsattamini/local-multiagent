#!/usr/bin/env python3
"""
Analyze SWE-bench Lite Phase 3 experiment results.

This script compares the three experimental conditions:
1. Baseline (single agent)
2. Control (random routing)
3. Experimental (affinity routing/emergent specialization)

Usage:
    python scripts/analyze_swebench_results.py --results-dir results/swebench
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_results(results_dir: Path, condition: str) -> Dict:
    """Load results for a specific condition."""
    results_file = results_dir / condition / "results.json"

    if not results_file.exists():
        logger.warning(f"Results file not found: {results_file}")
        return None

    with open(results_file) as f:
        return json.load(f)


def compute_metrics(data: Dict) -> Dict:
    """Compute summary metrics for a condition."""
    results = data.get("results", [])

    total = len(results)
    successful = sum(1 for r in results if r.get("success", False))
    generated = sum(1 for r in results if r.get("solution_generated", False))

    # Category breakdown
    categories = {}
    for r in results:
        cat = r.get("category", "unknown")
        if cat not in categories:
            categories[cat] = {"total": 0, "success": 0}
        categories[cat]["total"] += 1
        if r.get("success", False):
            categories[cat]["success"] += 1

    # Repository breakdown
    repos = {}
    for r in results:
        repo = r.get("repo", "unknown").split('/')[-1]
        if repo not in repos:
            repos[repo] = {"total": 0, "success": 0}
        repos[repo]["total"] += 1
        if r.get("success", False):
            repos[repo]["success"] += 1

    # Execution time stats
    times = [r.get("execution_time", 0) for r in results]
    avg_time = sum(times) / len(times) if times else 0
    max_time = max(times) if times else 0
    min_time = min(times) if times else 0

    return {
        "total": total,
        "successful": successful,
        "generated": generated,
        "success_rate": successful / total if total > 0 else 0,
        "categories": categories,
        "repositories": repos,
        "timing": {
            "avg": avg_time,
            "max": max_time,
            "min": min_time
        }
    }


def compare_conditions(baseline_metrics: Dict, control_metrics: Dict, experimental_metrics: Dict):
    """Compare metrics across conditions."""
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 3 RESULTS: SWE-BENCH LITE COMPARISON")
    logger.info("=" * 80)

    # Overall comparison
    logger.info("\n### Overall Performance ###")
    logger.info(f"{'Condition':<15} {'Total':<8} {'Successful':<12} {'Success Rate':<12} {'Avg Time (s)':<12}")
    logger.info("-" * 80)

    for name, metrics in [
        ("Baseline", baseline_metrics),
        ("Control", control_metrics),
        ("Experimental", experimental_metrics)
    ]:
        if metrics:
            logger.info(
                f"{name:<15} {metrics['total']:<8} {metrics['successful']:<12} "
                f"{metrics['success_rate']:<12.1%} {metrics['timing']['avg']:<12.2f}"
            )

    # Category-wise comparison
    logger.info("\n### Category-wise Performance ###")

    all_categories = set()
    for metrics in [baseline_metrics, control_metrics, experimental_metrics]:
        if metrics:
            all_categories.update(metrics['categories'].keys())

    for category in sorted(all_categories):
        logger.info(f"\n{category.upper()}:")
        logger.info(f"{'Condition':<15} {'Success/Total':<15} {'Success Rate':<12}")
        logger.info("-" * 50)

        for name, metrics in [
            ("Baseline", baseline_metrics),
            ("Control", control_metrics),
            ("Experimental", experimental_metrics)
        ]:
            if metrics and category in metrics['categories']:
                cat_stats = metrics['categories'][category]
                rate = cat_stats['success'] / cat_stats['total'] if cat_stats['total'] > 0 else 0
                logger.info(
                    f"{name:<15} {cat_stats['success']}/{cat_stats['total']:<14} {rate:<12.1%}"
                )

    # Repository-wise comparison
    logger.info("\n### Repository-wise Performance ###")

    all_repos = set()
    for metrics in [baseline_metrics, control_metrics, experimental_metrics]:
        if metrics:
            all_repos.update(metrics['repositories'].keys())

    for repo in sorted(all_repos):
        logger.info(f"\n{repo}:")
        logger.info(f"{'Condition':<15} {'Success/Total':<15} {'Success Rate':<12}")
        logger.info("-" * 50)

        for name, metrics in [
            ("Baseline", baseline_metrics),
            ("Control", control_metrics),
            ("Experimental", experimental_metrics)
        ]:
            if metrics and repo in metrics['repositories']:
                repo_stats = metrics['repositories'][repo]
                rate = repo_stats['success'] / repo_stats['total'] if repo_stats['total'] > 0 else 0
                logger.info(
                    f"{name:<15} {repo_stats['success']}/{repo_stats['total']:<14} {rate:<12.1%}"
                )

    # Relative improvement analysis
    logger.info("\n### Relative Improvements ###")

    if baseline_metrics and experimental_metrics:
        baseline_rate = baseline_metrics['success_rate']
        experimental_rate = experimental_metrics['success_rate']

        if baseline_rate > 0:
            improvement = ((experimental_rate - baseline_rate) / baseline_rate) * 100
            logger.info(f"Experimental vs Baseline: {improvement:+.1f}% relative improvement")

    if control_metrics and experimental_metrics:
        control_rate = control_metrics['success_rate']
        experimental_rate = experimental_metrics['success_rate']

        if control_rate > 0:
            improvement = ((experimental_rate - control_rate) / control_rate) * 100
            logger.info(f"Experimental vs Control: {improvement:+.1f}% relative improvement")

    logger.info("\n" + "=" * 80)


def generate_report(results_dir: Path, output_file: str = "swebench_analysis.md"):
    """Generate markdown report."""
    baseline_data = load_results(results_dir, "baseline")
    control_data = load_results(results_dir, "control")
    experimental_data = load_results(results_dir, "experimental")

    baseline_metrics = compute_metrics(baseline_data) if baseline_data else None
    control_metrics = compute_metrics(control_data) if control_data else None
    experimental_metrics = compute_metrics(experimental_data) if experimental_data else None

    # Print to console
    compare_conditions(baseline_metrics, control_metrics, experimental_metrics)

    # Generate markdown report
    output_path = results_dir / output_file

    with open(output_path, 'w') as f:
        f.write("# SWE-bench Lite Phase 3 Analysis\n\n")
        f.write("## Experimental Design\n\n")
        f.write("This experiment tested emergent specialization on SWE-bench Lite with three conditions:\n\n")
        f.write("1. **Baseline**: Single agent (equivalent token budget)\n")
        f.write("2. **Control**: 3 agents with random routing\n")
        f.write("3. **Experimental**: 3 agents with affinity-based routing (emergent specialization)\n\n")

        f.write("## Overall Results\n\n")
        f.write("| Condition | Total | Successful | Success Rate | Avg Time (s) |\n")
        f.write("|-----------|-------|------------|--------------|-------------|\n")

        for name, metrics in [
            ("Baseline", baseline_metrics),
            ("Control", control_metrics),
            ("Experimental", experimental_metrics)
        ]:
            if metrics:
                f.write(
                    f"| {name} | {metrics['total']} | {metrics['successful']} | "
                    f"{metrics['success_rate']:.1%} | {metrics['timing']['avg']:.2f} |\n"
                )

        f.write("\n## Category-wise Performance\n\n")

        if baseline_metrics:
            all_categories = sorted(baseline_metrics['categories'].keys())

            f.write("| Category | Baseline | Control | Experimental |\n")
            f.write("|----------|----------|---------|-------------|\n")

            for category in all_categories:
                baseline_cat = baseline_metrics['categories'].get(category, {"success": 0, "total": 0})
                control_cat = control_metrics['categories'].get(category, {"success": 0, "total": 0}) if control_metrics else {"success": 0, "total": 0}
                experimental_cat = experimental_metrics['categories'].get(category, {"success": 0, "total": 0}) if experimental_metrics else {"success": 0, "total": 0}

                b_rate = baseline_cat['success'] / baseline_cat['total'] if baseline_cat['total'] > 0 else 0
                c_rate = control_cat['success'] / control_cat['total'] if control_cat['total'] > 0 else 0
                e_rate = experimental_cat['success'] / experimental_cat['total'] if experimental_cat['total'] > 0 else 0

                f.write(
                    f"| {category} | {b_rate:.1%} ({baseline_cat['success']}/{baseline_cat['total']}) | "
                    f"{c_rate:.1%} ({control_cat['success']}/{control_cat['total']}) | "
                    f"{e_rate:.1%} ({experimental_cat['success']}/{experimental_cat['total']}) |\n"
                )

        f.write("\n## Repository-wise Performance\n\n")

        if baseline_metrics:
            all_repos = sorted(baseline_metrics['repositories'].keys())

            f.write("| Repository | Baseline | Control | Experimental |\n")
            f.write("|------------|----------|---------|-------------|\n")

            for repo in all_repos:
                baseline_repo = baseline_metrics['repositories'].get(repo, {"success": 0, "total": 0})
                control_repo = control_metrics['repositories'].get(repo, {"success": 0, "total": 0}) if control_metrics else {"success": 0, "total": 0}
                experimental_repo = experimental_metrics['repositories'].get(repo, {"success": 0, "total": 0}) if experimental_metrics else {"success": 0, "total": 0}

                b_rate = baseline_repo['success'] / baseline_repo['total'] if baseline_repo['total'] > 0 else 0
                c_rate = control_repo['success'] / control_repo['total'] if control_repo['total'] > 0 else 0
                e_rate = experimental_repo['success'] / experimental_repo['total'] if experimental_repo['total'] > 0 else 0

                f.write(
                    f"| {repo} | {b_rate:.1%} ({baseline_repo['success']}/{baseline_repo['total']}) | "
                    f"{c_rate:.1%} ({control_repo['success']}/{control_repo['total']}) | "
                    f"{e_rate:.1%} ({experimental_repo['success']}/{experimental_repo['total']}) |\n"
                )

        f.write("\n## Conclusions\n\n")
        f.write("**Note**: This is a mock experiment for infrastructure validation. ")
        f.write("Actual performance metrics would come from running the real multi-agent system with model inference.\n\n")

        f.write("For a full Phase 3 experiment:\n")
        f.write("1. Replace mock solution generation with actual model inference\n")
        f.write("2. Integrate SWE-bench evaluation harness for test execution\n")
        f.write("3. Run with sufficient compute budget (30-50 issues × 3 conditions)\n")
        f.write("4. Analyze specialization patterns and transfer from HumanEval\n")

    logger.info(f"\nMarkdown report saved to: {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Analyze SWE-bench Phase 3 results")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/swebench",
        help="Results directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="swebench_analysis.md",
        help="Output markdown file"
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        return 1

    generate_report(results_dir, args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
