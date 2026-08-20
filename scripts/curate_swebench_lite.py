#!/usr/bin/env python3
"""
Curate a subset of SWE-bench Lite for Phase 3 experiments.

This script:
1. Loads SWE-bench Lite dataset
2. Filters issues by complexity (lines, files, repositories)
3. Categorizes issues by type (bug/feature/refactor)
4. Saves curated subset to data/swebench_curated.json
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.swebench import SWEBenchLoader, TaskInstance

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Curate SWE-bench Lite subset")
    parser.add_argument(
        "--n-issues",
        type=int,
        default=50,
        help="Target number of issues (default: 50)"
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=100,
        help="Maximum lines changed per issue (default: 100)"
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=3,
        help="Maximum files changed per issue (default: 3)"
    )
    parser.add_argument(
        "--repositories",
        nargs="+",
        default=["django", "requests", "flask", "matplotlib", "scikit-learn", "pytest", "sympy"],
        help="Repositories to include"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/swebench_curated.json",
        help="Output file path"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to use (default: test)"
    )

    args = parser.parse_args()

    # Load and curate
    logger.info("Loading SWE-bench Lite dataset...")
    loader = SWEBenchLoader(dataset_name="princeton-nlp/SWE-bench_Lite")

    logger.info(f"Curating subset with criteria:")
    logger.info(f"  - Max issues: {args.n_issues}")
    logger.info(f"  - Max lines: {args.max_lines}")
    logger.info(f"  - Max files: {args.max_files}")
    logger.info(f"  - Repositories: {', '.join(args.repositories)}")

    curated = loader.get_curated_subset(
        n_issues=args.n_issues,
        max_lines=args.max_lines,
        max_files=args.max_files,
        repositories=args.repositories,
        split=args.split
    )

    # Categorize issues
    logger.info("Categorizing issues...")
    loader.categorize_issues(curated)

    # Analyze curation results
    logger.info("\n" + "=" * 60)
    logger.info("CURATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total curated issues: {len(curated)}")

    # Repository distribution
    repo_counts = {}
    for task in curated:
        repo = task.repo.split('/')[-1]
        repo_counts[repo] = repo_counts.get(repo, 0) + 1

    logger.info("\nRepository distribution:")
    for repo, count in sorted(repo_counts.items(), key=lambda x: -x[1]):
        logger.info(f"  {repo}: {count}")

    # Category distribution
    category_counts = {}
    for task in curated:
        if task.category:
            category_counts[task.category] = category_counts.get(task.category, 0) + 1

    logger.info("\nCategory distribution:")
    for category, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        logger.info(f"  {category}: {count}")

    # Complexity statistics
    avg_lines = sum(t.estimated_lines_changed or 0 for t in curated) / len(curated)
    avg_files = sum(t.estimated_files_changed or 0 for t in curated) / len(curated)

    logger.info("\nComplexity statistics:")
    logger.info(f"  Average lines changed: {avg_lines:.1f}")
    logger.info(f"  Average files changed: {avg_files:.1f}")

    # Save to file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to serializable format
    output_data = {
        "metadata": {
            "n_issues": len(curated),
            "max_lines": args.max_lines,
            "max_files": args.max_files,
            "repositories": args.repositories,
            "avg_lines_changed": avg_lines,
            "avg_files_changed": avg_files
        },
        "issues": []
    }

    for task in curated:
        output_data["issues"].append({
            "instance_id": task.instance_id,
            "problem_statement": task.problem_statement,
            "repo": task.repo,
            "base_commit": task.base_commit,
            "patch": task.patch,
            "test_patch": task.test_patch,
            "version": task.version,
            "environment_setup_commit": task.environment_setup_commit,
            "category": task.category,
            "difficulty": task.difficulty,
            "estimated_lines_changed": task.estimated_lines_changed,
            "estimated_files_changed": task.estimated_files_changed
        })

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"\nCurated subset saved to: {output_path}")
    logger.info("=" * 60)

    # Print sample issues
    logger.info("\nSample issues (first 3):")
    for i, task in enumerate(curated[:3], 1):
        logger.info(f"\n{i}. {task.instance_id}")
        logger.info(f"   Repo: {task.repo}")
        logger.info(f"   Category: {task.category}")
        logger.info(f"   Lines: {task.estimated_lines_changed}, Files: {task.estimated_files_changed}")
        logger.info(f"   Problem: {task.problem_statement[:100]}...")


if __name__ == "__main__":
    main()
