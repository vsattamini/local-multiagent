#!/usr/bin/env python3
"""
Validate existing SWE-bench results without re-running generation.

This script:
1. Loads existing results from sweep folders
2. Validates patches with format checking
3. Runs full SWE-bench validation with Docker
4. Saves incrementally to avoid losing progress

Usage:
    # Validate format only (fast)
    python scripts/validate_swebench_results.py

    # Validate with Docker harness (slow, requires swe-bench installed)
    python scripts/validate_swebench_results.py --full-validation
    
    # Validate specific folder
    python scripts/validate_swebench_results.py --results-dir results/swebench_sweep/1.5b/temp0_1_n5
    
    # Resume interrupted run
    python scripts/validate_swebench_results.py --full-validation --resume
"""

import argparse
import json
import subprocess
import tempfile
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import threading

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.swebench.validation import validate_patch, ValidationResult


@dataclass
class PostValidationResult:
    """Result of post-hoc validation."""
    instance_id: str
    generated: bool
    format_valid: bool
    has_diff_header: bool
    has_hunks: bool
    files_touched: int
    lines_added: int
    lines_removed: int
    # Full harness validation (optional)
    harness_validated: Optional[bool] = None
    harness_error: Optional[str] = None


def load_results_from_folder(folder: Path) -> List[Dict]:
    """Load results from a results folder."""
    # Try different result file locations
    candidates = [
        folder / "results.json",
        folder / "experimental" / "results.json",
        folder / "control" / "results.json",
        folder / "baseline" / "results.json",
    ]
    
    for path in candidates:
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            return data.get("results", [])
    
    return []


def load_existing_validation(folder: Path) -> Dict[str, Dict]:
    """Load existing validation results for resuming."""
    validation_file = folder / "validation_results.json"
    if validation_file.exists():
        with open(validation_file) as f:
            data = json.load(f)
        # Return dict keyed by instance_id for easy lookup
        return {r["instance_id"]: r for r in data.get("results", [])}
    return {}


def save_validation_incremental(folder: Path, validated: List[Dict], summary: Dict):
    """Save validation results incrementally to folder."""
    validation_file = folder / "validation_results.json"
    with open(validation_file, 'w') as f:
        json.dump({
            "summary": summary,
            "results": validated
        }, f, indent=2)


def validate_result(result: Dict) -> PostValidationResult:
    """Validate a single result using format checking."""
    model_output = result.get("model_output", "")
    
    if not model_output:
        return PostValidationResult(
            instance_id=result.get("instance_id", "unknown"),
            generated=False,
            format_valid=False,
            has_diff_header=False,
            has_hunks=False,
            files_touched=0,
            lines_added=0,
            lines_removed=0
        )
    
    validation = validate_patch(model_output)
    
    return PostValidationResult(
        instance_id=result.get("instance_id", "unknown"),
        generated=True,
        format_valid=validation.is_valid,
        has_diff_header=validation.has_diff_header,
        has_hunks=validation.has_hunks,
        files_touched=validation.files_touched,
        lines_added=validation.lines_added,
        lines_removed=validation.lines_removed
    )


def run_swebench_harness(instance_id: str, patch: str) -> tuple[bool, str]:
    """
    Run full SWE-bench harness validation using Docker.
    
    Requires swebench to be installed and Docker running.
    
    Returns:
        Tuple of (passed, error_message)
    """
    try:
        import json as json_module
        
        # Create predictions.jsonl with proper format
        prediction = {
            "instance_id": instance_id,
            "model_patch": patch,
            "model_name_or_path": "swarm-agent"
        }
        
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.jsonl', delete=False
        ) as f:
            f.write(json_module.dumps(prediction) + "\n")
            predictions_file = f.name
        
        # Create output directory for results
        with tempfile.TemporaryDirectory() as output_dir:
            # Run swe-bench evaluation
            cmd = [
                "python", "-m", "swebench.harness.run_evaluation",
                "--dataset_name", "princeton-nlp/SWE-bench_Lite",
                "--split", "test",
                "--predictions_path", predictions_file,
                "--instance_ids", instance_id,
                "--max_workers", "1",
                "--timeout", "300",
                "--run_id", "validation"
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=900,  # 15 minute timeout per issue
                cwd=output_dir
            )
            
            os.unlink(predictions_file)
            
            # Check result - look for "resolved: 1" or similar
            stdout = result.stdout.lower()
            if result.returncode == 0:
                # Check for resolved instances
                if "instances resolved: 1" in stdout or "resolved: 1" in stdout:
                    return True, ""
                # Check for explicit failure indicators  
                if "instances resolved: 0" in stdout:
                    return False, "Tests failed (not resolved)"
                # Patch apply failure
                if "patch apply failed" in stdout:
                    return False, "Patch apply failed"
                # Default: assume failure if not clear
                return False, result.stdout[:300] if result.stdout else "Unknown result"
            else:
                return False, result.stderr[:500] if result.stderr else result.stdout[:500]
            
    except subprocess.TimeoutExpired:
        return False, "Harness timeout (15 min)"
    except FileNotFoundError:
        return False, "swe-bench not installed (pip install swebench)"
    except Exception as e:
        return False, str(e)[:200]


def validate_single_issue(
    result: Dict,
    full_validation: bool,
    existing: Dict,
    resume: bool
) -> Tuple[Dict, bool]:
    """Validate a single issue (thread-safe for parallel execution)."""
    instance_id = result.get("instance_id", "unknown")
    
    # Check if already validated (for resume)
    if resume and instance_id in existing:
        existing_result = existing[instance_id]
        if not full_validation or existing_result.get("harness_validated") is not None:
            return existing_result, True  # True = skipped
    
    # Validate format
    v = validate_result(result)
    
    # Run harness if requested
    if full_validation and v.generated:
        passed, error = run_swebench_harness(
            instance_id,
            result.get("model_output", "")
        )
        v.harness_validated = passed
        v.harness_error = error if not passed else None
    
    return asdict(v), False  # False = not skipped


# Thread-safe lock for saving
_save_lock = threading.Lock()


def validate_folder(
    folder: Path,
    full_validation: bool = False,
    resume: bool = False,
    workers: int = 1
) -> Dict:
    """Validate all results in a folder with parallel processing and incremental saving."""
    results = load_results_from_folder(folder)
    
    if not results:
        return {"error": f"No results found in {folder}"}
    
    # Load existing validation if resuming
    existing = load_existing_validation(folder) if resume else {}
    
    validated = []
    skipped = 0
    
    if workers == 1:
        # Sequential processing (original behavior)
        iterator = tqdm(results, desc=f"Validating {folder.name}", unit="issue")
        
        for result in iterator:
            v_result, was_skipped = validate_single_issue(
                result, full_validation, existing, resume
            )
            validated.append(v_result)
            if was_skipped:
                skipped += 1
            
            # Update progress bar
            if full_validation:
                iterator.set_postfix(
                    valid=sum(1 for x in validated if x.get("format_valid", False)),
                    passed=sum(1 for x in validated if x.get("harness_validated", False))
                )
            
            # Save incrementally every 5 issues
            if len(validated) % 5 == 0:
                summary = compute_summary(validated, full_validation)
                save_validation_incremental(folder, validated, summary)
    else:
        # Parallel processing
        print(f"  Using {workers} parallel workers")
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    validate_single_issue,
                    result, full_validation, existing, resume
                ): result.get("instance_id", "unknown")
                for result in results
            }
            
            pbar = tqdm(total=len(results), desc=f"Validating {folder.name}", unit="issue")
            
            for future in as_completed(futures):
                instance_id = futures[future]
                try:
                    v_result, was_skipped = future.result()
                    validated.append(v_result)
                    if was_skipped:
                        skipped += 1
                except Exception as e:
                    validated.append({
                        "instance_id": instance_id,
                        "generated": False,
                        "format_valid": False,
                        "harness_error": str(e)[:200]
                    })
                
                pbar.update(1)
                
                # Update stats
                if full_validation:
                    pbar.set_postfix(
                        valid=sum(1 for x in validated if x.get("format_valid", False)),
                        passed=sum(1 for x in validated if x.get("harness_validated", False))
                    )
                
                # Save incrementally every 5 issues (thread-safe)
                if len(validated) % 5 == 0:
                    with _save_lock:
                        summary = compute_summary(validated, full_validation)
                        save_validation_incremental(folder, validated, summary)
            
            pbar.close()
    
    # Final save
    summary = compute_summary(validated, full_validation)
    save_validation_incremental(folder, validated, summary)
    
    if skipped > 0:
        print(f"  (Skipped {skipped} already-validated issues)")
    
    return {
        "summary": summary,
        "results": validated
    }


def compute_summary(validated: List[Dict], full_validation: bool) -> Dict:
    """Compute summary statistics."""
    total = len(validated)
    generated = sum(1 for v in validated if v["generated"])
    format_valid = sum(1 for v in validated if v["format_valid"])
    
    summary = {
        "total": total,
        "generated": generated,
        "format_valid": format_valid,
        "generation_rate": generated / total if total > 0 else 0,
        "format_valid_rate": format_valid / total if total > 0 else 0,
    }
    
    if full_validation:
        harness_passed = sum(1 for v in validated if v.get("harness_validated"))
        summary["harness_passed"] = harness_passed
        summary["harness_rate"] = harness_passed / total if total > 0 else 0
    
    return summary


def find_all_sweep_folders(base_dir: str = "results/swebench_sweep") -> List[Path]:
    """Find all result folders in the sweep directory."""
    base = Path(base_dir)
    folders = []
    
    if not base.exists():
        return folders
    
    for model_dir in base.iterdir():
        if not model_dir.is_dir():
            continue
        for config_dir in model_dir.iterdir():
            if config_dir.is_dir():
                folders.append(config_dir)
    
    return sorted(folders)


def main():
    parser = argparse.ArgumentParser(
        description="Validate existing SWE-bench results"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/swebench_sweep",
        help="Directory containing results to validate"
    )
    parser.add_argument(
        "--full-validation",
        action="store_true",
        help="Run full SWE-bench harness validation (requires Docker + swe-bench)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous run, skipping already-validated issues"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers for folder processing (default: 1)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/validation_report.json",
        help="Output file for combined validation report"
    )
    
    args = parser.parse_args()
    
    results_path = Path(args.results_dir)
    
    # Check if it's a single folder or sweep directory
    if (results_path / "results.json").exists() or \
       (results_path / "experimental").exists():
        # Single folder
        folders = [results_path]
    else:
        # Sweep directory
        folders = find_all_sweep_folders(str(results_path))
    
    if not folders:
        print(f"No results found in {args.results_dir}")
        return
    
    print(f"Found {len(folders)} result folder(s) to validate")
    if args.resume:
        print("Resume mode: will skip already-validated issues")
    if args.workers > 1:
        print(f"Using {args.workers} parallel workers for folders")
    print()
    
    all_results = []
    
    print("=" * 90)
    print("VALIDATION RESULTS")
    print("=" * 90)
    print()
    
    def process_folder(folder: Path) -> Dict:
        """Process a single folder (for parallel execution)."""
        validation = validate_folder(folder, args.full_validation, args.resume, workers=1)
        if "error" not in validation:
            validation["folder"] = str(folder)
        return validation
    
    if args.workers == 1:
        # Sequential processing
        for i, folder in enumerate(folders, 1):
            print(f"\n[{i}/{len(folders)}] Processing {folder.parent.name}/{folder.name}")
            
            validation = process_folder(folder)
            
            if "error" in validation:
                print(f"  ERROR: {validation['error']}")
                continue
            
            summary = validation["summary"]
            
            # Print folder summary
            print(f"  Total: {summary['total']}, Generated: {summary['generated']}, "
                  f"Format Valid: {summary['format_valid']} ({summary['format_valid_rate']:.1%})")
            
            if args.full_validation:
                print(f"  Harness Passed: {summary.get('harness_passed', 0)} "
                      f"({summary.get('harness_rate', 0):.1%})")
            
            print(f"  Saved to: {folder}/validation_results.json")
            
            all_results.append(validation)
    else:
        # Parallel folder processing
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process_folder, folder): folder
                for folder in folders
            }
            
            pbar = tqdm(total=len(folders), desc="Processing folders", unit="folder")
            
            for future in as_completed(futures):
                folder = futures[future]
                try:
                    validation = future.result()
                    
                    if "error" in validation:
                        tqdm.write(f"[{folder.parent.name}/{folder.name}] ERROR: {validation['error']}")
                    else:
                        summary = validation["summary"]
                        status = f"Valid: {summary['format_valid']}/{summary['total']}"
                        if args.full_validation:
                            status += f", Passed: {summary.get('harness_passed', 0)}"
                        tqdm.write(f"[{folder.parent.name}/{folder.name}] {status}")
                        all_results.append(validation)
                        
                except Exception as e:
                    tqdm.write(f"[{folder.parent.name}/{folder.name}] EXCEPTION: {e}")
                
                pbar.update(1)
            
            pbar.close()
    
    print()
    
    # Overall summary
    if all_results:
        total_issues = sum(r["summary"]["total"] for r in all_results)
        total_generated = sum(r["summary"]["generated"] for r in all_results)
        total_valid = sum(r["summary"]["format_valid"] for r in all_results)
        
        print("=" * 90)
        print("OVERALL SUMMARY")
        print("=" * 90)
        print(f"Folders validated: {len(all_results)}")
        print(f"Total issues:      {total_issues}")
        print(f"Generated:         {total_generated} ({total_generated/total_issues:.1%})")
        print(f"Format valid:      {total_valid} ({total_valid/total_issues:.1%})")
        
        if args.full_validation:
            total_harness = sum(r["summary"].get("harness_passed", 0) for r in all_results)
            print(f"Harness passed:    {total_harness} ({total_harness/total_issues:.1%})")
    
    # Save combined report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump({
            "summary": {
                "folders_validated": len(all_results),
                "total_issues": total_issues if all_results else 0,
                "total_generated": total_generated if all_results else 0,
                "total_format_valid": total_valid if all_results else 0,
            },
            "folder_results": all_results
        }, f, indent=2)
    
    print(f"\nCombined report saved to: {output_path}")
    print("\nPer-folder results saved to: <folder>/validation_results.json")


if __name__ == "__main__":
    main()

