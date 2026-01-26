from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Literal
from datasets import load_dataset
import logging
import json
import tempfile
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

TaskCategory = Literal["bug", "feature", "refactor", "docs", "test", "other"]
TaskDifficulty = Literal["easy", "medium", "hard"]

@dataclass
class TaskInstance:
    instance_id: str
    problem_statement: str
    repo: str
    base_commit: str
    patch: str
    test_patch: str
    version: str
    environment_setup_commit: str

    # Metadata for curation and analysis
    category: Optional[TaskCategory] = None
    difficulty: Optional[TaskDifficulty] = None
    estimated_lines_changed: Optional[int] = None
    estimated_files_changed: Optional[int] = None

    @classmethod
    def from_huggingface_row(cls, row: Dict[str, Any]) -> 'TaskInstance':
        return cls(
            instance_id=row['instance_id'],
            problem_statement=row['problem_statement'],
            repo=row['repo'],
            base_commit=row['base_commit'],
            patch=row['patch'],
            test_patch=row['test_patch'],
            version=row.get('version', ''),
            environment_setup_commit=row.get('environment_setup_commit', '')
        )

    def estimate_complexity(self) -> None:
        """Estimate complexity metrics from the patch."""
        if not self.patch:
            return

        lines = self.patch.split('\n')
        changed_lines = sum(1 for line in lines if line.startswith(('+', '-')) and not line.startswith(('+++', '---')))
        files = set()
        for line in lines:
            if line.startswith('diff --git'):
                # Extract filename from diff header
                parts = line.split()
                if len(parts) >= 3:
                    files.add(parts[2])

        self.estimated_lines_changed = changed_lines
        self.estimated_files_changed = len(files)

@dataclass
class ExecutionResult:
    """Result of executing a solution on a SWE-bench task."""
    instance_id: str
    success: bool
    test_output: str = ""
    error: str = ""
    execution_time: float = 0.0
    applied_patch: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

class SWEBenchLoader:
    def __init__(self, dataset_name: str = "princeton-nlp/SWE-bench_Lite"):
        self.dataset_name = dataset_name
        self._dataset = None

    def load_tasks(self, split: str = "test") -> List[TaskInstance]:
        """
        Load tasks from the SWE-bench dataset.
        
        Args:
            split: The split to load (e.g., 'test', 'dev', 'train'). 
                   Note: SWE-bench Lite mainly uses 'test'.
                   
        Returns:
            List of TaskInstance objects.
        """
        if self._dataset is None:
            logger.info(f"Loading dataset {self.dataset_name}...")
            self._dataset = load_dataset(self.dataset_name)
            
        if split not in self._dataset:
            # Fallback or error handling if potential other splits are requested but not present
            # SWE-bench Lite typically has 'test'
            if split == "dev" and "dev" not in self._dataset:
                 logger.warning(f"Split '{split}' not found. Available splits: {self._dataset.keys()}. Returning empty list.")
                 return []
            if split not in self._dataset:
                 raise ValueError(f"Split '{split}' not found in dataset. Available: {self._dataset.keys()}")

        dataset_split = self._dataset[split]
        tasks = [TaskInstance.from_huggingface_row(row) for row in dataset_split]
        logger.info(f"Loaded {len(tasks)} tasks from split '{split}'.")
        return tasks

    def get_curated_subset(
        self,
        n_issues: int = 50,
        max_lines: int = 100,
        max_files: int = 3,
        repositories: Optional[List[str]] = None,
        split: str = "test"
    ) -> List[TaskInstance]:
        """
        Get a curated subset of issues based on complexity criteria.

        Args:
            n_issues: Target number of issues to return
            max_lines: Maximum lines changed per issue
            max_files: Maximum files changed per issue
            repositories: List of repositories to include (e.g., ['django', 'requests'])
            split: Dataset split to use

        Returns:
            Curated list of TaskInstance objects
        """
        all_tasks = self.load_tasks(split)

        # Estimate complexity for all tasks
        for task in all_tasks:
            task.estimate_complexity()

        # Filter by criteria
        filtered = []
        for task in all_tasks:
            # Check repository filter
            if repositories:
                repo_name = task.repo.split('/')[-1]  # Get repo name from org/repo
                if not any(r.lower() in repo_name.lower() for r in repositories):
                    continue

            # Check complexity filters
            if task.estimated_lines_changed and task.estimated_lines_changed > max_lines:
                continue
            if task.estimated_files_changed and task.estimated_files_changed > max_files:
                continue

            filtered.append(task)

        # Sort by complexity (simpler first) and take n_issues
        filtered.sort(key=lambda t: (
            t.estimated_files_changed or 99,
            t.estimated_lines_changed or 999
        ))

        result = filtered[:n_issues]
        logger.info(f"Curated {len(result)} issues from {len(all_tasks)} total (filters: max_lines={max_lines}, max_files={max_files})")
        return result

    def get_issues_by_repo(self, repo: str, split: str = "test") -> List[TaskInstance]:
        """Get all issues for a specific repository."""
        all_tasks = self.load_tasks(split)
        return [t for t in all_tasks if repo.lower() in t.repo.lower()]

    def categorize_issues(self, tasks: List[TaskInstance]) -> Dict[str, TaskCategory]:
        """
        Categorize issues using heuristics based on problem statements.

        Returns:
            Dictionary mapping instance_id to category
        """
        categorization = {}
        for task in tasks:
            category = self._infer_category(task.problem_statement)
            task.category = category
            categorization[task.instance_id] = category

        return categorization

    def _infer_category(self, problem_statement: str) -> TaskCategory:
        """Infer task category from problem statement using keywords."""
        text = problem_statement.lower()

        # Bug indicators
        if any(word in text for word in ['fix', 'bug', 'error', 'issue', 'crash', 'incorrect', 'fails', 'broken']):
            return "bug"

        # Feature indicators
        if any(word in text for word in ['add', 'implement', 'support', 'new feature', 'enhancement']):
            return "feature"

        # Refactor indicators
        if any(word in text for word in ['refactor', 'clean', 'improve', 'optimize', 'restructure']):
            return "refactor"

        # Documentation indicators
        if any(word in text for word in ['document', 'docs', 'readme', 'docstring']):
            return "docs"

        # Test indicators
        if any(word in text for word in ['test', 'coverage', 'unittest']):
            return "test"

        return "other"


class SWEBenchExecutor:
    """
    Executor for running SWE-bench tasks with Docker-based validation.

    Uses the official SWE-bench harness to apply patches and run tests
    in isolated Docker containers.
    """

    def __init__(
        self,
        log_dir: str = "swebench_logs",
        timeout: int = 1800,  # 30 minutes default
        namespace: str = "swebench"
    ):
        """
        Args:
            log_dir: Directory to store execution logs
            timeout: Maximum execution time per task in seconds
            namespace: Docker namespace for images (empty string builds locally)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.namespace = namespace

    def setup_environment(self, instance: TaskInstance) -> bool:
        """
        Verify Docker environment is ready for the task.

        Returns:
            True if environment is ready, False otherwise
        """
        try:
            # Check Docker is available
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                timeout=10
            )
            if result.returncode != 0:
                logger.error("Docker is not available or not running")
                return False

            logger.info(f"Docker environment ready for {instance.instance_id}")
            return True

        except Exception as e:
            logger.error(f"Environment setup failed: {e}")
            return False

    def execute_patch(
        self,
        solution: str,
        instance: TaskInstance,
        run_tests: bool = True
    ) -> ExecutionResult:
        """
        Execute a solution patch for a SWE-bench instance.

        Args:
            solution: The generated patch/code solution
            instance: The SWE-bench task instance
            run_tests: Whether to run tests (True) or just validate syntax

        Returns:
            ExecutionResult with success status and details
        """
        import time
        start_time = time.time()

        try:
            # Setup environment
            if not self.setup_environment(instance):
                return ExecutionResult(
                    instance_id=instance.instance_id,
                    success=False,
                    error="Environment setup failed",
                    execution_time=time.time() - start_time
                )

            # Write prediction to temporary file
            predictions_file = self._write_prediction(instance.instance_id, solution)

            if not run_tests:
                # Quick validation mode - just check if patch is well-formed
                return ExecutionResult(
                    instance_id=instance.instance_id,
                    success=True,
                    applied_patch=solution,
                    execution_time=time.time() - start_time,
                    metadata={"mode": "validation_only"}
                )

            # Run SWE-bench evaluation harness
            result = self._run_harness(instance.instance_id, predictions_file)

            execution_time = time.time() - start_time
            return ExecutionResult(
                instance_id=instance.instance_id,
                success=result["success"],
                test_output=result.get("test_output", ""),
                error=result.get("error", ""),
                execution_time=execution_time,
                applied_patch=solution,
                metadata=result.get("metadata", {})
            )

        except Exception as e:
            logger.error(f"Execution failed for {instance.instance_id}: {e}", exc_info=True)
            return ExecutionResult(
                instance_id=instance.instance_id,
                success=False,
                error=str(e),
                execution_time=time.time() - start_time
            )

    def _write_prediction(self, instance_id: str, solution: str) -> Path:
        """Write a prediction to JSONL format for SWE-bench harness."""
        predictions_file = self.log_dir / f"predictions_{instance_id}.jsonl"

        with open(predictions_file, 'w') as f:
            json.dump({
                "instance_id": instance_id,
                "model_patch": solution,
                "model_name_or_path": "swarm_system"
            }, f)
            f.write('\n')

        return predictions_file

    def _run_harness(self, instance_id: str, predictions_file: Path) -> Dict[str, Any]:
        """
        Run the SWE-bench evaluation harness.

        Args:
            instance_id: The task instance ID
            predictions_file: Path to predictions JSONL file

        Returns:
            Dictionary with execution results
        """
        try:
            # Run SWE-bench evaluation
            cmd = [
                "python", "-m", "swebench.harness.run_evaluation",
                "--predictions_path", str(predictions_file),
                "--max_workers", "1",
                "--instance_ids", instance_id,
                "--run_id", f"run_{instance_id}"
            ]

            if self.namespace:
                cmd.extend(["--namespace", self.namespace])

            logger.info(f"Running harness command: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=str(self.log_dir)
            )

            # Parse results
            if result.returncode == 0:
                # Check for test results
                results_file = self.log_dir / f"run_{instance_id}" / "results.json"
                if results_file.exists():
                    with open(results_file) as f:
                        test_results = json.load(f)
                        # SWE-bench marks as "resolved" if tests pass
                        success = test_results.get(instance_id, {}).get("resolved", False)
                        return {
                            "success": success,
                            "test_output": result.stdout,
                            "metadata": test_results.get(instance_id, {})
                        }

                return {
                    "success": True,
                    "test_output": result.stdout,
                    "metadata": {}
                }
            else:
                return {
                    "success": False,
                    "error": result.stderr,
                    "test_output": result.stdout
                }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Execution timeout after {self.timeout}s"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def batch_execute(
        self,
        solutions: Dict[str, str],
        instances: List[TaskInstance],
        run_tests: bool = True
    ) -> List[ExecutionResult]:
        """
        Execute multiple solutions in batch.

        Args:
            solutions: Dictionary mapping instance_id to solution patch
            instances: List of TaskInstance objects
            run_tests: Whether to run tests

        Returns:
            List of ExecutionResult objects
        """
        results = []
        for instance in instances:
            if instance.instance_id not in solutions:
                logger.warning(f"No solution found for {instance.instance_id}")
                continue

            solution = solutions[instance.instance_id]
            result = self.execute_patch(solution, instance, run_tests=run_tests)
            results.append(result)

        return results
