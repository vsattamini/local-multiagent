"""Safe code execution for testing generated solutions."""

import subprocess
import tempfile
import time
import os
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path for utils import
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.validator import validate_python_code
from swarm.types import ExecutionResult


class CodeExecutor:
    """
    Execute generated code against test cases in a sandboxed environment.

    Uses subprocess isolation with timeout for safety.
    """

    def __init__(self, timeout: int = 5):
        """
        Initialize code executor.

        Args:
            timeout: Maximum execution time in seconds
        """
        self.timeout = timeout

    def execute(
        self,
        code: str,
        test_code: str,
        entry_point: Optional[str] = None
    ) -> ExecutionResult:
        """
        Run code against tests in sandboxed environment.

        Args:
            code: Generated solution code
            test_code: Test cases to run
            entry_point: Entry point function name (for HumanEval)

        Returns:
            ExecutionResult with success status and details
        """
        start_time = time.time()

        # Validate syntax first
        validation = validate_python_code(code)
        if not validation["valid"]:
            return ExecutionResult(
                success=False,
                stderr=validation["error"],
                error_message=f"Syntax error: {validation['error']}",
                execution_time=time.time() - start_time
            )

        # Create temporary file for execution
        try:
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.py',
                delete=False
            ) as f:
                temp_file = f.name

                # Write solution code
                f.write(code)
                f.write("\n\n")

                # Write test code
                f.write(test_code)
                f.write("\n")

            # Execute in subprocess
            result = subprocess.run(
                ["python", temp_file],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )

            execution_time = time.time() - start_time

            # Check if execution was successful
            success = result.returncode == 0

            return ExecutionResult(
                success=success,
                stdout=result.stdout,
                stderr=result.stderr,
                execution_time=execution_time,
                error_message=None if success else result.stderr
            )

        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                stderr="Execution timeout",
                error_message=f"Code execution exceeded {self.timeout}s timeout",
                execution_time=self.timeout
            )

        except Exception as e:
            return ExecutionResult(
                success=False,
                stderr=str(e),
                error_message=f"Execution error: {str(e)}",
                execution_time=time.time() - start_time
            )

        finally:
            # Clean up temporary file
            try:
                if 'temp_file' in locals():
                    os.unlink(temp_file)
            except Exception:
                pass

    def execute_with_assertion_check(
        self,
        code: str,
        test_code: str,
        entry_point: Optional[str] = None
    ) -> ExecutionResult:
        """
        Execute code and explicitly check for assertion errors.

        This is useful for HumanEval where tests use assert statements.

        Args:
            code: Generated solution code
            test_code: Test cases with assertions
            entry_point: Entry point function name

        Returns:
            ExecutionResult
        """
        # Wrap test code to catch assertion errors explicitly
        wrapped_test = f"""
import sys
import traceback

try:
{self._indent_code(test_code, 4)}
    print("ALL_TESTS_PASSED")
except AssertionError as e:
    print(f"ASSERTION_FAILED: {{e}}", file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print(f"RUNTIME_ERROR: {{e}}", file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)
"""

        result = self.execute(code, wrapped_test, entry_point)

        # Check for success marker
        if result.success and "ALL_TESTS_PASSED" in result.stdout:
            return result

        return result

    def _indent_code(self, code: str, spaces: int) -> str:
        """Indent code block by specified spaces."""
        indent = " " * spaces
        return "\n".join(indent + line for line in code.splitlines())


class HumanEvalExecutor(CodeExecutor):
    """
    Specialized executor for HumanEval benchmark.

    HumanEval has specific format requirements and test structure.
    """

    def execute_humaneval(
        self,
        solution_code: str,
        test_code: str,
        entry_point: str
    ) -> ExecutionResult:
        """
        Execute HumanEval solution.

        Args:
            solution_code: Generated function implementation
            test_code: HumanEval test assertions
            entry_point: Function name to test

        Returns:
            ExecutionResult
        """
        # Clean solution code (remove markdown if present)
        clean_code = self._clean_solution(solution_code)

        # HumanEval test_code already contains:
        # - METADATA dict
        # - def check(candidate): with assertions
        # We just need to call check(entry_point) after defining it
        
        test_with_call = f"""
{test_code}

check({entry_point})
"""

        return self.execute_with_assertion_check(
            clean_code,
            test_with_call,
            entry_point
        )

    def _clean_solution(self, code: str) -> str:
        """
        Clean solution code by removing markdown fencing.

        Args:
            code: Raw solution code

        Returns:
            Cleaned code
        """
        if "```" not in code:
            return code

        lines = code.splitlines()
        content = []
        in_block = False

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("```"):
                in_block = not in_block
                continue
            if in_block or not any(stripped.startswith(x) for x in ["```"]):
                content.append(line)

        return "\n".join(content) if content else code
