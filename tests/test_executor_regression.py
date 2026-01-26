#!/usr/bin/env python3
"""
Regression tests for HumanEval executor.

CRITICAL: These tests ensure the executor actually runs assertions.
See docs/POSTMORTEM_HUMANEVAL_BUG.md for context.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from swarm.executor import HumanEvalExecutor


def test_wrong_solution_must_fail():
    """
    CRITICAL TEST: A wrong solution MUST fail.
    
    If this test fails, assertions are not being executed!
    See: docs/POSTMORTEM_HUMANEVAL_BUG.md
    """
    executor = HumanEvalExecutor(timeout=5)
    
    # HumanEval/0 test code (has_close_elements)
    test_code = """
METADATA = {'author': 'test'}

def check(candidate):
    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True
    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False
"""
    
    # Deliberately WRONG solution
    wrong_solution = """
def has_close_elements(numbers, threshold):
    return False  # Always returns False - obviously wrong
"""
    
    result = executor.execute_humaneval(
        wrong_solution, 
        test_code, 
        "has_close_elements"
    )
    
    assert not result.success, (
        "CRITICAL BUG: Wrong solution passed!\n"
        "Assertions are NOT being executed.\n"
        "See: docs/POSTMORTEM_HUMANEVAL_BUG.md"
    )
    print("✓ test_wrong_solution_must_fail PASSED")


def test_correct_solution_must_pass():
    """Ensure correct solutions pass."""
    executor = HumanEvalExecutor(timeout=5)
    
    test_code = """
METADATA = {'author': 'test'}

def check(candidate):
    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True
    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False
"""
    
    correct_solution = """
def has_close_elements(numbers, threshold):
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if abs(numbers[i] - numbers[j]) < threshold:
                return True
    return False
"""
    
    result = executor.execute_humaneval(
        correct_solution,
        test_code,
        "has_close_elements"
    )
    
    assert result.success, f"Correct solution failed: {result.error_message}"
    print("✓ test_correct_solution_must_pass PASSED")


def test_syntax_error_must_fail():
    """Ensure syntax errors are caught."""
    executor = HumanEvalExecutor(timeout=5)
    
    test_code = """
def check(candidate):
    assert candidate(1) == 1
"""
    
    # Syntax error
    broken_solution = """
def foo(x)
    return x  # Missing colon
"""
    
    result = executor.execute_humaneval(broken_solution, test_code, "foo")
    
    assert not result.success, "Syntax error should fail"
    print("✓ test_syntax_error_must_fail PASSED")


def test_timeout_must_fail():
    """Ensure infinite loops are caught."""
    executor = HumanEvalExecutor(timeout=2)
    
    test_code = """
def check(candidate):
    assert candidate() == 1
"""
    
    # Infinite loop
    infinite_solution = """
def foo():
    while True:
        pass
"""
    
    result = executor.execute_humaneval(infinite_solution, test_code, "foo")
    
    assert not result.success, "Timeout should fail"
    assert "timeout" in result.error_message.lower(), "Should mention timeout"
    print("✓ test_timeout_must_fail PASSED")


if __name__ == "__main__":
    print("=" * 60)
    print("EXECUTOR REGRESSION TESTS")
    print("=" * 60)
    print()
    
    try:
        test_wrong_solution_must_fail()  # MOST CRITICAL
        test_correct_solution_must_pass()
        test_syntax_error_must_fail()
        test_timeout_must_fail()
        
        print()
        print("=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
    except AssertionError as e:
        print()
        print("=" * 60)
        print(f"TEST FAILED: {e}")
        print("=" * 60)
        sys.exit(1)
