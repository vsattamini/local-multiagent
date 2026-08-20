#!/usr/bin/env python3
"""
Independent regression test for the HumanEval executor (the 96%-false-positive
bug postmortem). Verifies, WITHOUT any model, that:
  - a deliberately WRONG solution is rejected (success == False)
  - a correct solution is accepted (success == True)
  - a solution that raises is rejected
Uses a real HumanEval-format test block (with its own `check(candidate)`).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from swarm.executor import HumanEvalExecutor

ex = HumanEvalExecutor(timeout=5)

ENTRY = "has_close_elements"
TEST = '''

METADATA = {}


def check(candidate):
    assert candidate([1.0, 2.0, 3.0], 0.5) == False
    assert candidate([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True
    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True
    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False
'''

CORRECT = '''
from typing import List
def has_close_elements(numbers, threshold):
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if abs(numbers[i] - numbers[j]) < threshold:
                return True
    return False
'''

WRONG_CONSTANT = '''
def has_close_elements(numbers, threshold):
    return False
'''

WRONG_INVERTED = '''
def has_close_elements(numbers, threshold):
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if abs(numbers[i] - numbers[j]) < threshold:
                return False
    return True
'''

RAISES = '''
def has_close_elements(numbers, threshold):
    raise ValueError("boom")
'''

cases = [
    ("correct",        CORRECT,        True),
    ("wrong_constant", WRONG_CONSTANT, False),
    ("wrong_inverted", WRONG_INVERTED, False),
    ("raises",         RAISES,         False),
]

all_ok = True
for name, code, expected in cases:
    r = ex.execute_humaneval(code, TEST, ENTRY)
    ok = (r.success == expected)
    all_ok &= ok
    print(f"[{'PASS' if ok else 'FAIL'}] {name:<15} expected success={expected!s:<5} got success={r.success!s:<5}"
          + ("" if r.success else f"  err={ (r.error_message or '')[:60] }"))

print("\n" + ("ALL EXECUTOR SANITY CHECKS PASSED — bug fix verified." if all_ok
              else "EXECUTOR SANITY CHECKS FAILED — investigate!"))
sys.exit(0 if all_ok else 1)
