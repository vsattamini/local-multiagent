# Critical Bug Postmortem: HumanEval Test Execution Failure

**Date**: 2026-01-24
**Severity**: CRITICAL (invalidated all experiment results)
**Status**: FIXED

---

## Summary

Tests appeared to pass (96% Pass@1) when they were **never actually executed**. A wrapper function nesting bug caused HumanEval assertions to be defined but never called.

---

## The Bug

### Root Cause

HumanEval test code has this structure:
```python
METADATA = {...}

def check(candidate):
    assert candidate([1.0, 2.0], 0.5) == True
    assert candidate([1.0, 2.0], 0.05) == False
```

The executor wrapped this in ANOTHER `def check(candidate):`:
```python
def check(candidate):           # OUTER - executor added this
    METADATA = {...}
    def check(candidate):       # INNER - from HumanEval
        assert ...              # Never called!

check(solution_function)        # Calls OUTER, which does nothing
```

**Result**: The outer `check` was called, but it only *defined* the inner function without calling it. Assertions never ran. All tests "passed".

### Symptoms

- 96% Pass@1 (expected ~30% for 1.5B model)
- Wrong solutions marked as correct
- No assertion errors in logs

### Detection

Manually testing a deliberately wrong solution:
```python
def has_close_elements(numbers, threshold):
    return False  # Obviously wrong
```
This returned `success=True`, revealing the bug.

---

## The Fix

**File**: `src/swarm/executor.py` → `execute_humaneval()`

**Before** (BROKEN):
```python
return self.execute_with_assertion_check(
    clean_code,
    f"""
def check(candidate):
{self._indent_code(test_code, 4)}  # WRONG: test_code already has check()

check({entry_point})
""",
    entry_point
)
```

**After** (FIXED):
```python
test_with_call = f"""
{test_code}

check({entry_point})
"""
return self.execute_with_assertion_check(clean_code, test_with_call, entry_point)
```

---

## Prevention Guidelines

### 1. Understand Benchmark Data Formats

Before writing test execution code:
- Print raw test data to understand its structure
- Check if tests are self-contained or need wrapping
- Read benchmark documentation

### 2. Always Verify with Known-Bad Inputs

Create sanity checks with deliberately wrong solutions:
```python
def test_executor_rejects_wrong_solutions():
    wrong_solution = "def foo(): return None"
    result = executor.execute(wrong_solution, tests)
    assert not result.success, "Wrong solution should fail!"
```

### 3. Validate Against Published Benchmarks

Compare results with published scores:
- Qwen2.5-Coder-1.5B: ~30-40% Pass@1 on HumanEval
- If you get 90%+, something is wrong

### 4. Log Execution Details

Log the actual code being executed so bugs are visible:
```python
logger.debug(f"Executing:\n{combined_code}")
```

### 5. Test the Test Infrastructure

Before running experiments:
```bash
python scripts/smoke_test.py  # Should include wrong-solution tests
```

---

## Impact

- All Phase 2 experiment results from 2026-01-24 are **INVALID**
- Results must be regenerated after fix
- ~4 hours of GPU compute wasted

---

## Checklist for Future Benchmark Integration

- [ ] Print raw test data format first
- [ ] Check if test code is self-contained
- [ ] Test with a deliberately WRONG solution (must fail)
- [ ] Test with a deliberately CORRECT solution (must pass)
- [ ] Compare initial results with published benchmarks
- [ ] Add regression test to prevent this bug from returning
