#!/usr/bin/env python3
"""Build a self-contained MBPP+ test set (run with .venv-data which has evalplus).
For each task: run the canonical solution on base+plus inputs to get expected
outputs, emit a check(candidate) test with repr-embedded asserts (tolerant compare
for float tasks via atol). Output: data/MbppPlus.jsonl with string-only fields
(JSON-safe), consumed at runtime by src/swarm/mbpp.py (no evalplus dep at runtime)."""
import json, math, traceback

from evalplus.data import get_mbpp_plus

# Explicit imports (NOT `import *`, which is illegal once the test is indented
# inside the executor's try-block). Mirrors EvalPlus's standard preamble enough
# to run canonical solutions that assume these are in scope.
PREAMBLE = (
    "import math, re, collections, itertools, heapq, bisect, functools, operator, string\n"
    "from collections import Counter, defaultdict, OrderedDict, deque\n"
    "from math import inf, sqrt, gcd, pi, floor, ceil, factorial\n"
    "from itertools import combinations, permutations, product, chain, groupby\n"
)

HARNESS_HEADER = PREAMBLE + '''def _eq(a, b, atol={atol}):
    if isinstance(a, float) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=1e-6, abs_tol=max(atol,1e-6))
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a)==len(b) and all(_eq(x, y) for x, y in zip(a, b))
    if isinstance(a, set) and isinstance(b, set):
        return a == b
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys()==b.keys() and all(_eq(a[k], b[k]) for k in a)
    try:
        return a == b
    except Exception:
        return False
def check(candidate):
'''

def build_test(entry, inputs, expected, atol):
    lines = [HARNESS_HEADER.format(atol=atol)]
    n = 0
    for inp, exp in zip(inputs, expected):
        if exp is _ERR:
            continue
        try:
            args = ", ".join(repr(a) for a in inp)
            lines.append(f"    assert _eq(candidate({args}), {exp!r})")
            n += 1
        except Exception:
            continue
    if n == 0:
        lines.append("    pass")
    return "\n".join(lines), n

_ERR = object()

def main():
    data = get_mbpp_plus()
    rows = []
    skipped = 0
    for tid, t in data.items():
        entry = t["entry_point"]
        canon = t["canonical_solution"]
        atol = t.get("atol", 0) or 0
        inputs = list(t.get("base_input", [])) + list(t.get("plus_input", []))
        # run canonical to get expected outputs
        ns = {}
        try:
            exec(PREAMBLE + canon, ns)
            fn = ns[entry]
        except Exception:
            skipped += 1
            continue
        expected = []
        for inp in inputs:
            try:
                expected.append(fn(*inp))
            except Exception:
                expected.append(_ERR)
        test_code, n = build_test(entry, inputs, expected, atol)
        if n == 0:
            skipped += 1
            continue
        rows.append({
            "task_id": tid,
            "prompt": t["prompt"],
            "entry_point": entry,
            "test_code": test_code,
            "n_tests": n,
            "canonical_solution": canon,
        })
    with open("data/MbppPlus.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"WROTE {len(rows)} tasks (skipped {skipped}); avg tests/task = "
          f"{sum(r['n_tests'] for r in rows)/max(len(rows),1):.1f}")

if __name__ == "__main__":
    main()
