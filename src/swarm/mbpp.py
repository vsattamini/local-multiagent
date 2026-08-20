"""MBPP+ (EvalPlus) loader for the swarm framework.

Reads the vendored, self-contained data/MbppPlus.jsonl (built by
scripts/build_mbpp_data.py: each task carries a check(candidate) harness with
expected outputs precomputed from the canonical solution). Categorizes each task
into the same 4 task types as HumanEval (string/math/list/logic) via a keyword
heuristic over the prompt, so the specialization metrics are directly comparable.
No evalplus dependency at runtime.
"""
import json
import re
from pathlib import Path
from typing import List, Optional

from .types import SwarmTask, TaskType

_DATA = Path(__file__).parent.parent.parent / "data" / "MbppPlus.jsonl"
_CATS = Path(__file__).parent.parent.parent / "data" / "mbpp_categories_full.json"

# keyword heuristic mirrors humaneval.expand_task_categorization
_KW = {
    TaskType.STRING: ["string", "char", "letter", "word", "substring", "vowel",
                      "uppercase", "lowercase", "concat", "text", "alphabet"],
    TaskType.MATH:   ["sum", "product", "number", "prime", "digit", "factor",
                      "fibonacci", "divisor", "even", "odd", "square", "multiple",
                      "arithmetic", "average", "integer", "modulo"],
    TaskType.LIST:   ["list", "array", "sort", "filter", "tuple", "element",
                      "sequence", "index", "subarray", "matrix", "dictionary"],
}


def _categorize(prompt: str) -> TaskType:
    p = prompt.lower()
    scores = {t: sum(p.count(k) for k in kws) for t, kws in _KW.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else TaskType.LOGIC


def _task_num(tid: str) -> int:
    m = re.search(r"(\d+)", tid)
    return int(m.group(1)) if m else 0


class MbppPlusLoader:
    """API-compatible with HumanEvalLoader for the bits experiment.py uses."""

    def __init__(self):
        self.dataset = None
        self._cats = None

    def load_dataset(self) -> None:
        if not _DATA.exists():
            raise FileNotFoundError(
                f"{_DATA} not found. Build it with: .venv-data/bin/python scripts/build_mbpp_data.py")
        self.dataset = [json.loads(l) for l in open(_DATA) if l.strip()]
        if _CATS.exists():
            raw = json.load(open(_CATS))
            self._cats = {k: TaskType(v) for k, v in raw.items()}

    def _type_for(self, row) -> TaskType:
        if self._cats and row["task_id"] in self._cats:
            return self._cats[row["task_id"]]
        return _categorize(row["prompt"])

    def get_tasks_from_json(self, n_tasks: Optional[int] = None) -> List[SwarmTask]:
        if self.dataset is None:
            self.load_dataset()
        tasks = [
            SwarmTask(
                id=row["task_id"],
                task_type=self._type_for(row),
                problem=row["prompt"],
                test_code=row["test_code"],
                entry_point=row["entry_point"],
                canonical_solution=row.get("canonical_solution"),
            )
            for row in self.dataset
        ]
        tasks.sort(key=lambda t: _task_num(t.id))
        if n_tasks is not None and n_tasks < len(tasks):
            tasks = tasks[:n_tasks]
        return tasks

    # pilot path (unused for full runs) falls back to the same full set
    def get_pilot_subset(self, n_tasks: int = 50) -> List[SwarmTask]:
        return self.get_tasks_from_json(n_tasks=n_tasks)
