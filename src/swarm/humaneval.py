"""HumanEval benchmark integration for swarm experiments."""

from typing import List, Dict, Optional
from datasets import load_dataset

from .types import SwarmTask, TaskType


# Manual task categorization for HumanEval problems
# Based on pilot-experiment.md categories: string, math, list, logic
TASK_CATEGORIZATION = {
    # String manipulation tasks
    "HumanEval/0": TaskType.STRING,   # has_close_elements
    "HumanEval/1": TaskType.STRING,   # separate_paren_groups
    "HumanEval/10": TaskType.STRING,  # is_palindrome
    "HumanEval/19": TaskType.STRING,  # sort_numbers
    "HumanEval/20": TaskType.LIST,    # find_closest_elements
    "HumanEval/22": TaskType.LIST,    # filter_integers
    "HumanEval/23": TaskType.MATH,    # strlen
    "HumanEval/25": TaskType.MATH,    # factorize
    "HumanEval/28": TaskType.STRING,  # concatenate
    "HumanEval/29": TaskType.LIST,    # filter_by_prefix

    # Math/arithmetic tasks
    "HumanEval/2": TaskType.MATH,     # truncate_number
    "HumanEval/4": TaskType.MATH,     # mean_absolute_deviation
    "HumanEval/5": TaskType.LIST,     # intersperse
    "HumanEval/6": TaskType.STRING,   # parse_nested_parens
    "HumanEval/7": TaskType.LIST,     # filter_by_substring
    "HumanEval/11": TaskType.STRING,  # string_xor
    "HumanEval/13": TaskType.MATH,    # greatest_common_divisor
    "HumanEval/15": TaskType.STRING,  # string_sequence
    "HumanEval/17": TaskType.STRING,  # parse_music
    "HumanEval/18": TaskType.MATH,    # how_many_times

    # List operations tasks
    "HumanEval/3": TaskType.LIST,     # below_zero
    "HumanEval/8": TaskType.MATH,     # sum_product
    "HumanEval/9": TaskType.LIST,     # rolling_max
    "HumanEval/12": TaskType.STRING,  # longest
    "HumanEval/14": TaskType.LIST,    # all_prefixes
    "HumanEval/21": TaskType.LIST,    # rescale_to_unit
    "HumanEval/24": TaskType.MATH,    # largest_divisor
    "HumanEval/26": TaskType.LIST,    # remove_duplicates
    "HumanEval/27": TaskType.STRING,  # flip_case
    "HumanEval/30": TaskType.LIST,    # get_positive

    # Logic/conditional tasks
    "HumanEval/16": TaskType.MATH,    # count_distinct_characters
    "HumanEval/31": TaskType.MATH,    # is_prime
    "HumanEval/32": TaskType.MATH,    # poly
    "HumanEval/33": TaskType.LIST,    # sort_third
    "HumanEval/34": TaskType.LIST,    # unique
    "HumanEval/35": TaskType.MATH,    # max_element
    "HumanEval/36": TaskType.MATH,    # fizz_buzz
    "HumanEval/37": TaskType.LIST,    # sort_even
    "HumanEval/38": TaskType.STRING,  # decode_cyclic
    "HumanEval/39": TaskType.MATH,    # prime_fib

    # Additional tasks for 50-problem pilot
    "HumanEval/40": TaskType.LIST,    # triples_sum_to_zero
    "HumanEval/41": TaskType.MATH,    # car_race_collision
    "HumanEval/42": TaskType.LIST,    # incr_list
    "HumanEval/43": TaskType.LIST,    # pairs_sum_to_zero
    "HumanEval/44": TaskType.MATH,    # change_base
    "HumanEval/45": TaskType.MATH,    # triangle_area
    "HumanEval/46": TaskType.MATH,    # fib4
    "HumanEval/47": TaskType.MATH,    # median
    "HumanEval/48": TaskType.LOGIC,   # is_palindrome
    "HumanEval/49": TaskType.MATH,    # modp
}


class HumanEvalLoader:
    """Loader for HumanEval benchmark tasks."""

    def __init__(self):
        """Initialize HumanEval loader."""
        self.dataset = None
        self.task_categorization = TASK_CATEGORIZATION

    def load_dataset(self) -> None:
        """Load HumanEval dataset from HuggingFace."""
        self.dataset = load_dataset("openai_humaneval", split="test")

    def get_task(self, task_id: str) -> Optional[SwarmTask]:
        """
        Get a single task by ID.

        Args:
            task_id: HumanEval task ID (e.g., "HumanEval/0")

        Returns:
            SwarmTask or None if not found
        """
        if self.dataset is None:
            self.load_dataset()

        # Find task in dataset
        for item in self.dataset:
            if item["task_id"] == task_id:
                return self._convert_to_swarm_task(item)

        return None

    def get_pilot_subset(self, n_tasks: int = 50) -> List[SwarmTask]:
        """
        Get pilot subset of tasks.

        Args:
            n_tasks: Number of tasks to include (default 50)

        Returns:
            List of SwarmTask objects
        """
        if self.dataset is None:
            self.load_dataset()

        tasks = []
        for task_id in sorted(self.task_categorization.keys())[:n_tasks]:
            task = self.get_task(task_id)
            if task is not None:
                tasks.append(task)

        return tasks

    def get_all_tasks(self) -> List[SwarmTask]:
        """
        Get all HumanEval tasks.

        Returns:
            List of all SwarmTask objects
        """
        if self.dataset is None:
            self.load_dataset()

        tasks = []
        for item in self.dataset:
            task_id = item["task_id"]
            # Use categorization if available, otherwise default to LOGIC
            task_type = self.task_categorization.get(task_id, TaskType.LOGIC)
            tasks.append(self._convert_to_swarm_task(item, task_type))

        return tasks

    def get_tasks_from_json(self, n_tasks: Optional[int] = None) -> List[SwarmTask]:
        """
        Get tasks using categories from data/humaneval_categories_full.json.

        Args:
            n_tasks: Number of tasks to return (None = all 164)
            
        Returns:
            List of SwarmTask objects with categories from JSON file
        """
        import json
        from pathlib import Path
        
        # Load categories from JSON
        json_path = Path(__file__).parent.parent.parent / "data" / "humaneval_categories_full.json"
        if not json_path.exists():
            raise FileNotFoundError(
                f"Categories file not found: {json_path}\n"
                "Run: python scripts/categorize_tasks.py"
            )
        
        with open(json_path, 'r') as f:
            json_categories = json.load(f)
        
        # Convert string categories to TaskType enum
        full_categorization = {}
        for task_id, category in json_categories.items():
            try:
                full_categorization[task_id] = TaskType(category)
            except ValueError:
                full_categorization[task_id] = TaskType.LOGIC
        
        if self.dataset is None:
            self.load_dataset()
        
        # Get tasks with JSON-based categorization
        tasks = []
        for item in self.dataset:
            task_id = item["task_id"]
            task_type = full_categorization.get(task_id, TaskType.LOGIC)
            tasks.append(self._convert_to_swarm_task(item, task_type))
        
        # Sort by task ID number
        tasks.sort(key=lambda t: int(t.id.split("/")[1]))
        
        # Limit to n_tasks if specified
        if n_tasks is not None and n_tasks < len(tasks):
            tasks = tasks[:n_tasks]
        
        return tasks

    def _convert_to_swarm_task(
        self,
        item: Dict,
        task_type: Optional[TaskType] = None
    ) -> SwarmTask:
        """
        Convert HumanEval dataset item to SwarmTask.

        Args:
            item: Dataset item
            task_type: Optional task type override

        Returns:
            SwarmTask object
        """
        task_id = item["task_id"]

        # Get task type from categorization or use provided
        if task_type is None:
            task_type = self.task_categorization.get(task_id, TaskType.LOGIC)

        return SwarmTask(
            id=task_id,
            task_type=task_type,
            problem=item["prompt"],
            test_code=item["test"],
            entry_point=item["entry_point"],
            canonical_solution=item.get("canonical_solution")
        )

    def get_tasks_by_type(self, task_type: TaskType) -> List[SwarmTask]:
        """
        Get all tasks of a specific type.

        Args:
            task_type: Type of tasks to retrieve

        Returns:
            List of SwarmTask objects
        """
        if self.dataset is None:
            self.load_dataset()

        tasks = []
        for task_id, cat_type in self.task_categorization.items():
            if cat_type == task_type:
                task = self.get_task(task_id)
                if task is not None:
                    tasks.append(task)

        return tasks

    def get_categorization_stats(self) -> Dict[str, int]:
        """
        Get statistics on task categorization.

        Returns:
            Dictionary mapping task type to count
        """
        stats = {task_type.value: 0 for task_type in TaskType}

        for task_type in self.task_categorization.values():
            stats[task_type.value] += 1

        return stats


def expand_task_categorization(loader: HumanEvalLoader) -> Dict[str, TaskType]:
    """
    Helper to expand categorization to all 164 HumanEval tasks.

    This uses heuristics based on problem description.
    For the pilot (50 tasks), manual categorization is preferred.

    Args:
        loader: HumanEvalLoader instance

    Returns:
        Extended categorization mapping
    """
    if loader.dataset is None:
        loader.load_dataset()

    categorization = dict(TASK_CATEGORIZATION)

    for item in loader.dataset:
        task_id = item["task_id"]

        # Skip if already categorized
        if task_id in categorization:
            continue

        # Simple heuristics based on prompt
        prompt = item["prompt"].lower()

        if any(kw in prompt for kw in ["string", "char", "letter", "word", "parse"]):
            categorization[task_id] = TaskType.STRING
        elif any(kw in prompt for kw in ["sum", "product", "number", "prime", "digit"]):
            categorization[task_id] = TaskType.MATH
        elif any(kw in prompt for kw in ["list", "array", "sort", "filter"]):
            categorization[task_id] = TaskType.LIST
        else:
            categorization[task_id] = TaskType.LOGIC

    return categorization
