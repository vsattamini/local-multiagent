from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from datasets import load_dataset
import logging

logger = logging.getLogger(__name__)

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
