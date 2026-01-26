"""Experiment logging for swarm system."""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np

from .agent import SwarmAgent
from .types import SwarmTask, ExecutionResult, TaskType


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, (np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
             return bool(obj)
        return json.JSONEncoder.default(self, obj)

class ExperimentLogger:
    """
    Logger for swarm experiments using JSONL format.

    Logs:
    - Per-task results (task_log.jsonl)
    - Periodic snapshots (snapshots.jsonl)
    - Final metrics (final_metrics.json)
    """

    def __init__(self, output_dir: str):
        """
        Initialize experiment logger.

        Args:
            output_dir: Directory for output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Output files
        self.task_log_file = self.output_dir / "task_log.jsonl"
        self.snapshot_file = self.output_dir / "snapshots.jsonl"
        self.final_metrics_file = self.output_dir / "final_metrics.json"

        # In-memory task log for metrics computation
        self.task_log: List[Dict] = []

        # Experiment start time
        self.start_time = time.time()

        # Metadata
        self.metadata = {
            "start_time": datetime.now().isoformat(),
            "output_dir": str(output_dir)
        }

    def log_task(
        self,
        task: SwarmTask,
        agent: SwarmAgent,
        result: ExecutionResult,
        solution: str,
        timestamp: Optional[float] = None
    ) -> None:
        """
        Log a single task execution.

        Args:
            task: Task that was executed
            agent: Agent that handled the task
            result: Execution result
            solution: Generated solution code
            timestamp: Optional timestamp (uses current time if None)
        """
        if timestamp is None:
            timestamp = time.time()

        entry = {
            "task_id": task.id,
            "task_type": task.task_type.value,
            "agent_id": agent.agent_id,
            "timestamp": timestamp,
            "elapsed_time": timestamp - self.start_time,
            "success": result.success,
            "execution_time": result.execution_time,
            "solution": solution,
            "error_message": result.error_message,
            "context_size": len(agent.context_buffer),
            "agent_total_tasks": agent.total_tasks(),
            "agent_success_rate": agent.overall_success_rate()
        }

        # Add to in-memory log
        self.task_log.append(entry)

        # Append to JSONL file
        with open(self.task_log_file, "a") as f:
            f.write(json.dumps(entry, cls=NumpyEncoder) + "\n")

    def log_snapshot(self, snapshot: Dict) -> None:
        """
        Log periodic snapshot of system state.

        Args:
            snapshot: Snapshot dictionary from MetricsEngine
        """
        # Add timestamp
        snapshot["timestamp"] = time.time()
        snapshot["elapsed_time"] = time.time() - self.start_time

        # Append to JSONL file
        with open(self.snapshot_file, "a") as f:
            f.write(json.dumps(snapshot, cls=NumpyEncoder) + "\n")

    def log_final_metrics(self, metrics: Dict) -> None:
        """
        Save final experiment metrics.

        Args:
            metrics: Final metrics dictionary
        """
        final_data = {
            **self.metadata,
            "end_time": datetime.now().isoformat(),
            "total_elapsed_time": time.time() - self.start_time,
            "total_tasks": len(self.task_log),
            "metrics": metrics
        }

        with open(self.final_metrics_file, "w") as f:
            json.dump(final_data, f, indent=2, cls=NumpyEncoder)

    def get_task_log(self) -> List[Dict]:
        """Get in-memory task log for metrics computation."""
        return self.task_log

    def finalize(self) -> None:
        """Mark experiment as complete."""
        # Write completion marker
        completion_file = self.output_dir / "COMPLETE"
        completion_file.write_text(datetime.now().isoformat())

    def load_task_log(self) -> List[Dict]:
        """
        Load task log from JSONL file.

        Returns:
            List of task entries
        """
        if not self.task_log_file.exists():
            return []

        entries = []
        with open(self.task_log_file, "r") as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))
        return entries

    def load_snapshots(self) -> List[Dict]:
        """
        Load snapshots from JSONL file.

        Returns:
            List of snapshot entries
        """
        if not self.snapshot_file.exists():
            return []

        snapshots = []
        with open(self.snapshot_file, "r") as f:
            for line in f:
                if line.strip():
                    snapshots.append(json.loads(line))
        return snapshots

    def get_summary_stats(self) -> Dict:
        """
        Get summary statistics of experiment.

        Returns:
            Dictionary with summary statistics
        """
        if not self.task_log:
            return {
                "total_tasks": 0,
                "successful_tasks": 0,
                "pass_at_1": 0.0,
                "avg_execution_time": 0.0,
                "tasks_by_type": {},
                "tasks_by_agent": {}
            }

        successful = sum(1 for t in self.task_log if t["success"])
        total = len(self.task_log)

        # Group by task type
        tasks_by_type = {}
        for entry in self.task_log:
            task_type = entry["task_type"]
            if task_type not in tasks_by_type:
                tasks_by_type[task_type] = {"total": 0, "successful": 0}
            tasks_by_type[task_type]["total"] += 1
            if entry["success"]:
                tasks_by_type[task_type]["successful"] += 1

        # Group by agent
        tasks_by_agent = {}
        for entry in self.task_log:
            agent_id = entry["agent_id"]
            if agent_id not in tasks_by_agent:
                tasks_by_agent[agent_id] = {"total": 0, "successful": 0}
            tasks_by_agent[agent_id]["total"] += 1
            if entry["success"]:
                tasks_by_agent[agent_id]["successful"] += 1

        return {
            "total_tasks": total,
            "successful_tasks": successful,
            "pass_at_1": successful / total if total > 0 else 0.0,
            "avg_execution_time": sum(t["execution_time"] for t in self.task_log) / total,
            "tasks_by_type": tasks_by_type,
            "tasks_by_agent": tasks_by_agent
        }

    def export_config(self, config: Dict) -> None:
        """
        Export experiment configuration.

        Args:
            config: Configuration dictionary
        """
        config_file = self.output_dir / "config.json"
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)
