import pytest
import asyncio
from unittest.mock import MagicMock, patch
from src.evaluation.swebench import SWEBenchLoader, TaskInstance

class TestSWEBenchLoader:
    @pytest.fixture
    def mock_dataset(self):
        return [
            {
                "instance_id": "test-1",
                "problem_statement": "Fix the bug",
                "repo": "test/repo",
                "base_commit": "abc1234",
                "patch": "diff --git...",
                "test_patch": "diff --git...",
                "version": "1.0",
                "environment_setup_commit": "xyz789"
            }
        ]

    def test_loader_initialization(self):
        loader = SWEBenchLoader()
        assert loader is not None

    @patch("src.evaluation.swebench.load_dataset")
    def test_load_tasks(self, mock_load_dataset, mock_dataset):
        # Mock load_dataset to return a dict-like object (DatasetDict)
        mock_load_dataset.return_value = {"dev": mock_dataset}
        
        loader = SWEBenchLoader()
        tasks = loader.load_tasks(split="dev")
        
        assert len(tasks) == 1
        assert isinstance(tasks[0], TaskInstance)
        assert tasks[0].instance_id == "test-1"
        assert tasks[0].problem_statement == "Fix the bug"

class TestBenchmarkRunner:
    @pytest.mark.asyncio
    async def test_run_task(self):
        # Mock Coordinator
        mock_coordinator = MagicMock()
        # process_task is async, so we need to mock it as an async function or return a Future
        future = asyncio.Future()
        future.set_result("diff --git a/file.py b/file.py...")
        mock_coordinator.process_task.return_value = future

        from src.evaluation.swebench import TaskInstance
        from src.evaluation.runner import BenchmarkRunner

        instance = TaskInstance(
            instance_id="test-1",
            problem_statement="Fix it",
            repo="repo",
            base_commit="abc",
            patch="",
            test_patch="",
            version="1",
            environment_setup_commit="xyz"
        )
        
        runner = BenchmarkRunner(coordinator=mock_coordinator)
        result = await runner.run_task(instance)
        
        assert result["status"] == "success"
        assert result["instance_id"] == "test-1"
        assert result["patch"] == "diff --git a/file.py b/file.py..."
        mock_coordinator.process_task.assert_called_once()
