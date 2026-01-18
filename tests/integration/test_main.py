import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import sys
from src.cli import main

class TestCLI:
    @patch('src.cli.BenchmarkRunner')
    @patch('src.cli.Coordinator')
    @patch('src.cli.SWEBenchLoader')
    def test_evaluate_command(self, MockLoader, MockCoordinator, MockRunner):
        # Setup mocks
        mock_runner_instance = MockRunner.return_value
        mock_runner_instance.run_batch = AsyncMock()
        
        mock_loader_instance = MockLoader.return_value
        mock_loader_instance.load_tasks.return_value = ["task1"]

        # Mock sys.argv
        with patch('sys.argv', ['cli.py', 'evaluate', '--split', 'dev', '--limit', '1']):
             # We need to run the async main inside the sync test, 
             # but cli.main() likely handles the event loop or is synchronous and calls async stuff.
             # If cli.main() starts the loop, we can just call it.
             # If cli.main() is async, we await it.
             # Usually CLIs are sync entry points that run async code.
             
             # Assuming main() handles the loop:
             main()
             
             MockRunner.assert_called()
             mock_runner_instance.run_batch.assert_called()
