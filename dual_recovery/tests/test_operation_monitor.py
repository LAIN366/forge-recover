"""Tests for append-only operation logging behavior."""

import json
from pathlib import Path
import tempfile
import unittest

from cruzr_sim.diagnosis.operation_monitor import OperationMonitor


class OperationMonitorTest(unittest.TestCase):
    def test_append_mode_preserves_records_before_failure_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode.jsonl"
            monitor = OperationMonitor(path)
            monitor.record_event("episode_started", simulation_time=0.0)
            monitor.close()

            failure_monitor = OperationMonitor(path, append=True)
            failure_monitor.record_summary(
                success=False,
                terminal_stage="exception",
                failure_reason="exception",
            )
            failure_monitor.close()

            records = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["record_type"], "event")
        self.assertEqual(records[1]["record_type"], "summary")


if __name__ == "__main__":
    unittest.main()
