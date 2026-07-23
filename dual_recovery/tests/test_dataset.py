"""Tests for flattening diagnosis traces into training rows."""

import csv
import json
from pathlib import Path
import tempfile
import unittest

from cruzr_sim.experiments.dataset import export_dataset


class DatasetExportTest(unittest.TestCase):
    def test_fault_label_starts_at_injection_event(self):
        records = [
            {"record_type": "observation", "payload": {
                "timestamp": 0.0, "stage": "approach",
                "object_position": [0, 0, 0.8], "tool_position": [0, 0, 0.9],
            }},
            {"record_type": "event", "payload": {
                "name": "fault_injected", "scenario": "slip",
            }},
            {"record_type": "observation", "payload": {
                "timestamp": 1.0, "stage": "lift",
                "object_position": [0, 0, 0.7], "tool_position": [0, 0, 0.9],
                "metadata": {
                    "visual_confidence": 0.91,
                    "visual_position": [-1.2, 0.5, 0.8],
                    "visual_rpy": [0.0, 0.0, 0.3],
                    "visual_bbox": [10, 20, 30, 40],
                },
            }},
        ]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "episode.jsonl"
            output = Path(directory) / "dataset.csv"
            source.write_text(
                "\n".join(json.dumps(item) for item in records), encoding="utf-8"
            )
            self.assertEqual(export_dataset([source], output), 2)
            with output.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
        self.assertEqual([row["label"] for row in rows], ["normal", "slip"])
        self.assertEqual(rows[1]["visual_detected"], "1")
        self.assertEqual(rows[1]["visual_x"], "-1.2")
        self.assertEqual(rows[1]["visual_yaw"], "0.3")


if __name__ == "__main__":
    unittest.main()
