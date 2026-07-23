"""Tests for experiment log metric extraction."""

import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from cruzr_sim.experiments.metrics import (
    aggregate_episode_metrics,
    extract_episode_metrics,
)
from cruzr_sim.experiments.evaluation import evaluate_diagnosis
from cruzr_sim.experiments.report import generate_markdown_report


class MetricsTest(unittest.TestCase):
    def test_missing_log_becomes_failed_episode(self):
        metrics = extract_episode_metrics(
            Path("does-not-exist.jsonl"),
            "missing", "no_recovery", "slip", 1, 124,
        )
        self.assertFalse(metrics.success)
        self.assertEqual(metrics.exit_code, 124)

    def test_extracts_latency_and_summary(self):
        records = [
            {"record_type": "event", "payload": {
                "name": "fault_injected", "simulation_time": 5.0,
            }},
            {"record_type": "diagnosis", "payload": {
                "timestamp": 5.4, "primary_fault": "grasp_slip",
            }},
            {"record_type": "recovery_plan", "payload": {"plan_id": "one"}},
            {"record_type": "summary", "payload": {
                "success": True, "terminal_stage": "hold",
                "simulation_duration": 14.0, "cube_height_gain": 0.1,
                "recovery_count": 1,
            }},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode.jsonl"
            path.write_text(
                "\n".join(json.dumps(item) for item in records), encoding="utf-8"
            )
            metrics = extract_episode_metrics(
                path, "run", "active_case", "slip", 7, 0
            )
        self.assertTrue(metrics.success)
        self.assertEqual(metrics.policy, "active_case")
        self.assertAlmostEqual(metrics.diagnosis_latency, 0.4)
        self.assertEqual(metrics.recovery_plan_count, 1)
        self.assertEqual(metrics.confirmed_fault, "unknown")
        grouped = aggregate_episode_metrics([metrics])
        self.assertEqual(grouped[0]["policy"], "active_case")
        self.assertEqual(grouped[0]["success_rate"], 1.0)
        confirmed = replace(metrics, confirmed_fault="grasp_slip")
        evaluation, confusion = evaluate_diagnosis([confirmed])
        self.assertEqual(evaluation[0]["classification_accuracy"], 1.0)
        self.assertEqual(confusion[0]["expected"], "grasp_slip")
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "report.md"
            generate_markdown_report([confirmed], report_path)
            self.assertIn("Diagnosis Metrics", report_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
