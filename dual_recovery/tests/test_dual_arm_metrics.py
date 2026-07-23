"""Tests for dual-arm episode and aggregate metrics."""

import json
import math
from pathlib import Path
import tempfile
import unittest

from cruzr_sim.experiments.dual_arm_metrics import (
    aggregate_dual_arm_metrics,
    extract_dual_arm_metrics,
)


class DualArmMetricsTest(unittest.TestCase):
    def test_extracts_diagnosis_and_recovery_latency(self):
        records = [
            {"record_type": "event", "payload": {
                "name": "fault_injected", "simulation_time": 2.0,
            }},
            {"record_type": "event", "payload": {
                "name": "temporal_fault_belief_updated",
                "leading_probability": 0.8,
                "entropy": 0.5,
                "expected_information_gain": 0.2,
                "posterior": {
                    "synchronization_error": 0.8,
                    "bimanual_slip": 0.2,
                },
            }},
            {"record_type": "event", "payload": {
                "name": "active_probe_started", "prior_entropy": 0.8,
            }},
            {"record_type": "event", "payload": {
                "name": "active_probe_completed", "positive": True,
                "entropy": 0.5,
            }},
            {"record_type": "diagnosis", "payload": {
                "timestamp": 2.12, "primary_fault": "synchronization_error",
            }},
            {"record_type": "event", "payload": {
                "name": "recovery_started", "simulation_time": 2.2,
            }},
            {"record_type": "event", "payload": {
                "name": "recovery_completed", "simulation_time": 2.5,
            }},
            {"record_type": "summary", "payload": {
                "success": True,
                "terminal_stage": "verify_completion",
                "simulation_duration": 8.0,
                "recovery_count": 1,
                "maximum_synchronization_error": 0.2,
                "maximum_object_tilt": 0.1,
                "final_position_error": 0.02,
                "final_object_tilt": 0.03,
            }},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.jsonl"
            path.write_text(
                "\n".join(json.dumps(item) for item in records),
                encoding="utf-8",
            )
            result = extract_dual_arm_metrics(
                path, "run", "full", "synchronization_delay", 3, 0
            )
        self.assertAlmostEqual(result.diagnosis_latency, 0.12)
        self.assertAlmostEqual(result.recovery_duration, 0.3)
        self.assertAlmostEqual(result.final_position_error, 0.02)
        self.assertAlmostEqual(result.final_object_tilt, 0.03)
        self.assertAlmostEqual(result.diagnosis_confidence, 0.8)
        self.assertAlmostEqual(result.diagnosis_brier_score, 0.08)
        self.assertAlmostEqual(
            result.diagnosis_negative_log_likelihood, -math.log(0.8)
        )
        self.assertEqual(result.diagnostic_probe_started_count, 1)
        self.assertEqual(result.active_probe_completed_count, 1)
        self.assertEqual(result.active_probe_positive_count, 1)
        self.assertAlmostEqual(result.probe_entropy_reduction_sum, 0.3)

    def test_aggregate_reports_uncertainty_and_tail_risk(self):
        with tempfile.TemporaryDirectory() as directory:
            episodes = []
            for seed, success, duration in ((1, True, 5.0), (2, False, 9.0)):
                path = Path(directory) / f"{seed}.jsonl"
                path.write_text(json.dumps({
                    "record_type": "summary",
                    "payload": {
                        "success": success,
                        "simulation_duration": duration,
                        "maximum_synchronization_error": 0.1 * seed,
                        "final_position_error": 0.01 * seed,
                        "final_object_tilt": 0.02 * seed,
                    },
                }), encoding="utf-8")
                episodes.append(extract_dual_arm_metrics(
                    path, str(seed), "full", "delay", seed, int(not success)
                ))
        row = aggregate_dual_arm_metrics(episodes)[0]
        self.assertEqual(row["success_rate"], 0.5)
        self.assertLess(row["success_ci95_low"], 0.5)
        self.assertGreater(row["success_ci95_high"], 0.5)
        self.assertEqual(row["duration_cvar90"], 9.0)

    def test_extracts_llm_acceptance_and_fallback_audit(self):
        records = [
            {"record_type": "event", "payload": {
                "name": "llm_candidate_audit", "status": "accepted",
                "latency_seconds": 0.4, "accepted_count": 2,
                "rejected_count": 0,
            }},
            {"record_type": "event", "payload": {
                "name": "llm_candidate_audit", "status": "fallback",
                "latency_seconds": 1.0, "accepted_count": 2,
                "rejected_count": 0,
            }},
            {"record_type": "recovery_plan", "payload": {
                "strategy_id": "visual_search_regrasp",
            }},
            {"record_type": "summary", "payload": {"success": True}},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "llm.jsonl"
            path.write_text(
                "\n".join(json.dumps(item) for item in records), encoding="utf-8"
            )
            episode = extract_dual_arm_metrics(
                path, "llm", "ours_llm", "transport_slip", 7, 0
            )
        self.assertEqual(episode.llm_request_count, 2)
        self.assertEqual(episode.llm_accept_count, 1)
        self.assertEqual(episode.llm_fallback_count, 1)
        self.assertAlmostEqual(episode.llm_mean_latency, 0.7)
        self.assertEqual(episode.recovery_strategy_id, "visual_search_regrasp")
        row = aggregate_dual_arm_metrics([episode])[0]
        self.assertEqual(row["llm_accept_rate"], 0.5)
        self.assertEqual(row["llm_fallback_rate"], 0.5)
        self.assertEqual(
            row["recovery_strategy_counts"], {"visual_search_regrasp": 1}
        )


if __name__ == "__main__":
    unittest.main()
