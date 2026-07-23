"""Tests for multimodal cooperative-failure diagnosis and recovery mapping."""

import unittest

from cruzr_sim.diagnosis.dual_arm import (
    DualArmAnomalyDetector,
    DualArmFaultType,
    DualArmObservation,
)
from cruzr_sim.recovery.dual_arm import DualArmRecoveryPlanner


def observation(**overrides):
    values = {
        "timestamp": 1.0,
        "stage": "cooperative_transport",
        "object_position": (0.0, 0.0, 0.9),
        "object_rpy": (0.0, 0.0, 0.0),
        "left_tool_position": (-0.1, 0.0, 0.9),
        "right_tool_position": (0.1, 0.0, 0.9),
        "left_contacts": (True, True),
        "right_contacts": (True, True),
    }
    values.update(overrides)
    return DualArmObservation(**values)


class DualArmDiagnosisTest(unittest.TestCase):
    def test_normal_cooperation_is_not_anomalous(self):
        report = DualArmAnomalyDetector().update(observation())
        self.assertFalse(report.anomalous)

    def test_cross_arm_contact_identifies_failed_side(self):
        report = DualArmAnomalyDetector().update(observation(
            left_contacts=(False, False),
        ))
        self.assertEqual(report.primary_fault, DualArmFaultType.LEFT_GRASP_LOSS)
        plan = DualArmRecoveryPlanner().plan(report, "cooperative_transport")
        self.assertEqual(plan.recovery_node, "stabilize_object")
        self.assertIn("left_regrasp", plan.actions)

    def test_synchronization_uses_tracking_and_object_tilt(self):
        report = DualArmAnomalyDetector().update(observation(
            synchronization_error=0.055,
            right_tracking_error=0.06,
            object_rpy=(0.25, 0.0, 0.0),
        ))
        self.assertEqual(
            report.primary_fault, DualArmFaultType.SYNCHRONIZATION_ERROR
        )
        self.assertGreaterEqual(report.confidence, 0.90)

    def test_object_tilt_is_an_independent_closed_chain_residual(self):
        report = DualArmAnomalyDetector().update(observation(
            object_rpy=(0.24, 0.0, 0.0),
        ))
        self.assertEqual(
            report.primary_fault, DualArmFaultType.SYNCHRONIZATION_ERROR
        )
        self.assertIn("object tilt", " ".join(report.hypotheses[0].reasons))

    def test_visual_dropout_selects_reobservation(self):
        report = DualArmAnomalyDetector().update(observation(
            stage="plan_dual_pregrasp",
            visual_valid=False,
            visual_confidence=0.0,
        ))
        self.assertEqual(
            report.primary_fault, DualArmFaultType.VISUAL_DEGRADATION
        )
        self.assertEqual(report.selected_probe, "active_reobserve")


if __name__ == "__main__":
    unittest.main()
