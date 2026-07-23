"""Unit tests for stage-aware manipulation diagnosis."""

import unittest

from cruzr_sim.diagnosis import FaultType, ManipulationObservation
from cruzr_sim.diagnosis.anomaly_detector import StageAwareAnomalyDetector


def observation(timestamp, stage, z=0.80, vz=0.0, left=False, right=False,
                normal=0.0, target=None, collision=False):
    return ManipulationObservation(
        timestamp=timestamp,
        stage=stage,
        object_position=(0.5, 0.0, z),
        tool_position=(0.5, 0.0, z + 0.03),
        left_contact=left,
        right_contact=right,
        normal_force=normal,
        object_vertical_velocity=vz,
        target_position=target,
        collision=collision,
    )


class StageAwareDetectorTest(unittest.TestCase):
    def test_normal_grasp_is_not_anomalous(self):
        detector = StageAwareAnomalyDetector()
        report = detector.update(observation(0.0, "close", left=True, right=True, normal=6.0))
        self.assertFalse(report.anomalous)
        self.assertEqual(report.primary_fault, FaultType.NORMAL)

    def test_transient_single_contact_during_close_is_not_unstable(self):
        detector = StageAwareAnomalyDetector()
        detector.update(observation(0.0, "close"))
        report = detector.update(observation(
            1.0, "close", left=True, right=False, normal=1.0
        ))
        self.assertFalse(report.anomalous)

    def test_missed_grasp_after_close_delay(self):
        detector = StageAwareAnomalyDetector()
        detector.update(observation(0.0, "close"))
        report = detector.update(observation(2.0, "close"))
        self.assertTrue(report.anomalous)
        self.assertEqual(report.primary_fault, FaultType.MISSED_GRASP)

    def test_slip_uses_contact_and_temporal_motion(self):
        detector = StageAwareAnomalyDetector()
        detector.update(observation(0.0, "lift", z=0.90, vz=0.0, left=True, right=True, normal=8.0))
        detector.update(observation(0.3, "lift", z=0.88, vz=-0.04, left=False, right=False))
        report = detector.update(observation(0.6, "lift", z=0.85, vz=-0.06, left=False, right=False))
        self.assertTrue(report.anomalous)
        self.assertEqual(report.primary_fault, FaultType.GRASP_SLIP)

    def test_target_displacement_is_detected(self):
        detector = StageAwareAnomalyDetector()
        report = detector.update(observation(
            0.0, "approach", target=(0.40, 0.0, 0.80)
        ))
        self.assertTrue(report.anomalous)
        self.assertEqual(report.primary_fault, FaultType.TARGET_DISPLACEMENT)

    def test_contact_motion_is_not_target_displacement(self):
        detector = StageAwareAnomalyDetector()
        report = detector.update(ManipulationObservation(
            timestamp=0.0,
            stage="close",
            object_position=(0.5, 0.0, 0.8),
            tool_position=(0.5, 0.0, 0.83),
            left_contact=True,
            target_position=(0.4, 0.0, 0.8),
        ))
        self.assertNotEqual(report.primary_fault, FaultType.TARGET_DISPLACEMENT)

    def test_collision_is_immediate(self):
        detector = StageAwareAnomalyDetector()
        report = detector.update(observation(0.0, "pregrasp", collision=True))
        self.assertTrue(report.anomalous)
        self.assertEqual(report.primary_fault, FaultType.COLLISION)


if __name__ == "__main__":
    unittest.main()
