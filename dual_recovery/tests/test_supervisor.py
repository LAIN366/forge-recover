"""Tests for recovery gating in the high-level supervisor."""

import unittest

from cruzr_sim.diagnosis.types import (
    DiagnosisReport,
    DiagnosticEvidence,
    FaultType,
    ManipulationObservation,
)
from cruzr_sim.tasks.manipulation_supervisor import ManipulationSupervisor
from cruzr_sim.tasks.active_probe_runtime import ActiveProbeRuntime


def unstable_sample(timestamp):
    return ManipulationObservation(
        timestamp=timestamp,
        stage="lift",
        object_position=(0.0, 0.0, 0.9),
        tool_position=(0.0, 0.0, 0.93),
        left_contact=True,
        right_contact=False,
        normal_force=2.0,
        tangent_force=1.7,
        object_vertical_velocity=0.02,
    )


def unstable_report(timestamp=0.0):
    return DiagnosisReport(
        timestamp=timestamp,
        stage="lift",
        anomalous=True,
        primary_fault=FaultType.UNSTABLE_GRASP,
        confidence=0.8,
        hypotheses=(
            DiagnosticEvidence(
                FaultType.UNSTABLE_GRASP, 0.8, ("asymmetric contact",)
            ),
        ),
    )


class SupervisorTest(unittest.TestCase):
    def test_immediate_collision_keeps_detector_confidence(self):
        supervisor = ManipulationSupervisor()
        sample = ManipulationObservation(
            timestamp=0.0,
            stage="right_detour",
            object_position=(0.0, 0.0, 0.8),
            tool_position=(0.0, 0.0, 0.9),
            collision=True,
        )
        decision = supervisor.observe(sample)
        self.assertEqual(decision.confirmed_fault.value, "collision")
        self.assertGreaterEqual(decision.confidence, 0.99)
        self.assertIsNotNone(decision.recovery_plan)

    def test_low_confidence_anomaly_does_not_trigger_recovery(self):
        supervisor = ManipulationSupervisor(confirmation_frames=2)
        decisions = []
        for i in range(8):
            sample = ManipulationObservation(
                timestamp=i * 0.2,
                stage="lift",
                object_position=(0.0, 0.0, 0.9 + i * 0.002),
                tool_position=(0.0, 0.0, 0.93),
                left_contact=False,
                right_contact=False,
                object_vertical_velocity=0.01,
            )
            decisions.append(supervisor.observe(sample))
        self.assertTrue(any(item.report.anomalous for item in decisions))
        self.assertTrue(any(item.probe is not None for item in decisions))
        self.assertTrue(all(item.recovery_plan is None for item in decisions))

    def test_pause_probe_rejects_contact_sensor_dropout(self):
        runtime = ActiveProbeRuntime()
        masked = unstable_sample(0.0)
        runtime.start("pause_and_hold", masked, unstable_report())
        follow_up = unstable_sample(0.5)
        outcome = runtime.update(follow_up, physical_contact_flags=(True, True))
        self.assertIsNotNone(outcome)
        self.assertFalse(outcome.positive)

    def test_reobserve_confirms_displaced_target(self):
        runtime = ActiveProbeRuntime()
        sample = ManipulationObservation(
            timestamp=0.0,
            stage="approach",
            object_position=(0.10, 0.0, 0.8),
            tool_position=(0.0, 0.0, 0.9),
            target_position=(0.0, 0.0, 0.8),
        )
        runtime.start("reobserve", sample, unstable_report())
        follow_up = ManipulationObservation(
            timestamp=0.3,
            stage="approach",
            object_position=sample.object_position,
            tool_position=sample.tool_position,
            target_position=sample.target_position,
        )
        outcome = runtime.update(follow_up)
        self.assertTrue(outcome.positive)

    def test_reobserve_keeps_precontact_target_reference(self):
        runtime = ActiveProbeRuntime()
        sample = ManipulationObservation(
            timestamp=0.0,
            stage="approach",
            object_position=(0.06, 0.0, 0.8),
            tool_position=(0.0, 0.0, 0.9),
            target_position=(0.0, 0.0, 0.8),
        )
        runtime.start("reobserve", sample, unstable_report())
        follow_up = ManipulationObservation(
            timestamp=0.3,
            stage="close",
            object_position=sample.object_position,
            tool_position=sample.tool_position,
            target_position=None,
        )
        outcome = runtime.update(follow_up)
        self.assertTrue(outcome.positive)

    def test_probe_runtime_reset_cancels_in_flight_probe(self):
        runtime = ActiveProbeRuntime()
        sample = unstable_sample(0.0)
        self.assertTrue(
            runtime.start("pause_and_hold", sample, unstable_report())
        )
        runtime.reset()
        self.assertIsNone(runtime.update(unstable_sample(1.0)))

    def test_positive_probe_can_confirm_transient_anomaly(self):
        supervisor = ManipulationSupervisor()
        supervisor.active_diagnoser.state.beliefs[FaultType.MISSED_GRASP] = 0.70
        normal_report = DiagnosisReport(
            timestamp=2.0,
            stage="close",
            anomalous=False,
            primary_fault=FaultType.NORMAL,
            confidence=1.0,
            hypotheses=(
                DiagnosticEvidence(FaultType.NORMAL, 1.0, ("currently normal",)),
            ),
        )
        decision = supervisor.complete_probe(
            normal_report,
            "close",
            "small_gripper_close",
            positive=True,
        )
        self.assertEqual(decision.confirmed_fault, FaultType.MISSED_GRASP)
        self.assertIsNotNone(decision.recovery_plan)
        self.assertEqual(decision.recovery_plan.fault, FaultType.MISSED_GRASP)

    def test_probe_preserves_report_when_latest_observation_is_normal(self):
        runtime = ActiveProbeRuntime()
        supervisor = ManipulationSupervisor()
        supervisor.active_diagnoser.state.beliefs[FaultType.MISSED_GRASP] = 0.70
        initiating_report = DiagnosisReport(
            timestamp=0.0,
            stage="close",
            anomalous=True,
            primary_fault=FaultType.MISSED_GRASP,
            confidence=0.70,
            hypotheses=(
                DiagnosticEvidence(
                    FaultType.MISSED_GRASP, 0.70, ("closure without contact",)
                ),
            ),
        )
        sample = ManipulationObservation(
            timestamp=0.0,
            stage="close",
            object_position=(0.0, 0.0, 0.9),
            tool_position=(0.0, 0.0, 0.93),
        )
        self.assertTrue(
            runtime.start("small_gripper_close", sample, initiating_report)
        )
        normal_observation = ManipulationObservation(
            timestamp=0.5,
            stage="close",
            object_position=sample.object_position,
            tool_position=sample.tool_position,
        )
        normal_decision = supervisor.observe(normal_observation)
        self.assertFalse(normal_decision.report.anomalous)

        outcome = runtime.update(normal_observation)
        self.assertTrue(outcome.positive)
        self.assertIs(outcome.report, initiating_report)
        decision = supervisor.complete_probe(
            outcome.report, "close", outcome.name, outcome.positive
        )
        self.assertEqual(decision.confirmed_fault, FaultType.MISSED_GRASP)
        self.assertIsNotNone(decision.recovery_plan)


if __name__ == "__main__":
    unittest.main()
