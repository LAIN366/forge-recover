"""Tests for temporal confirmation in dual-arm recovery supervision."""

import unittest

from cruzr_sim.diagnosis.dual_arm import DualArmFaultType, DualArmObservation
from cruzr_sim.tasks.dual_arm_supervisor import DualArmSupervisor


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


class DualArmSupervisorTest(unittest.TestCase):
    def test_transient_synchronization_error_is_not_recovered(self):
        supervisor = DualArmSupervisor(confirmation_frames=3)
        decision = supervisor.observe(observation(synchronization_error=0.06))
        self.assertEqual(decision.confirmed_fault, DualArmFaultType.NORMAL)
        self.assertIsNone(decision.recovery_plan)

    def test_persistent_error_creates_one_graph_plan(self):
        supervisor = DualArmSupervisor(confirmation_frames=2)
        supervisor.observe(observation(synchronization_error=0.06))
        decision = supervisor.observe(observation(
            timestamp=1.1, synchronization_error=0.06
        ))
        self.assertEqual(
            decision.confirmed_fault, DualArmFaultType.SYNCHRONIZATION_ERROR
        )
        self.assertEqual(decision.recovery_plan.recovery_node, "stabilize_object")
        repeated = supervisor.observe(observation(
            timestamp=1.2, synchronization_error=0.07
        ))
        self.assertIs(repeated.recovery_plan, decision.recovery_plan)
        self.assertEqual(supervisor.recovery_attempts, 1)
        self.assertGreater(repeated.fault_posterior, 0.68)
        self.assertGreaterEqual(repeated.expected_information_gain, 0.0)

    def test_no_temporal_belief_uses_frame_confirmation(self):
        supervisor = DualArmSupervisor(
            confirmation_frames=2, use_temporal_belief=False
        )
        first = supervisor.observe(observation(synchronization_error=0.06))
        second = supervisor.observe(observation(
            timestamp=1.1, synchronization_error=0.06
        ))
        self.assertEqual(first.confirmed_fault, DualArmFaultType.NORMAL)
        self.assertIsNone(first.fault_distribution)
        self.assertEqual(
            second.confirmed_fault, DualArmFaultType.SYNCHRONIZATION_ERROR
        )
        self.assertIsNotNone(second.recovery_plan)

    def test_dynamic_obstacle_is_immediate(self):
        supervisor = DualArmSupervisor(confirmation_frames=5)
        decision = supervisor.observe(observation(collision=True))
        self.assertEqual(
            decision.confirmed_fault, DualArmFaultType.DYNAMIC_OBSTACLE
        )
        self.assertIsNotNone(decision.recovery_plan)

    def test_successful_recovery_updates_contextual_experience(self):
        supervisor = DualArmSupervisor()
        decision = supervisor.observe(observation(
            left_contacts=(False, False),
            right_contacts=(False, False),
            object_vertical_velocity=-0.05,
        ))
        self.assertIsNotNone(decision.recovery_plan)
        node = supervisor.mark_recovery_complete(True, timestamp=2.5)
        self.assertEqual(node.successes, 1)
        self.assertEqual(node.failures, 0)
        self.assertEqual(node.costs, [1.5])


if __name__ == "__main__":
    unittest.main()
