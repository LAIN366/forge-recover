"""Cross-module tests for the paper-scale dual-arm research phase."""

import math
import unittest

from cruzr_sim.adapters.real_robot import (
    JointCommand,
    RobotSafetyLimits,
    SafetyGatedCommandPort,
)
from cruzr_sim.experiments.policy import ExperimentPolicy
from cruzr_sim.experiments.research_design import (
    build_paired_factorial_cells,
    paired_cohens_d,
)
from cruzr_sim.faults.dual_arm import DualArmFaultInjector
from cruzr_sim.perception.multimodal_state import (
    MultimodalMeasurement,
    ReliabilityAwareFusion,
)
from cruzr_sim.planning.hierarchical_planner import (
    HierarchicalCandidate,
    RecoveryAwareHierarchicalPlanner,
)
from cruzr_sim.recovery.slip_recovery import (
    SearchObservation,
    SlipRecoveryPhase,
    SlipRecoveryPolicy,
)
from cruzr_sim.scenes.registry import load_research_scenarios
from cruzr_sim.tasks.cooperative_task_graph import ArmRole, build_cooperative_transport_graph


class FakeDriver:
    def __init__(self):
        self.sent = []
        self.stops = []

    def read_joint_positions(self, side):
        return (0.0, 0.0)

    def send_joint_command(self, command):
        self.sent.append(command)
        return True

    def stop(self, reason):
        self.stops.append(reason)


class ResearchPhaseTest(unittest.TestCase):
    def test_scene_registry_contains_all_primary_faults(self):
        scenarios = load_research_scenarios()
        for name in (
            "transport_slip", "vision_occlusion", "target_pose_shift",
            "sensor_dropout", "left_arm_failure", "right_arm_failure",
        ):
            self.assertIn(name, scenarios)

    def test_transport_slip_is_seeded_and_randomized(self):
        first = DualArmFaultInjector("transport_slip", seed=41)
        second = DualArmFaultInjector("transport_slip", seed=41)
        a = first.poll("cooperative_transport", 0.12)
        b = second.poll("cooperative_transport", 0.12)
        self.assertEqual(a, b)
        self.assertEqual(a.action, "release_object")
        self.assertNotEqual(a.parameters["drop_offset_x"], 0.0)

    def test_reliable_multimodal_evidence_yields_high_belief(self):
        measurement = MultimodalMeasurement(
            timestamp=1.0,
            pose_6d=(0.0, 0.0, 0.8, 0.0, 0.0, 0.0),
            vision_confidence=0.95,
            depth_valid=True,
            left_contacts=(True, True),
            right_contacts=(True, True),
            left_force=2.0,
            right_force=2.0,
            joint_tracking_error=0.005,
        )
        state = ReliabilityAwareFusion().update(measurement)
        self.assertGreater(state.pose_belief, 0.80)
        self.assertGreater(state.grasp_belief, 0.90)
        self.assertTrue(state.observable)

    def test_hierarchical_planner_rejects_brittle_short_plan(self):
        brittle = HierarchicalCandidate(
            "short", "transport", "left", "right", 1.0, 0.45,
            (8.0, 10.0), 0.03, 0.8, 0.7,
        )
        recoverable = HierarchicalCandidate(
            "recoverable", "transport", "right", "left", 1.5, 0.05,
            (2.0, 2.5), 0.06, 0.9, 0.9,
        )
        decision = RecoveryAwareHierarchicalPlanner().select((brittle, recoverable))
        self.assertEqual(decision.candidate.candidate_id, "recoverable")
        self.assertEqual(decision.evaluated_candidates, 2)

    def test_slip_policy_uses_visual_pose_before_regrasp(self):
        policy = SlipRecoveryPolicy()
        missing = SearchObservation(0.0, False, 0.0, None, True)
        self.assertEqual(policy.update(missing).phase, SlipRecoveryPhase.LOOK_DOWN)
        self.assertEqual(policy.update(missing).phase, SlipRecoveryPhase.SEARCH)
        pose = (0.1, 0.2, 0.8, 0.0, 0.0, 0.2)
        found = SearchObservation(0.2, True, 0.9, pose, True)
        command = policy.update(found)
        self.assertEqual(command.phase, SlipRecoveryPhase.APPROACH)
        self.assertEqual(command.target_pose_6d, pose)

    def test_task_graph_records_online_role_switch(self):
        graph = build_cooperative_transport_graph()
        graph.update_role("cooperative_transport", ArmRole.RIGHT, "left arm degraded")
        self.assertEqual(graph.nodes["cooperative_transport"].role, ArmRole.RIGHT)
        self.assertEqual(graph.history[-1][0], "role")
        self.assertGreater(graph.rollback_depth("cooperative_lift"), 0)

    def test_paired_design_has_four_methods_per_seed(self):
        cells = build_paired_factorial_cells(
            ("transport_slip",), severities=(1.0,), seeds=(7, 17)
        )
        self.assertEqual(len(cells), 8)
        self.assertEqual(len({cell.policy for cell in cells}), 4)
        self.assertTrue(math.isinf(paired_cohens_d((0, 0), (1, 1))))

    def test_real_robot_commands_require_operator_enable_and_limits(self):
        driver = FakeDriver()
        port = SafetyGatedCommandPort(
            driver,
            RobotSafetyLimits((-1.0, -1.0), (1.0, 1.0), maximum_joint_step=0.1),
        )
        command = JointCommand("left", (0.05, -0.05), 0.1, "test")
        self.assertFalse(port.execute(command, operator_enabled=False))
        port.heartbeat()
        self.assertTrue(port.execute(command, operator_enabled=True))
        self.assertEqual(driver.sent, [command])

    def test_policy_capabilities_match_ablation_contract(self):
        self.assertFalse(ExperimentPolicy.B0_FIXED_FSM.task_graph_enabled)
        self.assertTrue(ExperimentPolicy.B1_TASK_GRAPH.task_graph_enabled)
        self.assertFalse(ExperimentPolicy.B1_TASK_GRAPH.belief_enabled)
        self.assertTrue(ExperimentPolicy.B2_BELIEF_GRAPH.active_diagnosis_enabled)
        self.assertTrue(ExperimentPolicy.OURS.recovery_aware_cost_enabled)
        self.assertTrue(ExperimentPolicy.OURS.role_switch_enabled)


if __name__ == "__main__":
    unittest.main()
