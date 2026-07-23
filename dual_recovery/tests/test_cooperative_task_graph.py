"""Tests for belief-aware dual-arm task execution and recovery."""

import unittest

from cruzr_sim.tasks.cooperative_task_graph import (
    NodeState,
    build_cooperative_transport_graph,
)


class CooperativeTaskGraphTest(unittest.TestCase):
    def test_dependencies_unlock_nodes_in_order(self):
        graph = build_cooperative_transport_graph()
        ready = {node.node_id for node in graph.ready_nodes()}
        self.assertEqual(ready, {"observe_scene"})

        graph.start("observe_scene")
        graph.complete("observe_scene")
        self.assertEqual(
            graph.nodes["estimate_object_pose"].state,
            NodeState.READY,
        )

    def test_low_belief_blocks_pose_dependent_work(self):
        graph = build_cooperative_transport_graph()
        graph.start("observe_scene")
        graph.complete("observe_scene")
        graph.update_belief("estimate_object_pose", 0.45)
        self.assertEqual(
            graph.nodes["estimate_object_pose"].state,
            NodeState.BLOCKED,
        )

    def test_failure_activates_local_recovery(self):
        graph = build_cooperative_transport_graph()
        graph.start("observe_scene")
        graph.complete("observe_scene")
        graph.start("estimate_object_pose")
        recovery = graph.fail("estimate_object_pose", "target occluded")
        self.assertEqual(recovery.node_id, "active_reobserve")
        self.assertEqual(recovery.state, NodeState.READY)
        self.assertEqual(recovery.metadata["failed_node"], "estimate_object_pose")

    def test_recovery_transaction_resumes_failed_node(self):
        graph = build_cooperative_transport_graph()
        graph.start("observe_scene")
        recovery = graph.fail("observe_scene", "camera unavailable")
        self.assertIsNone(recovery)

        graph = build_cooperative_transport_graph()
        graph.start("observe_scene")
        graph.complete("observe_scene")
        graph.start("estimate_object_pose")
        recovery = graph.fail("estimate_object_pose", "target occluded")
        graph.start(recovery.node_id)
        graph.resume_after_recovery(recovery.node_id, "estimate_object_pose")
        self.assertEqual(
            graph.nodes["estimate_object_pose"].state, NodeState.RUNNING
        )
        self.assertEqual(
            graph.nodes["estimate_object_pose"].metadata["recovery_count"], 1
        )

    def test_diagnosis_can_select_a_context_specific_recovery_branch(self):
        graph = build_cooperative_transport_graph()
        graph.start("observe_scene")
        graph.complete("observe_scene")
        graph.start("estimate_object_pose")
        graph.complete("estimate_object_pose")
        graph.start("assign_arm_roles")
        graph.complete("assign_arm_roles")
        graph.start("plan_dual_pregrasp")
        recovery = graph.fail(
            "plan_dual_pregrasp",
            "visual confidence collapsed",
            recovery_node_id="active_reobserve",
        )
        self.assertEqual(recovery.node_id, "active_reobserve")
        self.assertEqual(recovery.state, NodeState.READY)


if __name__ == "__main__":
    unittest.main()
