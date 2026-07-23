"""Tests for collision-aware recovery path semantics."""

import importlib.util
from types import SimpleNamespace
import unittest

PLANNING_DEPS_AVAILABLE = (
    importlib.util.find_spec("numpy") is not None
    and importlib.util.find_spec("mujoco") is not None
)

if PLANNING_DEPS_AVAILABLE:
    import numpy as np

    from cruzr_sim.planning.arm_motion_planner import ArmMotionPlanner


@unittest.skipUnless(PLANNING_DEPS_AVAILABLE, "requires NumPy and MuJoCo")
class ArmMotionPlannerTest(unittest.TestCase):
    @staticmethod
    def planner_with_collision_rule(rule):
        planner = ArmMotionPlanner.__new__(ArmMotionPlanner)
        planner.qpos_adrs = np.array([0], dtype=int)
        planner.configuration_in_collision = (
            lambda qpos, allow_cube=False: bool(rule(float(qpos[0])))
        )
        return planner

    def test_recovery_edge_can_leave_but_not_reenter_collision(self):
        planner = self.planner_with_collision_rule(lambda value: value < 0.3)
        template = np.array([0.0])
        start = np.array([0.0])
        goal = np.array([1.0])
        self.assertTrue(planner.edge_in_collision(template, start, goal))
        self.assertFalse(planner.edge_in_collision(
            template,
            start,
            goal,
            resolution=0.1,
            allow_initial_collision=True,
        ))

        planner.configuration_in_collision = (
            lambda qpos, allow_cube=False: qpos[0] < 0.3 or qpos[0] > 0.8
        )
        self.assertTrue(planner.edge_in_collision(
            template,
            start,
            goal,
            resolution=0.1,
            allow_initial_collision=True,
        ))

    def test_recovery_edge_must_reach_free_space(self):
        planner = self.planner_with_collision_rule(lambda value: True)
        self.assertTrue(planner.edge_in_collision(
            np.array([0.0]),
            np.array([0.0]),
            np.array([1.0]),
            resolution=0.1,
            allow_initial_collision=True,
        ))

    def test_collision_boolean_is_derived_from_structured_reasons(self):
        planner = ArmMotionPlanner.__new__(ArmMotionPlanner)
        planner.configuration_collision_reasons = (
            lambda qpos, allow_cube=False: ("test_collision",)
            if qpos[0] > 0.5 else ()
        )
        self.assertFalse(planner.configuration_in_collision(np.array([0.0])))
        self.assertTrue(planner.configuration_in_collision(np.array([1.0])))

    def test_mocap_obstacle_state_is_synchronized(self):
        planner = ArmMotionPlanner.__new__(ArmMotionPlanner)
        planner.model = SimpleNamespace(nmocap=1)
        planner.data = SimpleNamespace(
            mocap_pos=np.zeros((1, 3)),
            mocap_quat=np.zeros((1, 4)),
        )
        source = SimpleNamespace(
            mocap_pos=np.array([[1.0, 2.0, 3.0]]),
            mocap_quat=np.array([[1.0, 0.0, 0.0, 0.0]]),
        )
        planner.sync_mocap_state(source)
        np.testing.assert_allclose(planner.data.mocap_pos, source.mocap_pos)
        np.testing.assert_allclose(planner.data.mocap_quat, source.mocap_quat)


if __name__ == "__main__":
    unittest.main()
