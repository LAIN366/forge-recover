from types import SimpleNamespace

import numpy as np

from cruzr_sim.planning.dual_arm_execution import plan_dual_goal


class Solver:
    def __init__(self, address):
        self.qpos_adrs = np.array([address])

    def solve(self, *_):
        raise AssertionError("a validated IK candidate must not be solved again")


class Planner:
    def plan(self, start, goal, **_):
        return SimpleNamespace(success=True, method="direct", waypoints=(start, goal))

    def configuration_collision_reasons(self, *_):
        return ()


def test_precomputed_joint_goal_reuses_candidate_but_still_plans_both_arms():
    start = np.array([0.0, 0.0])
    goal = np.array([0.2, -0.2])
    left = Planner()
    right = Planner()
    result = plan_dual_goal(
        start, np.zeros(3), np.zeros(3), Solver(0), Solver(1), left, right,
        precomputed_joint_goal=goal,
    )
    assert np.allclose(result[-1], goal)


def test_precomputed_joint_goal_shape_is_validated():
    start = np.array([0.0, 0.0])
    try:
        plan_dual_goal(
            start, np.zeros(3), np.zeros(3), Solver(0), Solver(1), Planner(), Planner(),
            precomputed_joint_goal=np.array([0.0]),
        )
    except ValueError as error:
        assert "wrong shape" in str(error)
    else:
        raise AssertionError("invalid precomputed goal was accepted")
