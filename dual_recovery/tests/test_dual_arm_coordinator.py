"""Tests for risk-aware role assignment and dual-arm time alignment."""

import importlib.util
import unittest

NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None

if NUMPY_AVAILABLE:
    import numpy as np

    from cruzr_sim.planning.dual_arm_coordinator import (
        ArmCapability,
        assign_primary_and_support,
        synchronize_paths,
    )


@unittest.skipUnless(NUMPY_AVAILABLE, "requires NumPy")
class DualArmCoordinatorTest(unittest.TestCase):
    def test_safer_arm_becomes_primary(self):
        left = ArmCapability("left", True, 0.9, 0.8, 0.2)
        right = ArmCapability("right", True, 0.4, 0.5, 0.6)
        assignment = assign_primary_and_support(left, right)
        self.assertEqual(assignment.primary, "left")
        self.assertEqual(assignment.support, "right")

    def test_unreachable_arm_cannot_be_primary(self):
        left = ArmCapability("left", False, 1.0, 1.0, 0.0)
        right = ArmCapability("right", True, 0.3, 0.3, 0.4)
        self.assertEqual(assign_primary_and_support(left, right).primary, "right")

    def test_synchronized_paths_share_timeline_and_endpoints(self):
        left = [np.array([0.0, 0.0]), np.array([1.0, 0.5])]
        right = [
            np.array([0.0, 0.0]),
            np.array([0.2, -0.1]),
            np.array([0.4, -0.2]),
        ]
        plan = synchronize_paths(left, right)
        self.assertEqual(len(plan.left_waypoints), len(plan.right_waypoints))
        np.testing.assert_allclose(plan.left_waypoints[-1], left[-1])
        np.testing.assert_allclose(plan.right_waypoints[-1], right[-1])


if __name__ == "__main__":
    unittest.main()
