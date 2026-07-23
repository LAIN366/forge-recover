"""Tests for deterministic fault-injection triggers."""

import unittest

from cruzr_sim.faults import FaultInjectionConfig, FaultInjector, FaultScenario


class FaultInjectorTest(unittest.TestCase):
    def test_slip_waits_until_object_is_visibly_lifted(self):
        injector = FaultInjector(FaultInjectionConfig(FaultScenario.SLIP))
        self.assertIsNone(injector.poll("lift", 0.074, 3.0))
        self.assertIsNone(injector.poll("lift", 0.080, 1.9))
        directive = injector.poll("lift", 0.080, 2.1)
        self.assertEqual(directive.action, "release_grasp")
        self.assertNotEqual(directive.parameters["velocity_x"], 0.0)
        self.assertIn("velocity_y", directive.parameters)
        self.assertGreater(directive.parameters["angular_velocity"], 0.0)

    def test_target_shift_is_seeded_and_injected_once(self):
        config = FaultInjectionConfig(FaultScenario.TARGET_SHIFT, seed=10)
        first = FaultInjector(config)
        second = FaultInjector(config)
        directive_a = first.poll("approach", 0.0, 0.2)
        directive_b = second.poll("approach", 0.0, 0.2)
        self.assertEqual(directive_a.parameters, directive_b.parameters)
        self.assertIsNone(first.poll("approach", 0.0, 0.4))

    def test_planning_failure_uses_planning_stage(self):
        injector = FaultInjector(FaultInjectionConfig(FaultScenario.PLANNING_FAILURE))
        self.assertIsNone(injector.poll("clearance", 0.0, 0.0))
        directive = injector.poll("right_detour", 0.0, 0.0)
        self.assertEqual(directive.action, "report_planning_failure")

    def test_collision_event_moves_seeded_physical_obstacle(self):
        config = FaultInjectionConfig(FaultScenario.COLLISION_EVENT, seed=22)
        first_injector = FaultInjector(config)
        self.assertIsNone(first_injector.poll("right_detour", 0.0, 0.09))
        first = first_injector.poll("right_detour", 0.0, 0.10)
        second = FaultInjector(config).poll("right_detour", 0.0, 0.10)
        self.assertEqual(first.action, "move_dynamic_obstacle")
        self.assertEqual(first.parameters, second.parameters)
        self.assertGreaterEqual(first.parameters["path_fraction"], 0.22)
        self.assertLessEqual(first.parameters["path_fraction"], 0.32)


if __name__ == "__main__":
    unittest.main()
