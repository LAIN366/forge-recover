"""Tests for seeded cooperative dual-arm fault directives."""

import unittest

from cruzr_sim.faults.dual_arm import DualArmFaultInjector


class DualArmFaultInjectorTest(unittest.TestCase):
    def test_synchronization_delay_is_seeded(self):
        first = DualArmFaultInjector("synchronization_delay", seed=12)
        second = DualArmFaultInjector("synchronization_delay", seed=12)
        self.assertIsNone(first.poll("cooperative_transport", 0.09))
        self.assertIsNone(second.poll("cooperative_transport", 0.09))
        directive_a = first.poll("cooperative_transport", 0.10)
        directive_b = second.poll("cooperative_transport", 0.10)
        self.assertEqual(directive_a, directive_b)
        self.assertIn(directive_a.parameters["arm"], {"left", "right"})

    def test_contact_loss_waits_for_cooperative_lift(self):
        injector = DualArmFaultInjector("left_contact_loss")
        self.assertIsNone(injector.poll("dual_grasp", 1.0))
        self.assertIsNone(injector.poll("cooperative_lift", 0.09))
        directive = injector.poll("cooperative_lift", 0.10)
        self.assertEqual(directive.action, "disable_left_contact")

    def test_dynamic_obstacle_targets_a_seeded_future_arm_path(self):
        first = DualArmFaultInjector("dynamic_obstacle", seed=21)
        second = DualArmFaultInjector("dynamic_obstacle", seed=21)
        self.assertIsNone(first.poll("cooperative_transport", 0.001))
        self.assertIsNone(second.poll("cooperative_transport", 0.001))
        directive_a = first.poll("cooperative_transport", 0.002)
        directive_b = second.poll("cooperative_transport", 0.002)
        self.assertEqual(directive_a, directive_b)
        self.assertEqual(directive_a.action, "insert_obstacle")
        self.assertIn(directive_a.parameters["side"], {"left", "right"})
        self.assertGreaterEqual(directive_a.parameters["path_fraction"], 0.25)
        self.assertLessEqual(directive_a.parameters["path_fraction"], 0.45)


if __name__ == "__main__":
    unittest.main()
