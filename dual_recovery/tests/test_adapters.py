"""Tests for the transport-neutral Cruzr S2 telemetry boundary."""

import unittest

from cruzr_sim.adapters import CruzrObservationAssembler, CruzrTelemetry


class CruzrAdapterTest(unittest.TestCase):
    def test_wrench_and_gripper_fields_are_preserved(self):
        telemetry = CruzrTelemetry(
            timestamp=1.0,
            stage="lift",
            object_position=(0.0, 0.0, 0.8),
            tool_position=(0.0, 0.0, 0.9),
            wrench_force=(3.0, 4.0, 12.0),
            wrench_torque=(0.1, 0.2, 0.3),
            left_contact=True,
            right_contact=True,
            gripper_current=1.2,
        )
        observation = CruzrObservationAssembler().build(telemetry)
        self.assertEqual(observation.normal_force, 13.0)
        self.assertEqual(observation.tangent_force, 5.0)
        self.assertEqual(observation.metadata["gripper_current"], 1.2)


if __name__ == "__main__":
    unittest.main()
