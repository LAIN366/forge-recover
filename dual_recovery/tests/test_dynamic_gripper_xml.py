"""Tests for uniquely named dual dynamic gripper MJCF fragments."""

import importlib.util
import unittest

MUJOCO_AVAILABLE = importlib.util.find_spec("mujoco") is not None

if MUJOCO_AVAILABLE:
    from cruzr_sim.control.dynamic_gripper import (
        GRIPPER_MODEL_XML,
        GRIPPER_WORLDBODY_XML,
        build_gripper_model_xml,
        build_gripper_worldbody_xml,
    )


@unittest.skipUnless(MUJOCO_AVAILABLE, "requires MuJoCo")
class DynamicGripperXmlTest(unittest.TestCase):
    def test_default_builder_preserves_legacy_names(self):
        self.assertIn("dynamic_gripper_mount_joint", GRIPPER_WORLDBODY_XML)
        self.assertIn("dynamic_left_finger_actuator", GRIPPER_MODEL_XML)

    def test_two_prefixes_do_not_share_names(self):
        left = build_gripper_worldbody_xml("dual_left")
        right = build_gripper_worldbody_xml("dual_right")
        self.assertIn("dual_left_gripper_mount", left)
        self.assertNotIn("dual_right_gripper_mount", left)
        self.assertIn("dual_right_gripper_mount", right)

    def test_contact_target_is_configurable(self):
        xml = build_gripper_model_xml("dual_left", "transport_object_geom")
        self.assertIn('geom2="transport_object_geom"', xml)


if __name__ == "__main__":
    unittest.main()
