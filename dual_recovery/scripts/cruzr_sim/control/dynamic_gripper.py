"""Actuated two-finger gripper attached to a kinematic Cruzr wrist."""

import mujoco
import numpy as np


COLLISION_ENVIRONMENT = 8


def build_gripper_worldbody_xml(prefix="dynamic"):
    """Build a uniquely named dynamic gripper for one robot wrist."""
    return f"""
<body name="{prefix}_gripper_target" mocap="true" pos="0 0 -2">
  <geom name="{prefix}_gripper_target_geom" type="sphere" size="0.001"
        mass="0" rgba="0 0 0 0" contype="0" conaffinity="0"/>
</body>
<body name="{prefix}_gripper_mount" pos="0 0 -2">
  <freejoint name="{prefix}_gripper_mount_joint"/>
  <geom name="{prefix}_gripper_mount_geom" type="box" size="0.025 0.025 0.018"
        mass="0.30" rgba="0.16 0.18 0.21 1" contype="0" conaffinity="0"/>
  <body name="{prefix}_left_finger" pos="0 0.055 0">
    <joint name="{prefix}_left_finger_joint" type="slide" axis="0 1 0"
           range="-0.017 0.030" damping="8.0" armature="0.02"/>
    <geom name="{prefix}_left_pad_geom" type="box" size="0.040 0.018 0.035"
          mass="0.14" rgba="0.50 0.53 0.58 1" contype="4" conaffinity="10"
          friction="2.0 0.02 0.002" solref="0.012 1" solimp="0.90 0.97 0.002"/>
    <geom name="{prefix}_left_lip_geom" type="box" pos="0 -0.020 -0.045"
          size="0.040 0.022 0.004" mass="0.03" rgba="0.16 0.18 0.21 1"
          contype="4" conaffinity="10" friction="2.0 0.02 0.002"
          solref="0.012 1" solimp="0.90 0.97 0.002"/>
  </body>
  <body name="{prefix}_right_finger" pos="0 -0.055 0">
    <joint name="{prefix}_right_finger_joint" type="slide" axis="0 1 0"
           range="-0.030 0.017" damping="8.0" armature="0.02"/>
    <geom name="{prefix}_right_pad_geom" type="box" size="0.040 0.018 0.035"
          mass="0.14" rgba="0.50 0.53 0.58 1" contype="4" conaffinity="10"
          friction="2.0 0.02 0.002" solref="0.012 1" solimp="0.90 0.97 0.002"/>
    <geom name="{prefix}_right_lip_geom" type="box" pos="0 0.020 -0.045"
          size="0.040 0.022 0.004" mass="0.03" rgba="0.16 0.18 0.21 1"
          contype="4" conaffinity="10" friction="2.0 0.02 0.002"
          solref="0.012 1" solimp="0.90 0.97 0.002"/>
  </body>
</body>
"""


def build_gripper_model_xml(prefix="dynamic", object_geom="blue_cube_geom"):
    return f"""
<equality>
  <weld name="{prefix}_gripper_tracking_weld"
        body1="{prefix}_gripper_target" body2="{prefix}_gripper_mount"
        solref="0.015 1" solimp="0.90 0.97 0.002"/>
</equality>
<contact>
  <pair geom1="{prefix}_left_pad_geom" geom2="{object_geom}"
        condim="6" friction="4.0 4.0 0.05 0.01 0.01"
        solref="0.010 1" solimp="0.92 0.98 0.002"/>
  <pair geom1="{prefix}_right_pad_geom" geom2="{object_geom}"
        condim="6" friction="4.0 4.0 0.05 0.01 0.01"
        solref="0.010 1" solimp="0.92 0.98 0.002"/>
</contact>
<actuator>
  <position name="{prefix}_left_finger_actuator"
            joint="{prefix}_left_finger_joint" kp="260"
            ctrlrange="-0.017 0.030" forcerange="-18 18"/>
  <position name="{prefix}_right_finger_actuator"
            joint="{prefix}_right_finger_joint" kp="260"
            ctrlrange="-0.030 0.017" forcerange="-18 18"/>
</actuator>
"""


GRIPPER_WORLDBODY_XML = build_gripper_worldbody_xml()
GRIPPER_MODEL_XML = build_gripper_model_xml()


class DynamicGripper:
    """Tracks the wrist with a mocap mount and actuates dynamic fingers."""

    def __init__(self, model, data, ik_solver, prefix="dynamic"):
        self.model = model
        self.data = data
        self.ik_solver = ik_solver
        self.prefix = str(prefix)
        self.target_body = self._id(
            mujoco.mjtObj.mjOBJ_BODY, f"{self.prefix}_gripper_target"
        )
        self.mount_body = self._id(
            mujoco.mjtObj.mjOBJ_BODY, f"{self.prefix}_gripper_mount"
        )
        self.mount_mocap = model.body_mocapid[self.target_body]
        self.mount_joint = self._id(
            mujoco.mjtObj.mjOBJ_JOINT, f"{self.prefix}_gripper_mount_joint"
        )
        self.left_joint = self._id(
            mujoco.mjtObj.mjOBJ_JOINT, f"{self.prefix}_left_finger_joint"
        )
        self.right_joint = self._id(
            mujoco.mjtObj.mjOBJ_JOINT, f"{self.prefix}_right_finger_joint"
        )
        self.left_actuator = self._id(
            mujoco.mjtObj.mjOBJ_ACTUATOR, f"{self.prefix}_left_finger_actuator"
        )
        self.right_actuator = self._id(
            mujoco.mjtObj.mjOBJ_ACTUATOR, f"{self.prefix}_right_finger_actuator"
        )
        self.left_geom = self._id(
            mujoco.mjtObj.mjOBJ_GEOM, f"{self.prefix}_left_pad_geom"
        )
        self.right_geom = self._id(
            mujoco.mjtObj.mjOBJ_GEOM, f"{self.prefix}_right_pad_geom"
        )
        self.left_lip_geom = self._id(
            mujoco.mjtObj.mjOBJ_GEOM, f"{self.prefix}_left_lip_geom"
        )
        self.right_lip_geom = self._id(
            mujoco.mjtObj.mjOBJ_GEOM, f"{self.prefix}_right_lip_geom"
        )
        self._saved_collision_masks = None
        self._saved_pair_margins = None
        tool_position = self.ik_solver.tool_position(self.data)
        mount_qpos = self.model.jnt_qposadr[self.mount_joint]
        self.data.qpos[mount_qpos:mount_qpos + 3] = tool_position
        self.data.qpos[mount_qpos + 3:mount_qpos + 7] = [1.0, 0.0, 0.0, 0.0]
        self.data.mocap_pos[self.mount_mocap] = tool_position
        self.data.mocap_quat[self.mount_mocap] = [1.0, 0.0, 0.0, 0.0]
        self.open()
        mujoco.mj_forward(self.model, self.data)

    def _id(self, object_type, name):
        object_id = mujoco.mj_name2id(self.model, object_type, name)
        if object_id < 0:
            raise ValueError(f"MuJoCo object not found: {name}")
        return object_id

    @property
    def dynamic_qpos_addresses(self):
        mount_qpos = self.model.jnt_qposadr[self.mount_joint]
        return np.array(list(range(mount_qpos, mount_qpos + 7)) + [
            self.model.jnt_qposadr[self.left_joint],
            self.model.jnt_qposadr[self.right_joint],
        ])

    @property
    def dynamic_dof_addresses(self):
        mount_dof = self.model.jnt_dofadr[self.mount_joint]
        return np.array(list(range(mount_dof, mount_dof + 6)) + [
            self.model.jnt_dofadr[self.left_joint],
            self.model.jnt_dofadr[self.right_joint],
        ])

    def follow_wrist(self):
        self.data.mocap_pos[self.mount_mocap] = self.ik_solver.tool_position(self.data)
        self.data.mocap_quat[self.mount_mocap] = np.array([1.0, 0.0, 0.0, 0.0])

    def open(self):
        self.data.ctrl[self.left_actuator] = 0.025
        self.data.ctrl[self.right_actuator] = -0.025

    def close(self, closure=0.017):
        closure = float(np.clip(closure, 0.0, 0.017))
        target = 0.025 - 2.45 * closure
        self.data.ctrl[self.left_actuator] = target
        self.data.ctrl[self.right_actuator] = -target

    def contact_flags(self, cube_geom):
        """Return independent left/right contact state for diagnosis."""
        touched = {self.left_geom: False, self.right_geom: False}
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            pair = {contact.geom1, contact.geom2}
            for pad_geom in touched:
                if pad_geom in pair and cube_geom in pair:
                    touched[pad_geom] = True
        return touched[self.left_geom], touched[self.right_geom]

    def both_pads_touch(self, cube_geom):
        return all(self.contact_flags(cube_geom))

    def enable_cube_contacts(self, cube_geom, enabled):
        gripper_geoms = (
            self.left_geom, self.right_geom,
            self.left_lip_geom, self.right_lip_geom,
        )
        if self._saved_collision_masks is None:
            self._saved_collision_masks = {
                geom: (
                    int(self.model.geom_contype[geom]),
                    int(self.model.geom_conaffinity[geom]),
                )
                for geom in gripper_geoms
            }
            self._saved_pair_margins = {
                pair_id: float(self.model.pair_margin[pair_id])
                for pair_id in range(self.model.npair)
                if cube_geom in {
                    int(self.model.pair_geom1[pair_id]),
                    int(self.model.pair_geom2[pair_id]),
                }
                and {
                    int(self.model.pair_geom1[pair_id]),
                    int(self.model.pair_geom2[pair_id]),
                }.intersection(gripper_geoms)
            }
        for geom in gripper_geoms:
            contype, conaffinity = self._saved_collision_masks[geom]
            self.model.geom_contype[geom] = contype if enabled else 0
            self.model.geom_conaffinity[geom] = conaffinity if enabled else COLLISION_ENVIRONMENT
        for pair_id, margin in self._saved_pair_margins.items():
            self.model.pair_margin[pair_id] = margin if enabled else -1.0

    def contact_diagnostics(self, cube_geom):
        forces = []
        gripper_geoms = {
            self.left_geom, self.right_geom,
            self.left_lip_geom, self.right_lip_geom,
        }
        minimum_distance = 0.0
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            pair = {int(contact.geom1), int(contact.geom2)}
            if cube_geom not in pair or not pair.intersection(gripper_geoms):
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, contact_index, force)
            forces.append(force[:3])
            minimum_distance = min(minimum_distance, float(contact.dist))
        total_normal = sum(abs(float(force[0])) for force in forces)
        total_tangent = sum(float(np.linalg.norm(force[1:3])) for force in forces)
        vertical_force = 0.0
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            pair = {int(contact.geom1), int(contact.geom2)}
            if cube_geom not in pair or not pair.intersection(gripper_geoms):
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, contact_index, force)
            world_force = np.asarray(contact.frame).reshape(3, 3).T @ force[:3]
            vertical_force += (1.0 if contact.geom1 == cube_geom else -1.0) * world_force[2]
        qpos = self.data.qpos[[
            self.model.jnt_qposadr[self.left_joint],
            self.model.jnt_qposadr[self.right_joint],
        ]]
        actuator_force = self.data.actuator_force[
            [self.left_actuator, self.right_actuator]
        ]
        return {
            "contacts": len(forces),
            "normal_force": total_normal,
            "tangent_force": total_tangent,
            "vertical_force": vertical_force,
            "minimum_distance": minimum_distance,
            "finger_qpos": qpos.copy(),
            "actuator_force": actuator_force.copy(),
        }

    def environment_penetration(self, environment_geoms):
        gripper_geoms = {
            self.left_geom, self.right_geom,
            self.left_lip_geom, self.right_lip_geom,
        }
        environment_geoms = set(environment_geoms)
        minimum_distance = 0.0
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            pair = {int(contact.geom1), int(contact.geom2)}
            if pair & gripper_geoms and pair & environment_geoms:
                minimum_distance = min(minimum_distance, float(contact.dist))
        return minimum_distance
