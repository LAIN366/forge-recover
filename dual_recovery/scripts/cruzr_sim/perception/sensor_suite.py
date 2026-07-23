"""Synchronized proprioceptive, force, contact, and task-space sensors."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SensorPacket:
    timestamp: float
    joint_position: tuple[float, ...]
    joint_velocity: tuple[float, ...]
    actuator_force: tuple[float, ...]
    wrist_force: tuple[float, float, float]
    wrist_torque: tuple[float, float, float]
    object_position: tuple[float, float, float]
    tool_position: tuple[float, float, float]
    gripper_contacts: tuple[bool, bool]


class MujocoSensorSuite:
    """Read a coherent sensor packet after a completed MuJoCo integration step."""

    def __init__(self, data, arm_qpos_addresses, arm_dof_addresses):
        self.data = data
        self.arm_qpos_addresses = tuple(int(value) for value in arm_qpos_addresses)
        self.arm_dof_addresses = tuple(int(value) for value in arm_dof_addresses)

    def sample(
        self,
        *,
        object_position,
        tool_position,
        gripper_contacts,
        wrist_force=(0.0, 0.0, 0.0),
        wrist_torque=(0.0, 0.0, 0.0),
    ) -> SensorPacket:
        return SensorPacket(
            timestamp=float(self.data.time),
            joint_position=tuple(float(self.data.qpos[index]) for index in self.arm_qpos_addresses),
            joint_velocity=tuple(float(self.data.qvel[index]) for index in self.arm_dof_addresses),
            actuator_force=tuple(float(value) for value in self.data.actuator_force),
            wrist_force=tuple(float(value) for value in wrist_force),
            wrist_torque=tuple(float(value) for value in wrist_torque),
            object_position=tuple(float(value) for value in object_position),
            tool_position=tuple(float(value) for value in tool_position),
            gripper_contacts=tuple(bool(value) for value in gripper_contacts),
        )
