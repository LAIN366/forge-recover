"""Transport-neutral Cruzr S2 ROS 2 telemetry adapter.

The actual rclpy subscriber remains deployment-specific because UBTECH custom
message packages are available only inside the robot container. This module is
fully testable without ROS and defines the exact boundary that subscriber uses.
"""

from dataclasses import dataclass, field
from typing import Any

from cruzr_sim.diagnosis.types import ManipulationObservation


@dataclass(frozen=True)
class CruzrTopicMap:
    robot_state: str = "/mc/sdk/robot_state"
    robot_command: str = "/mc/sdk/robot_command"
    right_gripper_state: str = "/ecat/right_grip/state"
    right_gripper_command: str = "/ecat/right_grip/cmd"
    front_rgb: str = "/sensor/camera/chassis_front_rgbd/color/image_raw"
    front_depth: str = "/sensor/camera/chassis_front_rgbd/depth/image_raw"


@dataclass(frozen=True)
class CruzrTelemetry:
    timestamp: float
    stage: str
    object_position: tuple[float, float, float]
    tool_position: tuple[float, float, float]
    joint_position: tuple[float, ...] = ()
    joint_velocity: tuple[float, ...] = ()
    wrench_force: tuple[float, float, float] = (0.0, 0.0, 0.0)
    wrench_torque: tuple[float, float, float] = (0.0, 0.0, 0.0)
    left_contact: bool = False
    right_contact: bool = False
    gripper_current: float = 0.0
    object_vertical_velocity: float = 0.0
    target_position: tuple[float, float, float] | None = None
    ik_success: bool = True
    planning_success: bool = True
    collision: bool = False
    sensor_valid: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


class CruzrObservationAssembler:
    """Convert synchronized ROS 2 telemetry to the common diagnosis contract."""

    def build(self, telemetry: CruzrTelemetry) -> ManipulationObservation:
        normal_force = sum(value * value for value in telemetry.wrench_force) ** 0.5
        tangent_force = (
            telemetry.wrench_force[0] ** 2 + telemetry.wrench_force[1] ** 2
        ) ** 0.5
        return ManipulationObservation(
            timestamp=telemetry.timestamp,
            stage=telemetry.stage,
            object_position=telemetry.object_position,
            tool_position=telemetry.tool_position,
            left_contact=telemetry.left_contact,
            right_contact=telemetry.right_contact,
            normal_force=normal_force,
            tangent_force=tangent_force,
            vertical_force=telemetry.wrench_force[2],
            object_vertical_velocity=telemetry.object_vertical_velocity,
            target_position=telemetry.target_position,
            ik_success=telemetry.ik_success,
            planning_success=telemetry.planning_success,
            collision=telemetry.collision,
            sensor_valid=telemetry.sensor_valid,
            metadata={
                **telemetry.metadata,
                "joint_position": telemetry.joint_position,
                "joint_velocity": telemetry.joint_velocity,
                "wrench_torque": telemetry.wrench_torque,
                "gripper_current": telemetry.gripper_current,
            },
        )
