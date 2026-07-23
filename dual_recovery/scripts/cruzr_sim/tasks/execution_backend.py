"""Portable execution contract shared by simulation and real-robot backends."""

from dataclasses import dataclass, field
import math
from typing import Mapping, Protocol, Sequence, runtime_checkable


@dataclass(frozen=True)
class Pose6D:
    position: tuple[float, float, float]
    quaternion: tuple[float, float, float, float]
    frame_id: str = "base_link"
    confidence: float = 1.0

    @property
    def rpy(self):
        w, x, y, z = self.quaternion
        roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return roll, pitch, yaw


@dataclass(frozen=True)
class ArmTelemetry:
    joints: tuple[float, ...]
    tool_pose: Pose6D
    contact: bool = False
    pad_contacts: tuple[bool, ...] = ()
    force: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class PortableDualArmObservation:
    timestamp: float
    left: ArmTelemetry
    right: ArmTelemetry
    object_pose: Pose6D | None = None
    object_linear_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class MotionConstraints:
    maximum_velocity_scale: float = 0.2
    minimum_clearance: float = 0.05
    synchronized: bool = True


@dataclass(frozen=True)
class DualArmPlan:
    left_waypoints: tuple[tuple[float, ...], ...]
    right_waypoints: tuple[tuple[float, ...], ...]
    durations: tuple[float, ...]
    source_node: str


@dataclass(frozen=True)
class ExecutionResult:
    accepted: bool
    reason: str


@runtime_checkable
class DualArmExecutionBackend(Protocol):
    """The task/recovery layer depends only on this SDK-neutral interface."""

    def observe(self) -> PortableDualArmObservation: ...

    def plan_dual_pose(
        self,
        left_pose: Pose6D,
        right_pose: Pose6D,
        constraints: MotionConstraints,
        *,
        source_node: str,
    ) -> DualArmPlan: ...

    def execute_waypoint(
        self,
        left_joints: Sequence[float],
        right_joints: Sequence[float],
        duration: float,
        *,
        source_node: str,
    ) -> ExecutionResult: ...

    def set_grippers(self, left_closed: bool, right_closed: bool) -> ExecutionResult: ...

    def set_camera_view(self, view: str) -> ExecutionResult: ...

    def stop(self, reason: str) -> None: ...
