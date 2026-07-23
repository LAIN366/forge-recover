"""Cruzr SDK backend skeleton with fail-closed real-motion gating."""

from typing import Protocol, Sequence

from cruzr_sim.tasks.execution_backend import (
    DualArmExecutionBackend,
    DualArmPlan,
    ExecutionResult,
    MotionConstraints,
    PortableDualArmObservation,
    Pose6D,
)

from .real_robot import JointCommand, SafetyGatedCommandPort


class CruzrSdkDriver(Protocol):
    """Minimal wrapper to implement against the vendor SDK or ROS 2 bridge."""

    def read_observation(self) -> PortableDualArmObservation: ...
    def plan_dual_pose(
        self, left_pose: Pose6D, right_pose: Pose6D,
        constraints: MotionConstraints, source_node: str,
    ) -> DualArmPlan: ...
    def set_grippers(self, left_closed: bool, right_closed: bool) -> bool: ...
    def set_camera_view(self, view: str) -> bool: ...
    def stop(self, reason: str) -> None: ...


class CruzrSdkBackend(DualArmExecutionBackend):
    """Hardware adapter; motion remains disabled until explicitly armed."""

    def __init__(self, driver: CruzrSdkDriver, command_port: SafetyGatedCommandPort):
        self.driver = driver
        self.command_port = command_port
        self.operator_enabled = False

    def set_operator_enabled(self, enabled: bool) -> None:
        self.operator_enabled = bool(enabled)
        if not self.operator_enabled:
            self.stop("operator enable is false")

    def observe(self) -> PortableDualArmObservation:
        observation = self.driver.read_observation()
        self.command_port.heartbeat()
        return observation

    def plan_dual_pose(
        self, left_pose: Pose6D, right_pose: Pose6D,
        constraints: MotionConstraints, *, source_node: str,
    ) -> DualArmPlan:
        return self.driver.plan_dual_pose(
            left_pose, right_pose, constraints, source_node
        )

    def execute_waypoint(
        self, left_joints: Sequence[float], right_joints: Sequence[float],
        duration: float, *, source_node: str,
    ) -> ExecutionResult:
        if not self.operator_enabled:
            self.stop("operator enable is false")
            return ExecutionResult(False, "operator enable is false")

        commands = (
            JointCommand("left", tuple(left_joints), duration, source_node),
            JointCommand("right", tuple(right_joints), duration, source_node),
        )
        accepted, reason = self.command_port.execute_batch(
            commands, operator_enabled=self.operator_enabled
        )
        return ExecutionResult(accepted, reason)

    def set_grippers(self, left_closed: bool, right_closed: bool) -> ExecutionResult:
        if not self.operator_enabled:
            self.stop("operator enable is false")
            return ExecutionResult(False, "operator enable is false")
        accepted = self.driver.set_grippers(left_closed, right_closed)
        return ExecutionResult(accepted, "accepted" if accepted else "SDK rejected gripper command")

    def set_camera_view(self, view: str) -> ExecutionResult:
        if not self.operator_enabled:
            self.stop("operator enable is false")
            return ExecutionResult(False, "operator enable is false")
        accepted = self.driver.set_camera_view(view)
        return ExecutionResult(accepted, "accepted" if accepted else "SDK rejected camera command")

    def stop(self, reason: str) -> None:
        self.driver.stop(reason)
