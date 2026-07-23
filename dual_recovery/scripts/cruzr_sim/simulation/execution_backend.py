"""MuJoCo implementation of the portable dual-arm execution boundary."""

import mujoco
import numpy as np
import math

from cruzr_sim.planning.dual_arm_execution import plan_dual_goal
from cruzr_sim.tasks.execution_backend import (
    ArmTelemetry,
    DualArmPlan,
    ExecutionResult,
    MotionConstraints,
    PortableDualArmObservation,
    Pose6D,
)


class MuJoCoDualArmBackend:
    """Adapts simulation state and commands without exposing them to task logic."""

    def __init__(
        self, model, data, left_solver, right_solver, left_planner, right_planner,
        left_gripper, right_gripper, camera, object_body, object_geom,
    ):
        self.model = model
        self.data = data
        self.left_solver = left_solver
        self.right_solver = right_solver
        self.left_planner = left_planner
        self.right_planner = right_planner
        self.left_gripper = left_gripper
        self.right_gripper = right_gripper
        self.camera = camera
        self.object_body = int(object_body)
        self.object_geom = int(object_geom)
        self.stopped_reason = None
        self.latest_valid_detection = camera.latest_detection

    @staticmethod
    def _pose(position, quaternion, *, confidence=1.0):
        return Pose6D(
            tuple(float(value) for value in position),
            tuple(float(value) for value in quaternion),
            confidence=float(confidence),
        )

    @staticmethod
    def _rpy_to_quaternion(rpy):
        roll, pitch, yaw = (float(value) for value in rpy)
        cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
        cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
        cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
        return (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        )

    def observe(self):
        detection = self.camera.latest_detection
        if detection is not None:
            self.latest_valid_detection = detection
        else:
            detection = self.latest_valid_detection
        left_contacts = tuple(self.left_gripper.contact_flags(self.object_geom))
        right_contacts = tuple(self.right_gripper.contact_flags(self.object_geom))
        left_force = self.left_gripper.contact_diagnostics(self.object_geom)[
            "normal_force"
        ]
        right_force = self.right_gripper.contact_diagnostics(self.object_geom)[
            "normal_force"
        ]
        object_pose = None
        if detection is not None:
            # Perception remains the grasping source; MuJoCo ground truth is metadata only.
            object_pose = Pose6D(
                tuple(float(value) for value in detection.position),
                self._rpy_to_quaternion(detection.rpy),
                frame_id="base_link",
                confidence=float(detection.confidence),
            )
        return PortableDualArmObservation(
            timestamp=float(self.data.time),
            left=ArmTelemetry(
                tuple(float(self.data.qpos[index]) for index in self.left_solver.qpos_adrs),
                self._pose(
                    self.left_solver.tool_position(self.data),
                    self.data.xquat[self.left_solver.body_id],
                ),
                contact=any(left_contacts),
                pad_contacts=left_contacts,
                force=(0.0, 0.0, float(left_force)),
            ),
            right=ArmTelemetry(
                tuple(float(self.data.qpos[index]) for index in self.right_solver.qpos_adrs),
                self._pose(
                    self.right_solver.tool_position(self.data),
                    self.data.xquat[self.right_solver.body_id],
                ),
                contact=any(right_contacts),
                pad_contacts=right_contacts,
                force=(0.0, 0.0, float(right_force)),
            ),
            object_pose=object_pose,
            object_linear_velocity=tuple(
                float(self.data.qvel[
                    self.model.jnt_dofadr[
                        self.model.body_jntadr[self.object_body]
                    ] + index
                ])
                for index in range(3)
            ),
            metadata={
                "simulated_object_position": tuple(
                    float(value) for value in self.data.xpos[self.object_body]
                ),
            },
        )

    def plan_dual_pose(
        self, left_pose, right_pose, constraints, *, source_node,
    ):
        if not isinstance(constraints, MotionConstraints):
            raise TypeError("constraints must be MotionConstraints")
        path = plan_dual_goal(
            self.data.qpos.copy(),
            np.asarray(left_pose.position, dtype=float),
            np.asarray(right_pose.position, dtype=float),
            self.left_solver,
            self.right_solver,
            self.left_planner,
            self.right_planner,
        )
        duration = max(0.05, 0.1 / max(constraints.maximum_velocity_scale, 1e-3))
        return DualArmPlan(
            tuple(tuple(float(point[index]) for index in self.left_solver.qpos_adrs) for point in path),
            tuple(tuple(float(point[index]) for index in self.right_solver.qpos_adrs) for point in path),
            tuple(duration for _ in path),
            source_node,
        )

    def execute_waypoint(
        self, left_joints, right_joints, duration, *, source_node,
    ):
        if self.stopped_reason is not None:
            return ExecutionResult(False, f"backend stopped: {self.stopped_reason}")
        if duration <= 0.0:
            return ExecutionResult(False, "duration must be positive")
        if len(left_joints) != len(self.left_solver.qpos_adrs):
            return ExecutionResult(False, "left joint vector length mismatch")
        if len(right_joints) != len(self.right_solver.qpos_adrs):
            return ExecutionResult(False, "right joint vector length mismatch")
        targets = tuple(left_joints) + tuple(right_joints)
        if not all(np.isfinite(value) for value in targets):
            return ExecutionResult(False, "non-finite joint target")
        joint_ids = tuple(self.left_solver.joint_ids) + tuple(
            self.right_solver.joint_ids
        )
        for joint_id, value in zip(joint_ids, targets):
            if self.model.jnt_limited[joint_id]:
                lower, upper = self.model.jnt_range[joint_id]
                if not float(lower) <= value <= float(upper):
                    self.stop("joint limit violation")
                    return ExecutionResult(False, "joint limit violation")

        qpos_adrs = tuple(self.left_solver.qpos_adrs) + tuple(
            self.right_solver.qpos_adrs
        )
        starts = np.asarray([self.data.qpos[index] for index in qpos_adrs])
        targets = np.asarray(targets, dtype=float)
        step_count = max(1, int(math.ceil(duration / self.model.opt.timestep)))
        for step in range(1, step_count + 1):
            values = starts + (step / step_count) * (targets - starts)
            for index, value in zip(qpos_adrs, values):
                self.data.qpos[index] = value
            for joint_id in joint_ids:
                self.data.qvel[self.model.jnt_dofadr[joint_id]] = 0.0
            mujoco.mj_step(self.model, self.data)
        return ExecutionResult(True, "accepted")

    def set_grippers(self, left_closed, right_closed):
        (self.left_gripper.close if left_closed else self.left_gripper.open)()
        (self.right_gripper.close if right_closed else self.right_gripper.open)()
        return ExecutionResult(True, "accepted")

    def set_camera_view(self, view):
        if view == "waist_search":
            view = "waist"
            self.camera.views[view]["pitch_down"] = 35.0
        if view not in self.camera.views:
            return ExecutionResult(False, f"unknown camera view: {view}")
        self.camera.active_view = view
        return ExecutionResult(True, "accepted")

    def stop(self, reason):
        self.stopped_reason = str(reason)
        for joint_id in tuple(self.left_solver.joint_ids) + tuple(
            self.right_solver.joint_ids
        ):
            self.data.qvel[self.model.jnt_dofadr[joint_id]] = 0.0
