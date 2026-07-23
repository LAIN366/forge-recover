"""Position-only damped least-squares inverse kinematics for Cruzr S2."""

from dataclasses import dataclass

import mujoco
import numpy as np


def arm_joint_names(side):
    """Return the Cruzr seven-joint chain for ``L`` or ``R``."""
    side = str(side).upper()
    if side not in {"L", "R"}:
        raise ValueError("side must be 'L' or 'R'")
    return tuple(
        f"{side}_{name}_joint"
        for name in (
            "shoulder_pitch", "shoulder_roll", "shoulder_yaw",
            "elbow_roll", "elbow_yaw", "wrist_pitch", "wrist_roll",
        )
    )


def eef_body_candidates(side):
    """Return end-effector bodies from the most to least accurate tool frame."""
    side = str(side).upper()
    if side not in {"L", "R"}:
        raise ValueError("side must be 'L' or 'R'")
    return (
        f"{side}_pgc_base_link",
        f"{side}_sixforce_link",
        f"{side}_wrist_roll_link",
    )


RIGHT_ARM_JOINTS = arm_joint_names("R")
LEFT_ARM_JOINTS = arm_joint_names("L")
RIGHT_EEF_BODY_CANDIDATES = eef_body_candidates("R")
LEFT_EEF_BODY_CANDIDATES = eef_body_candidates("L")


def rpy_matrix(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


# URDF fixed chain: wrist -> six-force sensor -> gripper base -> finger midpoint.
_SENSOR_POS = np.array([0.0, 0.07712, 0.0])
_SENSOR_ROT = rpy_matrix(-np.pi / 2, np.pi / 2, 0.0)
_GRIPPER_ROT = rpy_matrix(0.0, 0.0, np.pi / 2)
_FINGER_MIDPOINT = np.array([-0.00001362, -0.0018605, 0.22048])


@dataclass
class IKResult:
    success: bool
    qpos: np.ndarray
    error_norm: float
    iterations: int


class DampedLeastSquaresIK:
    def __init__(
        self,
        model: mujoco.MjModel,
        eef_body: str | None = None,
        joint_names=None,
        side: str = "R",
        damping: float = 0.03,
        max_step: float = 0.08,
        tolerance: float = 0.005,
        max_iterations: int = 200,
    ):
        self.model = model
        self.side = str(side).upper()
        if self.side not in {"L", "R"}:
            raise ValueError("side must be 'L' or 'R'")
        self.damping = damping
        self.max_step = max_step
        self.tolerance = tolerance
        self.max_iterations = max_iterations

        candidates = (eef_body,) if eef_body else eef_body_candidates(self.side)
        self.body_id = -1
        self.eef_body_name = None
        for candidate in candidates:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, candidate)
            if body_id >= 0:
                self.body_id = body_id
                self.eef_body_name = candidate
                break
        if self.body_id < 0:
            raise ValueError(
                f"No {self.side} end-effector body found; tried: {candidates}"
            )
        self.tool_offset_local = self._tool_offset_for_body(self.eef_body_name)

        joint_names = arm_joint_names(self.side) if joint_names is None else joint_names
        self.joint_ids = []
        for name in joint_names:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"Arm joint not found: {name}")
            self.joint_ids.append(joint_id)

        self.qpos_adrs = np.array(
            [model.jnt_qposadr[joint_id] for joint_id in self.joint_ids], dtype=int
        )
        self.dof_adrs = np.array(
            [model.jnt_dofadr[joint_id] for joint_id in self.joint_ids], dtype=int
        )

    @staticmethod
    def _tool_offset_for_body(body_name):
        if body_name.endswith("_pgc_base_link"):
            return _FINGER_MIDPOINT.copy()
        if body_name.endswith("_sixforce_link"):
            return _GRIPPER_ROT @ _FINGER_MIDPOINT
        # Wrist fallback: include both fixed transforms.
        return _SENSOR_POS + _SENSOR_ROT @ (_GRIPPER_ROT @ _FINGER_MIDPOINT)

    def tool_position(self, data):
        rotation = data.xmat[self.body_id].reshape(3, 3)
        return data.xpos[self.body_id] + rotation @ self.tool_offset_local

    def tool_position_for_qpos(self, qpos):
        """Evaluate the tool position at a full robot configuration."""
        data = mujoco.MjData(self.model)
        data.qpos[:] = np.asarray(qpos, dtype=float)
        data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, data)
        return self.tool_position(data)

    def solve(self, seed_qpos: np.ndarray, target_pos) -> IKResult:
        """Return a full qpos vector with the selected arm moved to target_pos."""
        target_pos = np.asarray(target_pos, dtype=float)
        if target_pos.shape != (3,):
            raise ValueError("target_pos must have shape (3,)")

        data = mujoco.MjData(self.model)
        data.qpos[:] = np.asarray(seed_qpos, dtype=float).copy()
        data.qvel[:] = 0.0

        jacobian_pos = np.zeros((3, self.model.nv))
        jacobian_rot = np.zeros((3, self.model.nv))

        for iteration in range(1, self.max_iterations + 1):
            mujoco.mj_forward(self.model, data)
            tool_pos = self.tool_position(data)
            error = target_pos - tool_pos
            error_norm = float(np.linalg.norm(error))
            if error_norm <= self.tolerance:
                return IKResult(True, data.qpos.copy(), error_norm, iteration)

            jacobian_pos.fill(0.0)
            jacobian_rot.fill(0.0)
            mujoco.mj_jac(self.model, data, jacobian_pos, jacobian_rot, tool_pos, self.body_id)
            jacobian = jacobian_pos[:, self.dof_adrs]

            # dq = J^T (J J^T + lambda^2 I)^-1 e
            regularized = jacobian @ jacobian.T + (self.damping ** 2) * np.eye(3)
            dq = jacobian.T @ np.linalg.solve(regularized, error)
            dq = np.clip(dq, -self.max_step, self.max_step)

            data.qpos[self.qpos_adrs] += dq
            self._clip_joint_limits(data.qpos)

        mujoco.mj_forward(self.model, data)
        error_norm = float(np.linalg.norm(target_pos - self.tool_position(data)))
        return IKResult(False, data.qpos.copy(), error_norm, self.max_iterations)

    def _clip_joint_limits(self, qpos: np.ndarray) -> None:
        for joint_id, qpos_adr in zip(self.joint_ids, self.qpos_adrs):
            if self.model.jnt_limited[joint_id]:
                lower, upper = self.model.jnt_range[joint_id]
                qpos[qpos_adr] = np.clip(qpos[qpos_adr], lower, upper)
