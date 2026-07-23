"""Low-level trajectory stepping and rotation conversion utilities."""

import numpy as np


DEFAULT_MAX_JOINT_DELTA = 0.0012


def step_toward(current, goal, maximum_delta=DEFAULT_MAX_JOINT_DELTA):
    current[:] += np.clip(goal - current, -maximum_delta, maximum_delta)
    return bool(np.max(np.abs(goal - current)) < 0.003)


def step_dual_toward(
    current,
    goal,
    left_addresses,
    right_addresses,
    frozen_arm=None,
    maximum_delta=DEFAULT_MAX_JOINT_DELTA,
):
    delta = np.clip(goal - current, -maximum_delta, maximum_delta)
    if frozen_arm == "left":
        delta[left_addresses] = 0.0
    elif frozen_arm == "right":
        delta[right_addresses] = 0.0
    current[:] += delta
    return bool(np.max(np.abs(goal - current)) < 0.003)


def matrix_to_rpy(matrix):
    rotation = np.asarray(matrix, dtype=float).reshape(3, 3)
    pitch = np.arcsin(np.clip(-rotation[2, 0], -1.0, 1.0))
    if abs(np.cos(pitch)) > 1e-6:
        roll = np.arctan2(rotation[2, 1], rotation[2, 2])
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = np.arctan2(-rotation[1, 2], rotation[1, 1])
        yaw = 0.0
    return float(roll), float(pitch), float(yaw)


def rpy_to_matrix(rpy):
    roll, pitch, yaw = (float(value) for value in rpy)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])
