"""Role assignment and synchronized waypoint generation for two arms."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ArmCapability:
    side: str
    reachable: bool
    clearance: float
    visibility: float
    motion_cost: float


@dataclass(frozen=True)
class RoleAssignment:
    primary: str
    support: str
    score: float
    rationale: str


@dataclass(frozen=True)
class SynchronizedPlan:
    left_waypoints: list
    right_waypoints: list
    coordination_steps: int


def _capability_score(capability):
    if not capability.reachable:
        return -np.inf
    return (
        0.45 * float(capability.clearance)
        + 0.35 * float(capability.visibility)
        - 0.20 * float(capability.motion_cost)
    )


def assign_primary_and_support(left, right):
    """Choose roles using reachability, clearance, visibility, and motion cost."""
    if left.side.lower() != "left" or right.side.lower() != "right":
        raise ValueError("capabilities must be ordered as left, right")
    left_score = _capability_score(left)
    right_score = _capability_score(right)
    if not np.isfinite(left_score) and not np.isfinite(right_score):
        raise RuntimeError("object is unreachable by both arms")
    if left_score >= right_score:
        primary, support, score = "left", "right", left_score
    else:
        primary, support, score = "right", "left", right_score
    return RoleAssignment(
        primary=primary,
        support=support,
        score=float(score),
        rationale="risk-aware capability score",
    )


def _resample_path(path, sample_count):
    array = np.asarray(path, dtype=float)
    if array.ndim != 2 or len(array) == 0:
        raise ValueError("path must contain one or more joint vectors")
    if len(array) == 1:
        return np.repeat(array, sample_count, axis=0)
    source = np.linspace(0.0, 1.0, len(array))
    target = np.linspace(0.0, 1.0, sample_count)
    result = np.empty((sample_count, array.shape[1]), dtype=float)
    for joint_index in range(array.shape[1]):
        result[:, joint_index] = np.interp(target, source, array[:, joint_index])
    return result


def synchronize_paths(left_path, right_path, minimum_steps=2):
    """Time-align independently planned paths without changing their endpoints."""
    steps = max(int(minimum_steps), len(left_path), len(right_path))
    left = _resample_path(left_path, steps)
    right = _resample_path(right_path, steps)
    return SynchronizedPlan(
        left_waypoints=[waypoint.copy() for waypoint in left],
        right_waypoints=[waypoint.copy() for waypoint in right],
        coordination_steps=steps,
    )
