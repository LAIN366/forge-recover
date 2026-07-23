"""Reusable dual-arm path composition and planning operations."""

import numpy as np

from .dual_arm_coordinator import synchronize_paths


def combine_synchronized_paths(
    template, left_solver, right_solver, left_path, right_path,
):
    left_joints = [waypoint[left_solver.qpos_adrs] for waypoint in left_path]
    right_joints = [waypoint[right_solver.qpos_adrs] for waypoint in right_path]
    synchronized = synchronize_paths(left_joints, right_joints)
    result = []
    for left_waypoint, right_waypoint in zip(
        synchronized.left_waypoints, synchronized.right_waypoints,
    ):
        combined = template.copy()
        combined[left_solver.qpos_adrs] = left_waypoint
        combined[right_solver.qpos_adrs] = right_waypoint
        result.append(combined)
    return result


def plan_dual_goal(
    robot_qpos,
    left_target,
    right_target,
    left_solver,
    right_solver,
    left_planner,
    right_planner,
    allow_object=False,
    allow_start_in_collision=False,
    precomputed_joint_goal=None,
):
    if precomputed_joint_goal is None:
        left_result = left_solver.solve(robot_qpos, left_target)
        right_result = right_solver.solve(left_result.qpos, right_target)
        if not left_result.success or not right_result.success:
            raise RuntimeError(
                "dual IK failed: "
                f"left={left_result.error_norm:.4f}, right={right_result.error_norm:.4f}"
            )
        joint_goal = right_result.qpos
    else:
        joint_goal = np.asarray(precomputed_joint_goal, dtype=float).copy()
        if joint_goal.shape != np.asarray(robot_qpos).shape:
            raise ValueError("precomputed joint goal has the wrong shape")
    left_plan = left_planner.plan(
        robot_qpos,
        joint_goal,
        allow_cube=allow_object,
        allow_start_in_collision=allow_start_in_collision,
    )
    right_plan = right_planner.plan(
        robot_qpos,
        joint_goal,
        allow_cube=allow_object,
        allow_start_in_collision=allow_start_in_collision,
    )
    if not left_plan.success or not right_plan.success:
        raise RuntimeError(
            f"dual planning failed: left={left_plan.method}, right={right_plan.method}, "
            f"left_reasons={left_planner.configuration_collision_reasons(joint_goal, allow_object)}, "
            f"right_reasons={right_planner.configuration_collision_reasons(joint_goal, allow_object)}"
        )
    return combine_synchronized_paths(
        robot_qpos,
        left_solver,
        right_solver,
        left_plan.waypoints,
        right_plan.waypoints,
    )


def plan_contact_anchored_regrasp(
    robot_qpos, failed_side, support_tool_position, object_rotation,
    observed_center, nominal_span, tool_object_offset_z, left_solver, right_solver,
    left_planner, right_planner,
):
    """Search geometry-scaled regrasp targets while preserving support contact."""
    failures = []
    for span_scale in (1.0, 0.88, 0.76, 0.64):
        span = float(nominal_span) * span_scale
        for height_offset in (0.0, 0.015, -0.012, 0.030):
            if failed_side == "left":
                right_target = np.asarray(support_tool_position, dtype=float)
                anchored_center = right_target - object_rotation @ np.array([
                    span, 0.0, tool_object_offset_z,
                ])
            elif failed_side == "right":
                left_target = np.asarray(support_tool_position, dtype=float)
                anchored_center = left_target - object_rotation @ np.array([
                    -span, 0.0, tool_object_offset_z,
                ])
            else:
                raise ValueError(f"invalid failed side: {failed_side}")
            for center in (np.asarray(observed_center), anchored_center):
                if failed_side == "left":
                    left_target = center + object_rotation @ np.array([
                        -span, 0.0, tool_object_offset_z,
                    ]) + np.array([0.0, 0.0, height_offset])
                else:
                    right_target = center + object_rotation @ np.array([
                        span, 0.0, tool_object_offset_z,
                    ]) + np.array([0.0, 0.0, height_offset])
                try:
                    path = plan_dual_goal(
                        robot_qpos, left_target, right_target,
                        left_solver, right_solver, left_planner, right_planner,
                        allow_object=True,
                    )
                except RuntimeError as error:
                    failures.append(str(error))
                    continue
                return path, left_target, right_target, center, span
    raise RuntimeError(
        "no contact-anchored regrasp candidate is feasible: "
        + "; ".join(failures[-4:])
    )


def tracking_error(robot_qpos, goal, addresses):
    if goal is None:
        return 0.0
    return float(np.linalg.norm(goal[addresses] - robot_qpos[addresses]))


def trajectory_phase(current, start, final_goal, addresses):
    if start is None or final_goal is None:
        return 1.0
    path_length = float(np.linalg.norm(final_goal[addresses] - start[addresses]))
    if path_length < 1e-8:
        return 1.0
    remaining = float(np.linalg.norm(final_goal[addresses] - current[addresses]))
    return float(np.clip(1.0 - remaining / path_length, 0.0, 1.0))
