"""Physical dynamic-obstacle placement for MuJoCo path invalidation tests."""

import mujoco
import numpy as np


def place_dynamic_obstacle(
    data,
    mocap_id,
    left_planner,
    right_planner,
    left_solver,
    right_solver,
    current_qpos,
    remaining_path,
    preferred_fraction,
    side,
):
    """Place an obstacle on a future arm path while preserving safe endpoints."""
    def reasons(planner, configuration):
        return tuple(
            reason
            for reason in planner.configuration_collision_reasons(
                configuration, allow_cube=True
            )
            if "dynamic_obstacle" in reason
        )

    def hits(planner, configuration, include_bounds=True):
        collisions = reasons(planner, configuration)
        if not include_bounds:
            collisions = tuple(
                reason for reason in collisions
                if not reason.startswith("bounds:")
            )
        return bool(collisions)

    if len(remaining_path) < 3:
        raise RuntimeError("dynamic obstacle requires an unfinished path")
    preferred_index = min(
        len(remaining_path) - 2,
        max(1, int(float(preferred_fraction) * len(remaining_path))),
    )
    candidate_indices = sorted(
        range(1, len(remaining_path) - 1),
        key=lambda index: abs(index - preferred_index),
    )
    selected_planner = left_planner if side == "left" else right_planner
    selected_solver = left_solver if side == "left" else right_solver
    terminal = remaining_path[-1]
    current_tool = selected_solver.tool_position_for_qpos(current_qpos)
    terminal_tool = selected_solver.tool_position_for_qpos(terminal)
    diagnostics = []
    outward = -1.0 if side == "left" else 1.0
    placement_offsets = (
        np.array([0.07 * outward, 0.0, 0.07]),
        np.array([-0.07 * outward, 0.0, 0.07]),
        np.array([0.0, 0.0, 0.07]),
        np.array([0.07 * outward, 0.0, 0.04]),
        np.array([-0.07 * outward, 0.0, 0.04]),
    )
    for offset in placement_offsets:
        for path_index in candidate_indices:
            position = selected_solver.tool_position_for_qpos(
                remaining_path[path_index]
            ) + offset
            data.mocap_pos[mocap_id] = position
            for planner in (left_planner, right_planner):
                planner.sync_mocap_state(data)
            endpoint_hits = {
                name: (
                    hits(planner, current_qpos, include_bounds=False),
                    hits(planner, terminal),
                )
                for name, planner in (
                    ("left", left_planner), ("right", right_planner),
                )
            }
            endpoint_distances = (
                round(float(np.linalg.norm(position - current_tool)), 4),
                round(float(np.linalg.norm(position - terminal_tool)), 4),
            )
            if any(hit for pair in endpoint_hits.values() for hit in pair):
                diagnostics.append((
                    path_index, offset.round(3).tolist(), endpoint_distances,
                    endpoint_hits, None,
                ))
                continue
            blocked = {
                name: sum(hits(planner, waypoint) for waypoint in remaining_path)
                for name, planner in (
                    ("left", left_planner), ("right", right_planner),
                )
            }
            if blocked[side] > 0:
                mujoco.mj_forward(left_planner.model, data)
                return np.asarray(position, dtype=float).copy(), blocked
            diagnostics.append((
                path_index, offset.round(3).tolist(), endpoint_distances,
                endpoint_hits, blocked,
            ))
    raise RuntimeError(
        "could not place dynamic obstacle on a future path with safe endpoints: "
        f"path_length={len(remaining_path)}, "
        f"candidates={diagnostics[:4] + diagnostics[-4:]}"
    )
