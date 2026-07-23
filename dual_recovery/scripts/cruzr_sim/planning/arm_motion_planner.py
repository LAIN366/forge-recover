"""Collision-aware joint-space planning for the Cruzr S2 right arm."""

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass
class PlannedPath:
    success: bool
    waypoints: list
    method: str


class ArmMotionPlanner:
    """Checks MuJoCo contacts and falls back to bidirectional RRT-Connect."""

    def __init__(self, model, arm_joint_ids, arm_geom_ids, obstacle_geom_ids,
                 cube_geom_id, execution_arm_geom_ids=None, tool_body_id=None,
                 tool_offset_local=None, tool_radius=0.07,
                 tool_half_extent=None,
                 arm_radius_cap=0.11, arm_obstacle_margin=0.006, seed=7):
        self.model = model
        self.data = mujoco.MjData(model)
        self.joint_ids = np.asarray(arm_joint_ids, dtype=int)
        self.qpos_adrs = np.asarray(
            [model.jnt_qposadr[joint_id] for joint_id in self.joint_ids], dtype=int
        )
        self.arm_geoms = set(np.asarray(arm_geom_ids, dtype=int).tolist())
        execution_geoms = arm_geom_ids if execution_arm_geom_ids is None else execution_arm_geom_ids
        self.execution_arm_geoms = set(np.asarray(execution_geoms, dtype=int).tolist())
        self.obstacle_geoms = set(np.asarray(obstacle_geom_ids, dtype=int).tolist())
        self.cube_geom = int(cube_geom_id)
        self.tool_body_id = tool_body_id
        self.tool_offset_local = (
            None if tool_offset_local is None else np.asarray(tool_offset_local, dtype=float)
        )
        self.tool_radius = float(tool_radius)
        self.tool_half_extent = (
            None
            if tool_half_extent is None
            else np.asarray(tool_half_extent, dtype=float)
        )
        self.arm_radius_cap = float(arm_radius_cap)
        self.arm_obstacle_margin = float(arm_obstacle_margin)
        self.rng = np.random.default_rng(seed)
        self.lower = np.array([
            model.jnt_range[joint_id, 0] if model.jnt_limited[joint_id] else -np.pi
            for joint_id in self.joint_ids
        ])
        self.upper = np.array([
            model.jnt_range[joint_id, 1] if model.jnt_limited[joint_id] else np.pi
            for joint_id in self.joint_ids
        ])

    def sync_mocap_state(self, source_data):
        """Synchronize externally moved obstacle poses into planner state."""
        if self.model.nmocap == 0:
            return
        self.data.mocap_pos[:] = source_data.mocap_pos
        self.data.mocap_quat[:] = source_data.mocap_quat

    def configuration_in_collision(self, full_qpos, allow_cube=False):
        return bool(self.configuration_collision_reasons(full_qpos, allow_cube))

    def configuration_collision_reasons(self, full_qpos, allow_cube=False):
        """Return structured reasons for rejecting a joint configuration."""
        self.data.qpos[:] = full_qpos
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        reasons = []
        for obstacle_geom in self._tool_obstacle_collisions():
            obstacle_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, obstacle_geom,
            ) or "unnamed_obstacle_geom"
            reasons.append(f"tool_bounds:{obstacle_name}")
        for arm_geom, obstacle_geom in self._arm_bounds_collisions():
            arm_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, arm_geom,
            )
            if not arm_name:
                body_id = int(self.model.geom_bodyid[arm_geom])
                body_name = mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, body_id,
                ) or "unnamed_body"
                arm_name = f"body/{body_name}"
            obstacle_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, obstacle_geom,
            ) or "unnamed_obstacle_geom"
            reasons.append(f"bounds:{arm_name}:{obstacle_name}")
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            if geom1 in self.arm_geoms:
                arm_geom, other_geom = geom1, geom2
            elif geom2 in self.arm_geoms:
                arm_geom, other_geom = geom2, geom1
            else:
                continue
            del arm_geom
            if other_geom == self.cube_geom and allow_cube:
                continue
            if other_geom in self.obstacle_geoms or other_geom == self.cube_geom:
                arm_name = mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_GEOM,
                    int(contact.geom1) if int(contact.geom1) in self.arm_geoms
                    else int(contact.geom2),
                ) or "unnamed_arm_geom"
                other_name = mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_GEOM, other_geom,
                ) or "unnamed_obstacle_geom"
                reasons.append(f"contact:{arm_name}:{other_name}")
        return tuple(dict.fromkeys(reasons))

    def _tool_hits_obstacle(self):
        return bool(self._tool_obstacle_collisions())

    def _tool_obstacle_collisions(self):
        if self.tool_body_id is None or self.tool_offset_local is None:
            return []
        rotation = self.data.xmat[self.tool_body_id].reshape(3, 3)
        tool_position = (
            self.data.xpos[self.tool_body_id] + rotation @ self.tool_offset_local
        )
        collisions = []
        for geom_id in self.obstacle_geoms:
            if self.model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_BOX:
                continue
            half_size = self.model.geom_size[geom_id]
            geom_rotation = self.data.geom_xmat[geom_id].reshape(3, 3)
            local = geom_rotation.T @ (tool_position - self.data.geom_xpos[geom_id])
            if self.tool_half_extent is not None:
                projected_extent = (
                    np.abs(geom_rotation.T) @ self.tool_half_extent
                )
                if np.all(
                    np.abs(local)
                    <= half_size + projected_extent + 0.002
                ):
                    collisions.append(geom_id)
                continue
            closest = np.clip(local, -half_size, half_size)
            if np.linalg.norm(local - closest) <= self.tool_radius + 0.002:
                collisions.append(geom_id)
        return collisions

    def _arm_bounds_hit_obstacle(self):
        """Conservative mesh-vs-box check when STL contacts miss thin surfaces."""
        return bool(self._arm_bounds_collisions())

    def _arm_bounds_collisions(self):
        collisions = []
        for arm_geom in self.arm_geoms:
            center = self.data.geom_xpos[arm_geom]
            radius = min(
                float(self.model.geom_rbound[arm_geom]), self.arm_radius_cap
            )
            for obstacle_geom in self.obstacle_geoms:
                if self.model.geom_type[obstacle_geom] != mujoco.mjtGeom.mjGEOM_BOX:
                    continue
                half_size = self.model.geom_size[obstacle_geom]
                rotation = self.data.geom_xmat[obstacle_geom].reshape(3, 3)
                local = rotation.T @ (center - self.data.geom_xpos[obstacle_geom])
                closest = np.clip(local, -half_size, half_size)
                if (
                    np.linalg.norm(local - closest)
                    <= radius + self.arm_obstacle_margin
                ):
                    collisions.append((arm_geom, obstacle_geom))
        return collisions

    def current_arm_environment_collision(self, data):
        """Return the first forbidden right-arm/environment contact, if any."""
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            if geom1 in self.execution_arm_geoms and geom2 in self.obstacle_geoms:
                return geom1, geom2
            if geom2 in self.execution_arm_geoms and geom1 in self.obstacle_geoms:
                return geom2, geom1
        return None

    def edge_in_collision(self, template_qpos, start, goal, allow_cube=False,
                          resolution=0.035, allow_initial_collision=False):
        distance = float(np.max(np.abs(goal - start)))
        sample_count = max(2, int(np.ceil(distance / resolution)) + 1)
        candidate = template_qpos.copy()
        reached_free_space = False
        for alpha in np.linspace(0.0, 1.0, sample_count):
            candidate[self.qpos_adrs] = start + alpha * (goal - start)
            colliding = self.configuration_in_collision(candidate, allow_cube)
            if colliding and not (allow_initial_collision and not reached_free_space):
                return True
            reached_free_space = reached_free_space or not colliding
        return not reached_free_space

    def plan(self, start_qpos, goal_qpos, allow_cube=False, max_iterations=1800,
             allow_start_in_collision=False):
        start = start_qpos[self.qpos_adrs].copy()
        goal = goal_qpos[self.qpos_adrs].copy()
        if not self.edge_in_collision(
            start_qpos,
            start,
            goal,
            allow_cube,
            allow_initial_collision=allow_start_in_collision,
        ):
            waypoints = self._smooth_full_path(start_qpos, [start, goal])
            method = (
                "smoothed initial-contact escape"
                if allow_start_in_collision
                and self._state_collision(start_qpos, start, allow_cube)
                else "smoothed collision-free direct"
            )
            return PlannedPath(True, waypoints, method)
        if self._state_collision(start_qpos, goal, allow_cube):
            return PlannedPath(False, [], "goal in collision")

        first = [(start, -1)]
        second = [(goal, -1)]
        first_from_start = True
        step_size = 0.16
        for _ in range(max_iterations):
            sample = goal if self.rng.random() < 0.18 else self.rng.uniform(self.lower, self.upper)
            new_index = self._extend(first, sample, start_qpos, allow_cube, step_size)
            if new_index is not None:
                reached_index = self._connect(
                    second, first[new_index][0], start_qpos, allow_cube, step_size
                )
                if reached_index is not None:
                    first_path = self._trace(first, new_index)
                    second_path = self._trace(second, reached_index)
                    if first_from_start:
                        arm_path = first_path + list(reversed(second_path))[1:]
                    else:
                        arm_path = second_path + list(reversed(first_path))[1:]
                    arm_path = self._shortcut(arm_path, start_qpos, allow_cube)
                    return PlannedPath(
                        True, self._smooth_full_path(start_qpos, arm_path),
                        "smoothed RRT-Connect detour",
                    )
            first, second = second, first
            first_from_start = not first_from_start
        return PlannedPath(False, [], "RRT-Connect exhausted")

    def _state_collision(self, template, arm_qpos, allow_cube):
        candidate = template.copy()
        candidate[self.qpos_adrs] = arm_qpos
        return self.configuration_in_collision(candidate, allow_cube)

    def _extend(self, tree, target, template, allow_cube, step_size):
        distances = [np.linalg.norm(node - target) for node, _ in tree]
        parent = int(np.argmin(distances))
        source = tree[parent][0]
        delta = target - source
        norm = np.linalg.norm(delta)
        candidate = target.copy() if norm <= step_size else source + step_size * delta / norm
        if self.edge_in_collision(template, source, candidate, allow_cube):
            return None
        tree.append((candidate, parent))
        return len(tree) - 1

    def _connect(self, tree, target, template, allow_cube, step_size):
        for _ in range(80):
            index = self._extend(tree, target, template, allow_cube, step_size)
            if index is None:
                return None
            if np.linalg.norm(tree[index][0] - target) < 1e-3:
                return index
        return None

    @staticmethod
    def _trace(tree, index):
        path = []
        while index >= 0:
            node, index = tree[index]
            path.append(node)
        return list(reversed(path))

    def _full_waypoints(self, template, arm_path):
        result = []
        for arm_qpos in arm_path:
            waypoint = template.copy()
            waypoint[self.qpos_adrs] = arm_qpos
            result.append(waypoint)
        return result

    def _shortcut(self, path, template, allow_cube):
        """Remove unnecessary RRT corners without violating collisions."""
        shortened = [path[0]]
        index = 0
        while index < len(path) - 1:
            next_index = len(path) - 1
            while next_index > index + 1:
                if not self.edge_in_collision(
                    template, path[index], path[next_index], allow_cube,
                    resolution=0.025,
                ):
                    break
                next_index -= 1
            shortened.append(path[next_index])
            index = next_index
        return shortened

    def _smooth_full_path(self, template, arm_path, max_joint_step=0.009):
        """Densify segments with minimum-jerk time scaling."""
        result = []
        for start, goal in zip(arm_path[:-1], arm_path[1:]):
            distance = float(np.max(np.abs(goal - start)))
            sample_count = max(2, int(np.ceil(distance / max_joint_step)) + 1)
            for tau in np.linspace(0.0, 1.0, sample_count)[1:]:
                blend = 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5
                arm_qpos = start + blend * (goal - start)
                waypoint = template.copy()
                waypoint[self.qpos_adrs] = arm_qpos
                result.append(waypoint)
        return result
