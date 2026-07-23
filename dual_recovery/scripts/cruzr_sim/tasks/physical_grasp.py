#!/usr/bin/env python3
"""Real-contact cube pickup using an actuator-driven dynamic gripper."""

import argparse
import json
from pathlib import Path
import time

import glfw
import mujoco
import mujoco_viewer
import numpy as np

from collect_demo import (
    ASSET_XML, BASE_INITIAL_X, BASE_INITIAL_Y, BASE_INITIAL_YAW, BASE_MJCF,
    SCENE_MJCF, URDF_FILE, WORLD_XML, insert_before_close, move_base,
    prepare_robot_scene,
)
from cruzr_sim.control.dynamic_gripper import (
    DynamicGripper, GRIPPER_MODEL_XML, GRIPPER_WORLDBODY_XML,
)
from cruzr_sim.control.ik_solver import DampedLeastSquaresIK
from cruzr_sim.diagnosis.operation_monitor import OperationMonitor
from cruzr_sim.diagnosis.types import FaultType
from cruzr_sim.faults import FaultInjectionConfig, FaultInjector, FaultScenario
from cruzr_sim.experiments import ExperimentPolicy
from cruzr_sim.perception import (
    BlueCubeRgbdDetector,
    HeadCameraWindow,
    MujocoSensorSuite,
)
from cruzr_sim.planning.arm_motion_planner import ArmMotionPlanner
from cruzr_sim.recovery import RecoveryVerifier
from cruzr_sim.tasks.active_probe_runtime import ActiveProbeRuntime
from cruzr_sim.tasks.grasp_observation import build_grasp_observation
from cruzr_sim.tasks.grasp_recovery import (
    GraspRecoveryDirective,
    compile_grasp_recovery,
)
from cruzr_sim.tasks.manipulation_supervisor import ManipulationSupervisor


LIFT_TARGET_STEP = 0.000035


COLLISION_ROBOT = 1
COLLISION_CUBE = 2
COLLISION_GRIPPER = 4
COLLISION_ENVIRONMENT = 8

DYNAMIC_GRIPPER_GEOMS = (
    "dynamic_left_pad_geom", "dynamic_right_pad_geom",
    "dynamic_left_lip_geom", "dynamic_right_lip_geom",
)
REPLACED_DYNAMIC_GRIPPER_BODIES = {
    "R_wrist_roll_link", "R_finger1_link", "R_finger2_link",
}
GRASP_TOOL_OFFSET_Z = 0.022
VISUAL_GRASP_Z_MARGIN = 0.006
PREGRASP_HEIGHT = 0.14
RIGHT_DETOUR_OFFSET = np.array([0.22, -0.10, 0.20])
VIEWER_RENDER_DT = 1.0 / 60.0
ROBOT_CAMERA_RENDER_DT = 0.1
MAX_BASE_RECOVERY_SHIFT = 0.12

DYNAMIC_OBSTACLE_XML = """
<body name="dynamic_obstacle" mocap="true" pos="-4.0 -4.0 1.0">
  <geom name="dynamic_obstacle_geom" type="box" size="0.055 0.055 0.10"
        rgba="0.92 0.18 0.12 1" contype="8" conaffinity="5"/>
</body>
"""


def _geom_id(model, name):
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    if geom_id < 0:
        raise ValueError(f"MuJoCo geom not found: {name}")
    return geom_id


def _contacts_enabled(model, geom_a, geom_b):
    return bool(
        (int(model.geom_contype[geom_a]) & int(model.geom_conaffinity[geom_b]))
        or (int(model.geom_contype[geom_b]) & int(model.geom_conaffinity[geom_a]))
    )


def _require_collision_pairs(model, pairs):
    missing = []
    for geom_a, geom_b in pairs:
        if not _contacts_enabled(model, geom_a, geom_b):
            name_a = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_a)
            name_b = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_b)
            missing.append(f"{name_a}<->{name_b}")
    if missing:
        raise RuntimeError("Physical scene has disabled collision pairs: " + ", ".join(missing))


def build_scene():
    urdf_model = mujoco.MjModel.from_xml_path(str(URDF_FILE))
    robot_geom_count = urdf_model.ngeom
    mujoco.mj_saveLastXML(str(BASE_MJCF), urdf_model)
    text = BASE_MJCF.read_text(encoding="utf-8")
    text = insert_before_close(text, "asset", ASSET_XML)
    text = insert_before_close(
        text,
        "worldbody",
        WORLD_XML + DYNAMIC_OBSTACLE_XML + GRIPPER_WORLDBODY_XML,
    )
    text = text.replace("</mujoco>", GRIPPER_MODEL_XML + "\n</mujoco>", 1)
    SCENE_MJCF.write_text(text, encoding="utf-8")
    model = mujoco.MjModel.from_xml_path(str(SCENE_MJCF))
    data = mujoco.MjData(model)
    scene_state = prepare_robot_scene(model, data, robot_geom_count)
    move_base(model, scene_state, BASE_INITIAL_X, BASE_INITIAL_Y, BASE_INITIAL_YAW)

    cube_geom = _geom_id(model, "blue_cube_geom")
    gripper_geoms = [_geom_id(model, name) for name in DYNAMIC_GRIPPER_GEOMS]
    environment_geoms = []
    for geom_id in range(robot_geom_count, model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if name in DYNAMIC_GRIPPER_GEOMS or geom_id == cube_geom:
            continue
        environment_geoms.append(geom_id)

    right_arm_geoms = []
    planning_arm_geoms = []

    for geom_id in range(model.ngeom):
        model.geom_contype[geom_id] = 0
        model.geom_conaffinity[geom_id] = 0

    for geom_id in gripper_geoms:
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if "lip" in geom_name:
            model.geom_contype[geom_id] = 0
            model.geom_conaffinity[geom_id] = COLLISION_ENVIRONMENT
        else:
            model.geom_contype[geom_id] = COLLISION_GRIPPER
            model.geom_conaffinity[geom_id] = COLLISION_CUBE | COLLISION_ENVIRONMENT
    model.geom_contype[cube_geom] = COLLISION_CUBE
    model.geom_conaffinity[cube_geom] = (
        COLLISION_ROBOT | COLLISION_GRIPPER | COLLISION_ENVIRONMENT
    )
    for geom_id in environment_geoms:
        model.geom_contype[geom_id] = COLLISION_ENVIRONMENT
        model.geom_conaffinity[geom_id] = (
            COLLISION_ROBOT | COLLISION_CUBE | COLLISION_GRIPPER
        )
    for geom_id in range(robot_geom_count):
        body_id = model.geom_bodyid[geom_id]
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if body_name.startswith("R_") and not body_name.startswith("R_finger"):
            planning_arm_geoms.append(geom_id)
        if body_name in REPLACED_DYNAMIC_GRIPPER_BODIES:
            continue
        model.geom_contype[geom_id] = COLLISION_ROBOT
        model.geom_conaffinity[geom_id] = COLLISION_CUBE | COLLISION_ENVIRONMENT
        if body_name.startswith("R_"):
            right_arm_geoms.append(geom_id)

    if not right_arm_geoms:
        raise RuntimeError("Could not identify right-arm geoms for collision checks.")

    _require_collision_pairs(
        model,
        (
            (cube_geom, _geom_id(model, "left_table_top")),
            (cube_geom, _geom_id(model, "floor")),
            (cube_geom, gripper_geoms[0]),
            (gripper_geoms[0], _geom_id(model, "left_table_top")),
            (right_arm_geoms[0], cube_geom),
            (right_arm_geoms[0], _geom_id(model, "left_table_top")),
        ),
    )

    model.opt.timestep = 0.002
    model.opt.iterations = 100
    model.opt.noslip_iterations = 10
    mujoco.mj_forward(model, data)
    return (
        model, data, scene_state, right_arm_geoms, planning_arm_geoms,
        environment_geoms,
    )


def interpolate(current, goal, max_delta, tolerance=0.004):
    current[:] += np.clip(goal - current, -max_delta, max_delta)
    return bool(np.max(np.abs(goal - current)) < tolerance)


def keep_result_visible(viewer, robot_camera=None, data=None):
    if viewer is None:
        return
    print("Task complete. Close the MuJoCo window to exit.")
    while viewer.is_alive and not glfw.window_should_close(viewer.window):
        glfw.make_context_current(viewer.window)
        viewer.render()
        if robot_camera is not None and data is not None:
            robot_camera.render(data)
            glfw.make_context_current(viewer.window)
        glfw.poll_events()
        time.sleep(VIEWER_RENDER_DT)
    if robot_camera is not None:
        robot_camera.close()
    viewer.close()


def close_windows(viewer, robot_camera):
    if robot_camera is not None:
        robot_camera.close()
    if viewer is not None:
        viewer.close()


def plan_cartesian_approach(solver, planner, seed_qpos, start_position,
                            target_position, restart_qpos=None, steps=24):
    """Keep the tool in a checked vertical corridor instead of a joint-space arc."""
    waypoints = []
    current = seed_qpos.copy()
    for alpha in np.linspace(0.0, 1.0, steps + 1)[1:]:
        target = start_position + alpha * (target_position - start_position)
        result = solver.solve(current, target)
        if not result.success:
            candidates = [result]
            base_seeds = [current]
            if restart_qpos is not None:
                base_seeds.append(restart_qpos)
            for base_seed in base_seeds:
                candidates.append(solver.solve(base_seed, target))
                for joint_index in range(len(solver.qpos_adrs)):
                    for delta in (-0.28, 0.28):
                        retry_seed = base_seed.copy()
                        retry_seed[solver.qpos_adrs[joint_index]] += delta
                        candidates.append(solver.solve(retry_seed, target))
            result = min(candidates, key=lambda candidate: candidate.error_norm)
        if not result.success:
            raise RuntimeError(
                f"Cartesian approach IK failed: {result.error_norm:.4f} m"
            )
        start_arm = current[planner.qpos_adrs]
        goal_arm = result.qpos[planner.qpos_adrs]
        if planner.edge_in_collision(
            current, start_arm, goal_arm, allow_cube=True, resolution=0.015
        ):
            raise RuntimeError("Cartesian approach corridor is in collision")
        waypoints.append(result.qpos)
        current = result.qpos
    return waypoints


def place_dynamic_obstacle(
    data,
    mocap_id,
    planner,
    solver,
    current_qpos,
    retreat_qpos,
    remaining_path,
    preferred_fraction,
):
    """Place an obstacle on the future path without occupying safe endpoints."""
    if not remaining_path:
        raise RuntimeError("dynamic obstacle requires a planned path")
    preferred_index = min(
        len(remaining_path) - 1,
        max(0, int(preferred_fraction * len(remaining_path))),
    )
    candidate_indices = sorted(
        range(1, max(1, len(remaining_path) - 1)),
        key=lambda index: abs(index - preferred_index),
    )
    terminal_qpos = remaining_path[-1]
    for path_index in candidate_indices:
        obstacle_position = solver.tool_position_for_qpos(
            remaining_path[path_index]
        )
        data.mocap_pos[mocap_id] = obstacle_position
        planner.sync_mocap_state(data)
        safe_endpoints = (current_qpos, retreat_qpos, terminal_qpos)
        if any(
            planner.configuration_in_collision(configuration)
            for configuration in safe_endpoints
        ):
            continue
        blocked_waypoints = sum(
            planner.configuration_in_collision(waypoint)
            for waypoint in remaining_path
        )
        if blocked_waypoints:
            mujoco.mj_forward(planner.model, data)
            return obstacle_position, blocked_waypoints
    raise RuntimeError(
        "could not place dynamic obstacle without occupying a safe endpoint"
    )


def _run_episode(headless=False, timeout=45.0, fault="none", log_path=None,
                 fault_severity=1.0, seed=7, policy="active_case",
                 scene_jitter=0.0):
    policy = ExperimentPolicy(policy)
    (
        model, data, scene_state, right_arm_geoms, planning_arm_geoms,
        environment_geoms,
    ) = build_scene()
    robot_qpos = data.qpos.copy()
    initial_robot_qpos = robot_qpos.copy()
    solver = DampedLeastSquaresIK(model)
    gripper = DynamicGripper(model, data, solver)
    monitor = OperationMonitor(log_path)
    supervisor = ManipulationSupervisor()
    active_probe = ActiveProbeRuntime()
    recovery_verifier = RecoveryVerifier()
    fault_injector = FaultInjector(FaultInjectionConfig(
        scenario=FaultScenario(fault), severity=fault_severity, seed=seed,
    ))
    sensor_suite = MujocoSensorSuite(data, solver.qpos_adrs, solver.dof_adrs)
    planner = ArmMotionPlanner(
        model, solver.joint_ids, planning_arm_geoms, environment_geoms,
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "blue_cube_geom"),
        execution_arm_geom_ids=right_arm_geoms,
        tool_body_id=solver.body_id,
        tool_offset_local=solver.tool_offset_local,
        tool_radius=0.047,
        tool_half_extent=(0.042, 0.100, 0.050),
    )
    cube_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "blue_cube")
    cube_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "blue_cube_geom")
    cube_joint = model.body_jntadr[cube_body]
    dynamic_obstacle_body = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "dynamic_obstacle"
    )
    dynamic_obstacle_mocap = int(model.body_mocapid[dynamic_obstacle_body])
    if dynamic_obstacle_mocap < 0:
        raise RuntimeError("dynamic obstacle must be a MuJoCo mocap body")

    dynamic_qpos = set(gripper.dynamic_qpos_addresses.tolist())
    cube_qadr = model.jnt_qposadr[cube_joint]
    if scene_jitter > 0.0:
        scene_rng = np.random.default_rng(seed + 7919)
        data.qpos[cube_qadr:cube_qadr + 2] += scene_rng.uniform(
            -scene_jitter, scene_jitter, size=2
        )
        mujoco.mj_forward(model, data)
    dynamic_qpos.update(range(cube_qadr, cube_qadr + 7))
    kinematic_qpos = np.array([index for index in range(model.nq) if index not in dynamic_qpos])
    dynamic_dofs = set(gripper.dynamic_dof_addresses.tolist())
    cube_dadr = model.jnt_dofadr[cube_joint]
    dynamic_dofs.update(range(cube_dadr, cube_dadr + 6))
    kinematic_dofs = np.array([index for index in range(model.nv) if index not in dynamic_dofs])

    viewer = None if headless else mujoco_viewer.MujocoViewer(model, data)
    if viewer:
        viewer.cam.distance = 3.2
        viewer.cam.azimuth = 25
        viewer.cam.elevation = -18
    robot_camera = HeadCameraWindow(
        model,
        viewer.window if viewer is not None else None,
        detector=BlueCubeRgbdDetector(),
        display=not headless,
    )
    next_render_time = 0.0
    next_robot_camera_time = 0.0

    latest_visual_detection = robot_camera.render(data)
    if viewer is not None:
        glfw.make_context_current(viewer.window)
    if latest_visual_detection is None:
        close_windows(viewer, robot_camera)
        raise RuntimeError("RGB-D detector could not locate the initial target")
    perceived_object_position = np.asarray(
        latest_visual_detection.position, dtype=float
    )
    print(
        "VISION: initial 6D pose="
        f"{np.asarray(latest_visual_detection.pose6d).round(4).tolist()} "
        f"confidence={latest_visual_detection.confidence:.3f}"
    )

    initial_cube_z = float(data.xpos[cube_body, 2])
    initial_object_xy = data.xpos[cube_body, :2].copy()
    base_recovery_offset = np.zeros(2, dtype=float)
    stage = "clearance"
    stage_start = float(data.time)
    run_start = stage_start
    wall_start = time.perf_counter()
    goal = None
    planned_waypoints = []
    lift_target_z = None
    grasp_xy = None
    hold_start = None
    contact_frames = 0
    last_stage = None
    last_monitor_time = run_start
    last_diagnostic_time = wall_start
    recovery_count = 0
    recovery_directive = None
    last_recovery_plan_id = None
    planned_object_position = None
    sensor_mask_until = -1.0
    sensor_invalid_until = -1.0
    collision_report_until = -1.0
    ik_failure_until = -1.0
    planning_failure_until = -1.0
    injection_event = None
    last_recovery_fault = None
    latest_observation = None
    dynamic_obstacle_position = None

    print("Dynamic-gripper pickup started; no cube attachment or pose overwrite.")
    while float(data.time) - run_start < timeout:
        now = float(data.time)
        wall_now = time.perf_counter()
        cube_position = data.xpos[cube_body].copy()

        directive = fault_injector.poll(
            stage,
            float(cube_position[2] - initial_cube_z),
            float(now - stage_start),
        )
        if directive is not None:
            injection_event = directive.scenario.value
            if directive.action == "release_grasp":
                gripper.open()
                gripper.enable_cube_contacts(cube_geom, False)
                data.qvel[cube_dadr] = directive.parameters["velocity_x"]
                data.qvel[cube_dadr + 1] = directive.parameters["velocity_y"]
                data.qvel[cube_dadr + 3:cube_dadr + 6] = (
                    directive.parameters["angular_velocity"]
                    * np.array([0.55, -0.35, 1.0])
                )
            elif directive.action == "disable_gripper_contact":
                gripper.enable_cube_contacts(cube_geom, False)
            elif directive.action == "shift_object":
                data.qpos[cube_qadr] += directive.parameters["dx"]
                data.qpos[cube_qadr + 1] += directive.parameters["dy"]
                mujoco.mj_forward(model, data)
                cube_position = data.xpos[cube_body].copy()
            elif directive.action == "mask_contact_sensor":
                sensor_mask_until = now + float(directive.parameters["duration"])
            elif directive.action == "invalidate_sensors":
                sensor_invalid_until = now + float(directive.parameters["duration"])
            elif directive.action == "move_dynamic_obstacle":
                fraction = float(directive.parameters["path_fraction"])
                remaining_path = (
                    ([] if goal is None else [goal]) + list(planned_waypoints)
                )
                obstacle_position, blocked_waypoints = place_dynamic_obstacle(
                    data,
                    dynamic_obstacle_mocap,
                    planner,
                    solver,
                    robot_qpos,
                    initial_robot_qpos,
                    remaining_path,
                    fraction,
                )
                dynamic_obstacle_position = np.asarray(
                    obstacle_position, dtype=float
                ).copy()
                collision_report_until = (
                    now + float(directive.parameters["duration"])
                )
                monitor.record_event(
                    "dynamic_obstacle_moved",
                    position=obstacle_position,
                    path_fraction=fraction,
                    blocked_waypoints=blocked_waypoints,
                    simulation_time=now,
                )
            elif directive.action == "report_ik_failure":
                ik_failure_until = now + float(directive.parameters["duration"])
            elif directive.action == "report_planning_failure":
                planning_failure_until = now + float(directive.parameters["duration"])
            print(
                f"FAULT INJECTED: {directive.scenario.value} "
                f"({directive.action}), stage={stage}, "
                f"height_gain={cube_position[2] - initial_cube_z:.4f} m"
            )
            monitor.record_event(
                "fault_injected",
                scenario=directive.scenario.value,
                action=directive.action,
                parameters=directive.parameters,
                simulation_time=now,
            )
            if policy == ExperimentPolicy.FIXED_RULE:
                recovery_directive = GraspRecoveryDirective(
                    wait_for_grounded_object=directive.action == "release_grasp"
                )
                stage = (
                    "recovery_wait"
                    if recovery_directive.wait_for_grounded_object
                    else "recovery_reset"
                )
                stage_start = now
                goal = None
                planned_waypoints = []

        if stage == "clearance" and goal is None:
            if (
                recovery_count > 0
                and last_recovery_fault == FaultType.COLLISION
                and dynamic_obstacle_position is not None
            ):
                clearance_position = solver.tool_position(data).copy()
                retreat_direction = (
                    clearance_position[:2] - dynamic_obstacle_position[:2]
                )
                retreat_norm = float(np.linalg.norm(retreat_direction))
                if retreat_norm < 1e-6:
                    retreat_direction = np.array([0.0, -1.0])
                else:
                    retreat_direction /= retreat_norm
                clearance_position[:2] += 0.14 * retreat_direction
                clearance_position[2] = max(
                    clearance_position[2] + 0.08, 0.90
                )
                result = solver.solve(robot_qpos, clearance_position)
                if not result.success:
                    raise RuntimeError(
                        "Collision-retreat IK failed: "
                        f"{result.error_norm:.4f} m"
                    )
                clearance_goal = result.qpos
            else:
                clearance_position = solver.tool_position(data).copy()
                clearance_position[2] = max(clearance_position[2], 0.85)
                result = solver.solve(robot_qpos, clearance_position)
                if not result.success:
                    raise RuntimeError(
                        f"Clearance IK failed: {result.error_norm:.4f} m"
                    )
                clearance_goal = result.qpos
            # Recovery can begin while the arm is still touching the object.
            # The clearance motion is allowed to break that contact while all
            # environment collision constraints remain active.
            plan = planner.plan(
                robot_qpos,
                clearance_goal,
                allow_cube=recovery_count > 0,
                allow_start_in_collision=recovery_count > 0,
            )
            if not plan.success:
                raise RuntimeError(f"Clearance planning failed: {plan.method}")
            planned_waypoints = plan.waypoints
            goal = planned_waypoints.pop(0)
            print(f"Clearance planner: {plan.method}, waypoints={len(plan.waypoints)}")
        elif stage == "right_detour" and goal is None:
            detour_offsets = [RIGHT_DETOUR_OFFSET]
            if recovery_count > 0 and last_recovery_fault == FaultType.COLLISION:
                detour_offsets.extend((
                    np.array([0.28, -0.18, 0.28]),
                    np.array([0.32, -0.04, 0.30]),
                    np.array([0.16, -0.22, 0.30]),
                ))
            failures = []
            plan = None
            selected_offset = None
            for detour_offset in detour_offsets:
                detour_position = perceived_object_position + detour_offset
                result = solver.solve(robot_qpos, detour_position)
                if not result.success:
                    failures.append(f"ik:{result.error_norm:.4f}")
                    continue
                candidate_plan = planner.plan(
                    robot_qpos, result.qpos, allow_cube=False
                )
                if not candidate_plan.success:
                    failures.append(candidate_plan.method)
                    continue
                plan = candidate_plan
                selected_offset = detour_offset
                break
            if plan is None:
                raise RuntimeError(
                    "Right-detour planning failed: " + ", ".join(failures)
                )
            planned_waypoints = plan.waypoints
            goal = planned_waypoints.pop(0)
            print(
                f"Right-side planner: {plan.method}, "
                f"waypoints={len(plan.waypoints)}, "
                f"offset={np.asarray(selected_offset).round(3).tolist()}"
            )
        elif stage == "pregrasp" and goal is None:
            visual_grasp_position = perceived_object_position.copy()
            visual_grasp_position[2] += VISUAL_GRASP_Z_MARGIN
            result = solver.solve(
                robot_qpos,
                visual_grasp_position + [0.0, 0.0, PREGRASP_HEIGHT],
            )
            if not result.success:
                raise RuntimeError(f"Pregrasp IK failed: {result.error_norm:.4f} m")
            plan = planner.plan(robot_qpos, result.qpos, allow_cube=False)
            if not plan.success:
                raise RuntimeError(f"Pregrasp planning failed: {plan.method}")
            planned_waypoints = plan.waypoints
            goal = planned_waypoints.pop(0)
            print(f"Pregrasp planner: {plan.method}, waypoints={len(plan.waypoints)}")
        elif stage == "approach" and goal is None:
            planned_object_position = perceived_object_position.copy()
            start_position = solver.tool_position(data).copy()
            visual_grasp_position = perceived_object_position.copy()
            visual_grasp_position[2] += VISUAL_GRASP_Z_MARGIN
            target_position = (
                visual_grasp_position + [0.0, 0.0, GRASP_TOOL_OFFSET_Z]
            )
            planned_waypoints = plan_cartesian_approach(
                solver, planner, robot_qpos, start_position, target_position,
                restart_qpos=initial_robot_qpos,
            )
            goal = planned_waypoints.pop(0)
            print(
                "Approach planner: collision-checked Cartesian corridor, "
                f"waypoints={len(planned_waypoints) + 1}"
            )

        if stage in ("clearance", "right_detour", "pregrasp", "approach"):
            if interpolate(robot_qpos, goal, 0.0018):
                if planned_waypoints:
                    goal = planned_waypoints.pop(0)
                else:
                    if stage == "clearance":
                        stage = "right_detour"
                    elif stage == "right_detour":
                        stage = "pregrasp"
                    elif stage == "pregrasp":
                        stage = "approach"
                    else:
                        stage = "close"
                    stage_start = now
                    goal = None
        elif stage == "close":
            visual_grasp_position = (
                planned_object_position
                if planned_object_position is not None
                else perceived_object_position
            ).copy()
            visual_grasp_position[2] += VISUAL_GRASP_Z_MARGIN
            tracking = solver.solve(
                robot_qpos,
                visual_grasp_position + [0.0, 0.0, GRASP_TOOL_OFFSET_Z],
            )
            if tracking.success:
                interpolate(robot_qpos, tracking.qpos, 0.0006)
            closure = min((now - stage_start) / 2.5, 1.0) * 0.017
            gripper.close(closure)
            contact_frames = contact_frames + 1 if gripper.both_pads_touch(cube_geom) else 0
            if contact_frames >= 250:
                stage = "lift"
                grasp_xy = solver.tool_position(data)[:2].copy()
                lift_target_z = solver.tool_position(data)[2]
                stage_start = now
        elif stage == "lift":
            gripper.close()
            if active_probe.name != "pause_and_hold":
                lift_target_z = min(
                    lift_target_z + LIFT_TARGET_STEP,
                    initial_cube_z + 0.14,
                )
            result = solver.solve(robot_qpos, np.r_[grasp_xy, lift_target_z])
            if result.success:
                interpolate(robot_qpos, result.qpos, 0.00038)
            if lift_target_z >= initial_cube_z + 0.14:
                stage = "hold"
                hold_start = now
        elif stage == "recovery_wait":
            gripper.open()
            if (
                latest_visual_detection is not None
                and perceived_object_position[2] <= initial_cube_z + 0.035
                and abs(cube_position[2] - perceived_object_position[2]) <= 0.035
            ):
                stage = "recovery_reset"
                stage_start = now
        elif stage == "recovery_reset":
            gripper.open()
            gripper.enable_cube_contacts(cube_geom, True)
            if now - stage_start >= 0.3:
                recovery_count += 1
                requested_offset = (
                    perceived_object_position[:2] - initial_object_xy
                )
                offset_norm = float(np.linalg.norm(requested_offset))
                if offset_norm > MAX_BASE_RECOVERY_SHIFT:
                    requested_offset *= MAX_BASE_RECOVERY_SHIFT / offset_norm
                if np.linalg.norm(requested_offset - base_recovery_offset) > 0.005:
                    base_recovery_offset = requested_offset.copy()
                    move_base(
                        model,
                        scene_state,
                        BASE_INITIAL_X + base_recovery_offset[0],
                        BASE_INITIAL_Y + base_recovery_offset[1],
                        BASE_INITIAL_YAW,
                    )
                    mujoco.mj_forward(model, data)
                    monitor.record_event(
                        "base_reposition",
                        offset=base_recovery_offset,
                        simulation_time=now,
                    )
                    print(
                        "RECOVERY: base reposition offset="
                        f"{base_recovery_offset.round(4).tolist()}"
                    )
                goal = None
                planned_waypoints = []
                contact_frames = 0
                lift_target_z = None
                grasp_xy = None
                hold_start = None
                planned_object_position = None
                recovery_directive = None
                stage = "clearance"
                stage_start = now
                supervisor.mark_recovery_complete(successful=False)
                print(
                    f"RECOVERY: restarting collision-aware grasp attempt "
                    f"{recovery_count}, visual_position="
                    f"{perceived_object_position.round(4).tolist()}"
                )
                monitor.record_event(
                    "recovery_replan", attempt=recovery_count,
                    simulation_time=now,
                )
        elif stage == "hold":
            gripper.close()
            if now - hold_start >= 3.0:
                gain = float(data.xpos[cube_body, 2] - initial_cube_z)
                success = gain >= 0.08
                verification = None
                if last_recovery_fault is not None and latest_observation is not None:
                    verification = recovery_verifier.verify(
                        last_recovery_fault, latest_observation, initial_cube_z
                    )
                    success = success and verification.successful
                print(f"RESULT: {'SUCCESS' if success else 'FAIL'}")
                print(f"cube_height_gain={gain:.4f} m, hold_time=3.0 s")
                monitor.record_summary(
                    success=success,
                    terminal_stage=stage,
                    fault=fault,
                    recovery_count=recovery_count,
                    cube_height_gain=gain,
                    simulation_duration=now - run_start,
                    policy=policy.value,
                    seed=seed,
                    fault_severity=fault_severity,
                    scene_jitter=scene_jitter,
                    recovery_verified=(
                        verification.successful if verification is not None else None
                    ),
                    recovery_verification_reason=(
                        verification.reason if verification is not None else None
                    ),
                )
                monitor.close()
                if success:
                    keep_result_visible(viewer, robot_camera, data)
                else:
                    close_windows(viewer, robot_camera)
                return success

        data.qpos[kinematic_qpos] = robot_qpos[kinematic_qpos]
        data.qvel[kinematic_dofs] = 0.0
        mujoco.mj_forward(model, data)
        gripper.follow_wrist()
        mujoco.mj_step(model, data)

        # Once the gripper has physically engaged the object, subsequent object
        # motion belongs to the interaction rather than an external target move.
        # Keep the pre-contact target reference only while it is diagnostically
        # meaningful, including short intervals where contact is later lost.
        if (
            planned_object_position is not None
            and stage in {"close", "lift", "hold"}
            and any(gripper.contact_flags(cube_geom))
        ):
            planned_object_position = None

        penetration = gripper.environment_penetration(environment_geoms)
        if penetration < -0.0005:
            print(f"SAFETY STOP: gripper penetration {penetration:.6f} m")
            monitor.record(
                stage, data.xpos[cube_body], solver.tool_position(data),
                gripper.both_pads_touch(cube_geom), event="gripper_penetration",
                extra={"penetration": penetration},
            )
            monitor.close()
            close_windows(viewer, robot_camera)
            return False

        forbidden_contact = planner.current_arm_environment_collision(data)
        if forbidden_contact is not None:
            arm_geom, obstacle_geom = forbidden_contact
            arm_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, arm_geom)
            obstacle_name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, obstacle_geom
            )
            print(f"SAFETY STOP: {arm_name} collided with {obstacle_name}")
            monitor.close()
            close_windows(viewer, robot_camera)
            return False

        if stage != last_stage:
            print(
                f"stage={stage}, cube_z={data.xpos[cube_body, 2]:.3f}, "
                f"both_contacts={gripper.both_pads_touch(cube_geom)}"
            )
            last_stage = stage
        if now - last_monitor_time >= 0.1:
            diagnostic = gripper.contact_diagnostics(cube_geom)
            contact_flags = gripper.contact_flags(cube_geom)
            if now < sensor_mask_until:
                contact_flags = (False, False)
            sensor_packet = sensor_suite.sample(
                object_position=data.xpos[cube_body],
                tool_position=solver.tool_position(data),
                gripper_contacts=contact_flags,
                wrist_force=(0.0, 0.0, diagnostic["vertical_force"]),
            )
            observation = build_grasp_observation(
                timestamp=now,
                stage=stage,
                object_position=data.xpos[cube_body],
                tool_position=solver.tool_position(data),
                contact_flags=contact_flags,
                contact_diagnostics=diagnostic,
                object_vertical_velocity=data.qvel[cube_dadr + 2],
                planned_object_position=planned_object_position,
                collision=now < collision_report_until,
                sensor_valid=now >= sensor_invalid_until,
                ik_success=now >= ik_failure_until,
                planning_success=now >= planning_failure_until,
                event=injection_event,
                sensor_packet=sensor_packet,
                visual_detection=latest_visual_detection,
            )
            latest_observation = observation
            monitor.record_observation(observation)
            if (
                not stage.startswith("recovery_")
                and policy.diagnosis_enabled
            ):
                decision = supervisor.observe(observation)
                if decision.report.anomalous:
                    monitor.record_diagnosis(decision.report)
                if decision.probe and active_probe.start(
                    decision.probe, observation, decision.report
                ):
                    monitor.record_event(
                        "active_probe_started",
                        probe=decision.probe,
                        suspected_fault=decision.confirmed_fault.value,
                        simulation_time=now,
                    )
                probe_outcome = active_probe.update(
                    observation,
                    physical_contact_flags=gripper.contact_flags(cube_geom),
                )
                if probe_outcome is not None:
                    monitor.record_event(
                        "active_probe_completed",
                        probe=probe_outcome.name,
                        positive=probe_outcome.positive,
                        rationale=probe_outcome.rationale,
                        simulation_time=now,
                    )
                    decision = supervisor.complete_probe(
                        probe_outcome.report,
                        stage,
                        probe_outcome.name,
                        probe_outcome.positive,
                    )
                if (
                    decision.recovery_plan is not None
                    and policy.learned_recovery_enabled
                    and decision.recovery_plan.plan_id != last_recovery_plan_id
                ):
                    active_probe.reset()
                    last_recovery_plan_id = decision.recovery_plan.plan_id
                    last_recovery_fault = decision.recovery_plan.fault
                    recovery_directive = compile_grasp_recovery(
                        decision.recovery_plan
                    )
                    monitor.record_recovery_plan(decision.recovery_plan)
                    print(
                        "DIAGNOSIS: "
                        f"fault={decision.confirmed_fault.value} "
                        f"confidence={decision.confidence:.3f} "
                        f"plan={decision.recovery_plan.source}"
                    )
                    collision_report_until = -1.0
                    ik_failure_until = -1.0
                    planning_failure_until = -1.0
                    sensor_invalid_until = -1.0
                    gripper.open()
                    if recovery_directive.wait_for_grounded_object:
                        gripper.enable_cube_contacts(cube_geom, False)
                        stage = "recovery_wait"
                    else:
                        gripper.enable_cube_contacts(cube_geom, True)
                        stage = "recovery_reset"
                    stage_start = now
                    goal = None
                    planned_waypoints = []
            injection_event = None
            last_monitor_time = now
        if stage in ("close", "lift") and wall_now - last_diagnostic_time >= 0.5:
            diagnostic = gripper.contact_diagnostics(cube_geom)
            print(
                "contact_diag "
                f"stage={stage} count={diagnostic['contacts']} "
                f"normal={diagnostic['normal_force']:.2f}N "
                f"tangent={diagnostic['tangent_force']:.2f}N "
                f"vertical={diagnostic['vertical_force']:.2f}N "
                f"distance={diagnostic['minimum_distance']:.6f}m "
                f"cube_z={data.xpos[cube_body, 2]:.4f}m "
                f"cube_vz={data.qvel[cube_dadr + 2]:.4f}m/s "
                f"cube_fz={data.qfrc_constraint[cube_dadr + 2]:.2f}N "
                f"cube_az={data.qacc[cube_dadr + 2]:.3f}m/s2 "
                f"qpos={diagnostic['finger_qpos'].round(4).tolist()} "
                f"force={diagnostic['actuator_force'].round(2).tolist()}"
            )
            last_diagnostic_time = wall_now
        if robot_camera is not None and data.time >= next_robot_camera_time:
            detection = robot_camera.render(data)
            if detection is not None:
                latest_visual_detection = detection
                if stage in {
                    "clearance", "right_detour", "pregrasp",
                    "recovery_wait", "recovery_reset",
                }:
                    perceived_object_position = np.asarray(
                        detection.position, dtype=float
                    )
            if viewer is not None:
                glfw.make_context_current(viewer.window)
            next_robot_camera_time = data.time + ROBOT_CAMERA_RENDER_DT
        if viewer:
            if data.time >= next_render_time:
                glfw.make_context_current(viewer.window)
                viewer.render()
                next_render_time = data.time + VIEWER_RENDER_DT
            glfw.poll_events()
            if not viewer.is_alive or glfw.window_should_close(viewer.window):
                break

    monitor.record_summary(
        success=False,
        terminal_stage=stage,
        fault=fault,
        recovery_count=recovery_count,
        cube_height_gain=float(data.xpos[cube_body, 2] - initial_cube_z),
        simulation_duration=float(data.time) - run_start,
        policy=policy.value,
        seed=seed,
        fault_severity=fault_severity,
        scene_jitter=scene_jitter,
        failure_reason="timeout",
    )
    monitor.close()
    close_windows(viewer, robot_camera)
    print(f"RESULT: TIMEOUT at stage={stage}")
    return False


def _read_failure_context(log_path):
    context = {
        "terminal_stage": "exception",
        "recovery_count": 0,
        "cube_height_gain": 0.0,
        "simulation_duration": 0.0,
    }
    if not log_path or not Path(log_path).exists():
        return context
    initial_height = None
    try:
        for line in Path(log_path).read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            payload = record.get("payload", {})
            if record.get("record_type") == "observation":
                position = payload.get("object_position")
                if position and len(position) >= 3:
                    if initial_height is None:
                        initial_height = float(position[2])
                    context["cube_height_gain"] = float(position[2]) - initial_height
                context["simulation_duration"] = float(
                    payload.get("timestamp", context["simulation_duration"])
                )
                context["last_active_stage"] = str(
                    payload.get("stage", "unknown")
                )
            elif (
                record.get("record_type") == "event"
                and payload.get("name") == "recovery_replan"
            ):
                context["recovery_count"] = max(
                    context["recovery_count"], int(payload.get("attempt", 0))
                )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        pass
    return context


def run(headless=False, timeout=45.0, fault="none", log_path=None,
        fault_severity=1.0, seed=7, policy="active_case", scene_jitter=0.0):
    """Run one episode and guarantee a terminal summary for batch analysis."""
    try:
        return _run_episode(
            headless, timeout, fault, log_path, fault_severity, seed, policy,
            scene_jitter,
        )
    except Exception as error:
        failure_context = _read_failure_context(log_path)
        failure_monitor = OperationMonitor(log_path, append=True)
        failure_monitor.record_summary(
            success=False,
            fault=fault,
            policy=str(policy),
            seed=seed,
            fault_severity=fault_severity,
            scene_jitter=scene_jitter,
            failure_reason="exception",
            error_type=type(error).__name__,
            error_message=str(error),
            **failure_context,
        )
        failure_monitor.close()
        print(f"RESULT: ERROR {type(error).__name__}: {error}")
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument(
        "--fault", choices=tuple(item.value for item in FaultScenario),
        default=FaultScenario.SLIP.value,
    )
    parser.add_argument("--fault-severity", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--scene-jitter", type=float, default=0.0)
    parser.add_argument(
        "--policy",
        choices=tuple(item.value for item in ExperimentPolicy),
        default=ExperimentPolicy.ACTIVE_CASE.value,
    )
    parser.add_argument("--log", default=None, help="optional JSONL state log")
    args = parser.parse_args()
    raise SystemExit(0 if run(
        args.headless, args.timeout, args.fault, args.log,
        args.fault_severity, args.seed, args.policy, args.scene_jitter,
    ) else 1)


if __name__ == "__main__":
    main()
