#!/usr/bin/env python3
"""Task-graph-driven physical dual-arm transport of a long object."""

import argparse
from pathlib import Path
import time

import glfw
import mujoco
import mujoco_viewer
import numpy as np

from cruzr_sim.control import (
    DampedLeastSquaresIK,
    DynamicGripper,
    matrix_to_rpy as _matrix_to_rpy,
    rpy_to_matrix as _rpy_to_matrix,
    step_dual_toward as _step_dual_toward,
    step_toward as _step_toward,
)
from cruzr_sim.diagnosis import DualArmObservation
from cruzr_sim.diagnosis.operation_monitor import OperationMonitor
from cruzr_sim.faults import DualArmFaultInjector, DualArmFaultScenario
from cruzr_sim.experiments import ExperimentPolicy
from cruzr_sim.perception import (
    BlueCubeRgbdDetector,
    HeadCameraWindow,
    InstanceSegmentationRgbdDetector,
)
from cruzr_sim.planning import (
    ArmCapability,
    ArmMotionPlanner,
    PlanRiskProfile,
    RecoveryAwareEvaluator,
    assign_primary_and_support,
    plan_dual_goal as _plan_dual_goal,
    plan_contact_anchored_regrasp as _plan_contact_anchored_regrasp,
    tracking_error as _tracking_error,
    trajectory_phase as _trajectory_phase,
)
from cruzr_sim.simulation import (
    LEFT_GRIPPER_PREFIX,
    RIGHT_GRIPPER_PREFIX,
    build_dual_arm_scene as build_scene,
    mujoco_id as _id,
    place_dynamic_obstacle as _place_dynamic_obstacle,
    MuJoCoDualArmBackend,
)
from cruzr_sim.scenes.domain_randomization import sample_workpiece
from cruzr_sim.recovery import (
    ContextualRecoveryExperienceGraph,
    DualArmRecoveryPlanner,
)
from cruzr_sim.recovery.llm_adapter import (
    ConstrainedDualArmCandidateGenerator,
    QwenOpenAICompatibleClient,
    ReplayCandidateClient,
)
from cruzr_sim.tasks.belief_update import (
    BayesianBeliefUpdater,
    binary_evidence,
    confidence_evidence,
)
from cruzr_sim.tasks.cooperative_task_graph import build_cooperative_transport_graph
from cruzr_sim.tasks.dual_arm_supervisor import DualArmSupervisor
from cruzr_sim.tasks.diagnostic_probes import (
    evaluate_probe_outcome,
    probe_specification,
)


OBJECT_HALF_LENGTH = 0.18
FINGER_HALF_LENGTH_X = 0.04
GRASP_OFFSET_X = 0.13
PREGRASP_HEIGHT = 0.15
TOOL_OBJECT_OFFSET_Z = 0.022
LIFT_HEIGHT = 0.12
TRANSPORT_OFFSET_Y = -0.12
RELEASE_RETREAT_X = 0.08
RELEASE_RETREAT_Z = 0.06
MAX_JOINT_DELTA = 0.0012
VIEWER_RENDER_DT = 1.0 / 60.0
CAMERA_RENDER_DT = 0.1
OBSERVATION_DT = 0.1
LOCAL_REGRASP_TIMEOUT = 1.5
FALLBACK_RECOVERY_TIMEOUT = 5.0
FALLBACK_SETTLE_TIME = 0.4
OBSTACLE_RETREAT_DISTANCES = (0.14, 0.18, 0.22)
OBSTACLE_MAX_VERTICAL_RETREAT = 0.04


def _set_grippers(backend, left_closed, right_closed):
    result = backend.set_grippers(left_closed, right_closed)
    if not result.accepted:
        raise RuntimeError(result.reason)

def run(
    headless=False,
    timeout=90.0,
    fault="none",
    fault_severity=1.0,
    seed=7,
    policy="full",
    log_path=None,
    scene_jitter=0.0,
    experience_path=None,
    workpiece_domain=None,
    detector_kind="color",
    detector_weights=None,
    experience_mode="online",
    experience_ablation="full",
    diagnosis_ablation="full",
    qwen_model="qwen-plus",
    qwen_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    qwen_timeout=12.0,
    llm_replay_response=None,
):
    if diagnosis_ablation not in {"full", "no_active_probe", "no_temporal"}:
        raise ValueError(f"unknown diagnosis ablation: {diagnosis_ablation}")
    policy = ExperimentPolicy(policy)
    active_search_faults = {
        "transport_slip", "vision_occlusion", "vision_dropout",
        "sensor_dropout", "target_pose_shift", "dynamic_obstacle",
    }
    recovery_enabled = policy not in {
        ExperimentPolicy.NO_RECOVERY,
        ExperimentPolicy.B0_FIXED_FSM,
    } and not (
        policy == ExperimentPolicy.B1_TASK_GRAPH
        and fault in active_search_faults
    )
    monitor = OperationMonitor(log_path)
    monitor.record_event(
        "episode_started",
        simulation_time=0.0,
        fault=fault,
        fault_severity=float(fault_severity),
        seed=int(seed),
        policy=policy.value,
        scene_jitter=float(scene_jitter),
        workpiece_domain=workpiece_domain,
        llm_enabled=policy == ExperimentPolicy.OURS_LLM,
        llm_model=(qwen_model if policy == ExperimentPolicy.OURS_LLM else None),
        diagnosis_ablation=diagnosis_ablation,
    )
    workpiece = (
        sample_workpiece(workpiece_domain, seed)
        if workpiece_domain is not None else None
    )
    (
        model, data, object_geom,
        left_arm_geoms, right_arm_geoms, environment_geoms,
    ) = build_scene(workpiece)
    object_half_length = (
        workpiece.half_length if workpiece is not None else OBJECT_HALF_LENGTH
    )
    tool_object_offset_z = (
        0.55 * workpiece.half_size[2]
        if workpiece is not None else TOOL_OBJECT_OFFSET_Z
    )
    object_body = _id(model, mujoco.mjtObj.mjOBJ_BODY, "blue_cube")
    obstacle_body = _id(model, mujoco.mjtObj.mjOBJ_BODY, "dynamic_obstacle")
    obstacle_mocap = int(model.body_mocapid[obstacle_body])
    if obstacle_mocap < 0:
        raise RuntimeError("dynamic obstacle must be mocap controlled")
    object_joint = model.body_jntadr[object_body]
    object_qadr = model.jnt_qposadr[object_joint]
    object_dadr = model.jnt_dofadr[object_joint]
    if scene_jitter > 0.0:
        rng = np.random.default_rng(int(seed))
        data.qpos[object_qadr:object_qadr + 2] += rng.uniform(
            -float(scene_jitter), float(scene_jitter), size=2
        )
        mujoco.mj_forward(model, data)
    initial_object_position = data.xpos[object_body].copy()

    left_solver = DampedLeastSquaresIK(model, side="L")
    right_solver = DampedLeastSquaresIK(model, side="R")
    left_gripper = DynamicGripper(
        model, data, left_solver, prefix=LEFT_GRIPPER_PREFIX
    )
    right_gripper = DynamicGripper(
        model, data, right_solver, prefix=RIGHT_GRIPPER_PREFIX
    )
    left_planner = ArmMotionPlanner(
        model, left_solver.joint_ids, left_arm_geoms,
        list(environment_geoms) + list(right_arm_geoms), object_geom,
        execution_arm_geom_ids=left_arm_geoms,
        tool_body_id=left_solver.body_id,
        tool_offset_local=left_solver.tool_offset_local,
        tool_radius=0.047,
        arm_radius_cap=0.075,
    )
    right_planner = ArmMotionPlanner(
        model, right_solver.joint_ids, right_arm_geoms,
        list(environment_geoms) + list(left_arm_geoms), object_geom,
        execution_arm_geom_ids=right_arm_geoms,
        tool_body_id=right_solver.body_id,
        tool_offset_local=right_solver.tool_offset_local,
        tool_radius=0.047,
        arm_radius_cap=0.075,
    )

    graph = build_cooperative_transport_graph()
    fault_injector = DualArmFaultInjector(fault, fault_severity, seed)
    experience_path = Path(experience_path) if experience_path else None
    if experience_mode == "frozen" and (
        experience_path is None or not experience_path.exists()
    ):
        raise FileNotFoundError("frozen experience graph does not exist")
    experience_graph = (
        ContextualRecoveryExperienceGraph.load(
            experience_path, selection_mode=experience_ablation
        )
        if (
            policy.learned_recovery_enabled
            and experience_path is not None
            and experience_path.exists()
        )
        else ContextualRecoveryExperienceGraph(
            selection_mode=experience_ablation
        )
    )
    use_experience = (
        policy.learned_recovery_enabled and experience_mode != "off"
    )
    update_experience = use_experience and experience_mode in {"online", "train"}
    llm_client = None
    if policy == ExperimentPolicy.OURS_LLM:
        llm_client = (
            ReplayCandidateClient(llm_replay_response)
            if llm_replay_response is not None
            else QwenOpenAICompatibleClient(
                model=qwen_model, base_url=qwen_base_url, timeout=qwen_timeout,
            )
        )
    candidate_generator = (
        ConstrainedDualArmCandidateGenerator(llm_client)
        if llm_client is not None else None
    )
    recovery_planner = DualArmRecoveryPlanner(
        experience_graph, candidate_generator=candidate_generator
    )
    supervisor = DualArmSupervisor(
        recovery_planner=recovery_planner,
        use_temporal_belief=(
            policy.belief_enabled and diagnosis_ablation != "no_temporal"
        ),
        use_experience_graph=use_experience,
        update_experience_graph=update_experience,
    )
    robot_qpos = data.qpos.copy()
    dynamic_qpos = set(range(object_qadr, object_qadr + 7))
    dynamic_dofs = set(range(object_dadr, object_dadr + 6))
    for gripper in (left_gripper, right_gripper):
        dynamic_qpos.update(gripper.dynamic_qpos_addresses.tolist())
        dynamic_dofs.update(gripper.dynamic_dof_addresses.tolist())
    kinematic_qpos = np.array([
        index for index in range(model.nq) if index not in dynamic_qpos
    ])
    kinematic_dofs = np.array([
        index for index in range(model.nv) if index not in dynamic_dofs
    ])

    viewer = None if headless else mujoco_viewer.MujocoViewer(model, data)
    if viewer is not None:
        viewer.cam.distance = 3.1
        viewer.cam.azimuth = 25
        viewer.cam.elevation = -18
    object_half_extent = (
        max(workpiece.half_size[1:]) if workpiece is not None else 0.04
    )
    detector = (
        InstanceSegmentationRgbdDetector(
            detector_weights,
            object_half_extent=object_half_extent,
            label="WORKPIECE",
        )
        if detector_kind == "yolo-seg"
        else BlueCubeRgbdDetector(
            minimum_area=120,
            object_half_extent=object_half_extent,
            aspect_range=(0.10, 10.0),
            yaw_symmetry=2,
            label="TRANSPORT OBJECT",
            maximum_depth=0.8,
        )
    )
    camera = HeadCameraWindow(
        model,
        viewer.window if viewer is not None else None,
        detector=detector,
        display=not headless,
    )
    execution_backend = MuJoCoDualArmBackend(
        model, data, left_solver, right_solver, left_planner, right_planner,
        left_gripper, right_gripper, camera, object_body, object_geom,
    )
    detection = camera.render(data)
    if detection is None:
        camera.close()
        if viewer is not None:
            viewer.close()
        raise RuntimeError("RGB-D detector could not locate the transport object")

    graph.start("observe_scene")
    graph.complete("observe_scene")
    pose_updater = BayesianBeliefUpdater(prior=0.55)
    pose_belief = pose_updater.fuse([
        confidence_evidence(detection.confidence, source="rgbd_detector"),
        binary_evidence(True, "valid_depth"),
    ])
    graph.update_belief("estimate_object_pose", pose_belief)
    graph.nodes["estimate_object_pose"].metadata["belief_trace"] = (
        pose_updater.trace
    )
    graph.start("estimate_object_pose")
    graph.complete("estimate_object_pose")

    perceived_center = np.asarray(detection.position, dtype=float)
    initial_object_rotation = _rpy_to_matrix(detection.rpy)
    evaluator = (
        RecoveryAwareEvaluator()
        if policy.recovery_aware_cost_enabled
        else RecoveryAwareEvaluator(
            risk_weight=0.0,
            clearance_weight=0.0,
            visibility_weight=0.0,
            stability_weight=0.0,
        )
    )
    profiles = []
    probes = {}
    span_by_plan = {}
    candidate_spans = (
        tuple(object_half_length * fraction for fraction in (0.55, 0.70, 0.82))
        if workpiece is not None else (0.11, GRASP_OFFSET_X, 0.15)
    )
    for span in candidate_spans:
        if object_half_length - span < FINGER_HALF_LENGTH_X:
            continue
        candidate_left = perceived_center + initial_object_rotation @ np.array([
            -span, 0.0, tool_object_offset_z,
        ])
        candidate_right = perceived_center + initial_object_rotation @ np.array([
            span, 0.0, tool_object_offset_z,
        ])
        left_probe = left_solver.solve(
            robot_qpos, candidate_left + [0.0, 0.0, PREGRASP_HEIGHT]
        )
        right_probe = right_solver.solve(
            left_probe.qpos, candidate_right + [0.0, 0.0, PREGRASP_HEIGHT]
        )
        grasp_seed = right_probe.qpos
        left_grasp_probe = left_probe
        right_grasp_probe = right_probe
        for descent_height in np.linspace(PREGRASP_HEIGHT, 0.0, 5)[1:]:
            left_grasp_probe = left_solver.solve(
                grasp_seed, candidate_left + [0.0, 0.0, descent_height]
            )
            right_grasp_probe = right_solver.solve(
                left_grasp_probe.qpos,
                candidate_right + [0.0, 0.0, descent_height],
            )
            if not left_grasp_probe.success or not right_grasp_probe.success:
                break
            grasp_seed = right_grasp_probe.qpos
        if not all(probe.success for probe in (
            left_probe, right_probe, left_grasp_probe, right_grasp_probe,
        )):
            continue
        motion_cost = float(
            np.linalg.norm(
                left_probe.qpos[left_solver.qpos_adrs]
                - robot_qpos[left_solver.qpos_adrs]
            )
            + np.linalg.norm(
                right_probe.qpos[right_solver.qpos_adrs]
                - robot_qpos[right_solver.qpos_adrs]
            )
        )
        stability = min(1.0, span / object_half_length)
        failure_probability = float(np.clip(
            0.03
            + 2.5 * (left_probe.error_norm + right_probe.error_norm)
            + 0.12 * (1.0 - stability),
            0.01, 0.45,
        ))
        plan_id = f"grasp_span_{span:.2f}"
        profiles.append(PlanRiskProfile(
            plan_id=plan_id,
            nominal_cost=motion_cost,
            failure_probability=failure_probability,
            recovery_costs=(1.5 + motion_cost, 2.5 + motion_cost),
            minimum_clearance=object_half_length - span,
            visibility=detection.confidence,
            grasp_stability=stability,
        ))
        probes[plan_id] = (
            left_probe, right_probe, left_grasp_probe, right_grasp_probe,
        )
        span_by_plan[plan_id] = span
    selected_risk, risk_evaluations = evaluator.select(profiles)
    selected_plan_id = selected_risk.profile.plan_id
    grasp_offset_x = span_by_plan[selected_plan_id]
    left_probe, right_probe, left_grasp_probe, right_grasp_probe = probes[selected_plan_id]
    left_target = perceived_center + initial_object_rotation @ np.array([
        -grasp_offset_x, 0.0, tool_object_offset_z,
    ])
    right_target = perceived_center + initial_object_rotation @ np.array([
        grasp_offset_x, 0.0, tool_object_offset_z,
    ])
    assignment = assign_primary_and_support(
        ArmCapability(
            "left", left_probe.success, 0.8, detection.confidence,
            left_probe.error_norm,
        ),
        ArmCapability(
            "right", right_probe.success, 0.8, detection.confidence,
            right_probe.error_norm,
        ),
    )
    graph.start("assign_arm_roles")
    graph.nodes["assign_arm_roles"].metadata["assignment"] = assignment
    graph.nodes["assign_arm_roles"].metadata["selected_plan"] = selected_risk
    graph.nodes["assign_arm_roles"].metadata["risk_candidates"] = risk_evaluations
    graph.complete("assign_arm_roles")
    print(
        "TASK GRAPH: dual-arm transport, "
        f"primary={assignment.primary}, support={assignment.support}, "
        f"grasp_span={grasp_offset_x:.2f}, "
        f"expected_cost={selected_risk.expected_cost:.3f}, "
        f"cvar={selected_risk.cvar:.3f}"
    )

    stage = "plan_dual_pregrasp"
    stage_start = float(data.time)
    run_start = stage_start
    trajectory = []
    goal = None
    contact_frames = 0
    next_render_time = 0.0
    next_camera_time = data.time
    next_observation_time = data.time
    latest_detection = detection
    active_directive = None
    directive_end_time = -1.0
    recovery_plan = None
    recovery_started_time = None
    recovery_contact_frames = 0
    recovery_phase = None
    recovery_settle_until = None
    diagnosis_count = 0
    maximum_sync_error = 0.0
    maximum_object_tilt = 0.0
    stage_motion_start = None
    stage_motion_goal = None
    obstacle_position = None
    trajectory_invalidated = False
    obstacle_motion_hold = False
    obstacle_safety_violation = False
    slip_retry_count = 0
    pending_probe = None
    pending_probe_observation = None
    probe_complete_time = -1.0
    diagnostic_hold_until = -1.0

    while float(data.time) - run_start < timeout:
        now = float(data.time)
        directive = fault_injector.poll(stage, now - stage_start)
        if directive is not None:
            active_directive = directive
            directive_end_time = now + float(
                directive.parameters.get("duration", 0.0)
            )
            if directive.action == "disable_left_contact":
                left_gripper.enable_cube_contacts(object_geom, False)
            elif directive.action == "disable_right_contact":
                right_gripper.enable_cube_contacts(object_geom, False)
            elif directive.action == "release_object":
                left_gripper.enable_cube_contacts(object_geom, False)
                right_gripper.enable_cube_contacts(object_geom, False)
                _set_grippers(execution_backend, False, False)
                angle = float(directive.parameters["angle"])
                speed = float(directive.parameters["lateral_speed"])
                data.qvel[object_dadr] = np.cos(angle) * speed
                data.qvel[object_dadr + 1] = np.sin(angle) * speed
                data.qvel[object_dadr + 5] = float(
                    directive.parameters["angular_speed"]
                )
            elif directive.action == "shift_target":
                data.qpos[object_qadr] += float(directive.parameters["dx"])
                data.qpos[object_qadr + 1] += float(directive.parameters["dy"])
                mujoco.mj_forward(model, data)
            elif (
                directive.action == "delay_arm"
                and stage_motion_start is not None
            ):
                delayed_adrs = (
                    left_solver.qpos_adrs
                    if directive.parameters["arm"] == "left"
                    else right_solver.qpos_adrs
                )
                robot_qpos[delayed_adrs] = stage_motion_start[delayed_adrs]
            elif directive.action == "insert_obstacle":
                remaining_path = (
                    ([] if goal is None else [goal]) + list(trajectory)
                )
                obstacle_position, blocked_waypoints = _place_dynamic_obstacle(
                    data,
                    obstacle_mocap,
                    left_planner,
                    right_planner,
                    left_solver,
                    right_solver,
                    robot_qpos,
                    remaining_path,
                    directive.parameters["path_fraction"],
                    directive.parameters["side"],
                )
                trajectory_invalidated = True
                obstacle_motion_hold = True
                monitor.record_event(
                    "dynamic_obstacle_inserted",
                    simulation_time=now,
                    position=obstacle_position,
                    target_side=directive.parameters["side"],
                    path_fraction=directive.parameters["path_fraction"],
                    blocked_waypoints=blocked_waypoints,
                )
            monitor.record_event(
                "fault_injected",
                simulation_time=now,
                stage=stage,
                directive=directive,
            )

        if active_directive is not None and now >= directive_end_time:
            if active_directive.action == "disable_left_contact":
                left_gripper.enable_cube_contacts(object_geom, True)
            elif active_directive.action == "disable_right_contact":
                right_gripper.enable_cube_contacts(object_geom, True)
            elif active_directive.action == "release_object":
                left_gripper.enable_cube_contacts(object_geom, True)
                right_gripper.enable_cube_contacts(object_geom, True)
            monitor.record_event(
                "fault_effect_ended",
                simulation_time=now,
                action=active_directive.action,
            )
            active_directive = None

        if (
            recovery_plan is not None
            and recovery_plan.fault.value == "bimanual_slip"
            and recovery_phase == "slip_lift"
            and recovery_started_time is not None
            and now - recovery_started_time >= 1.0
        ):
            left_contact = any(left_gripper.contact_flags(object_geom))
            right_contact = any(right_gripper.contact_flags(object_geom))
            if not left_contact and not right_contact:
                _set_grippers(execution_backend, False, False)
                trajectory = []
                goal = None
                slip_retry_count += 1
                recovery_phase = "slip_wait_ground"
                recovery_started_time = now
                recovery_contact_frames = 0
                monitor.record_event(
                    "slip_relift_redrop_detected",
                    simulation_time=now,
                    retry_count=slip_retry_count,
                    next_action="wait_ground_then_visual_regrasp",
                )
            elif left_contact != right_contact:
                drop = max(
                    0.0,
                    float(data.xpos[object_body, 2] - initial_object_position[2]),
                )
                drop = max(0.0, drop - 0.012)
                if left_contact:
                    _set_grippers(execution_backend, True, False)
                    failed_side = "right"
                else:
                    _set_grippers(execution_backend, False, True)
                    failed_side = "left"
                trajectory = _plan_dual_goal(
                    robot_qpos,
                    left_solver.tool_position(data) + [0.0, 0.0, -drop],
                    right_solver.tool_position(data) + [0.0, 0.0, -drop],
                    left_solver,
                    right_solver,
                    left_planner,
                    right_planner,
                    allow_object=True,
                )
                stage_motion_start = robot_qpos.copy()
                stage_motion_goal = trajectory[-1].copy()
                goal = trajectory.pop(0)
                recovery_phase = "slip_fallback_lower"
                slip_retry_count += 1
                recovery_started_time = now
                recovery_contact_frames = 0
                monitor.record_event(
                    "slip_relift_setdown_started",
                    simulation_time=now,
                    failed_side=failed_side,
                    strategy="support_arm_setdown_then_visual_dual_regrasp",
                    drop_distance=drop,
                )

        if (
            recovery_plan is not None
            and recovery_plan.fault.value == "bimanual_slip"
            and recovery_phase == "slip_regrasp"
            and recovery_started_time is not None
            and now - recovery_started_time >= 3.0
            and not trajectory
            and not any(left_gripper.contact_flags(object_geom))
            and not any(right_gripper.contact_flags(object_geom))
        ):
            _set_grippers(execution_backend, False, False)
            goal = None
            slip_retry_count += 1
            recovery_phase = "slip_wait_ground"
            recovery_started_time = now
            recovery_contact_frames = 0
            monitor.record_event(
                "dropped_object_regrasp_retry",
                simulation_time=now,
                retry_count=slip_retry_count,
                reason="no_contact_after_visual_regrasp",
            )

        if (
            recovery_plan is not None
            and recovery_plan.fault.value == "bimanual_slip"
            and recovery_phase == "slip_wait_ground"
            and abs(float(data.qvel[object_dadr + 2])) < 0.025
            and data.xpos[object_body, 2] <= initial_object_position[2] + 0.02
            and latest_detection is not None
            and latest_detection.confidence >= 0.55
        ):
            observed_center = np.asarray(latest_detection.position, dtype=float)
            observed_center[2] = max(
                observed_center[2], initial_object_position[2]
            )
            object_rotation = _rpy_to_matrix(latest_detection.rpy)
            recovery_grasp_z = tool_object_offset_z - min(
                0.012, 0.006 * slip_retry_count
            )
            trajectory = _plan_dual_goal(
                robot_qpos,
                observed_center + object_rotation @ np.array([
                    -grasp_offset_x, 0.0, recovery_grasp_z,
                ]),
                observed_center + object_rotation @ np.array([
                    grasp_offset_x, 0.0, recovery_grasp_z,
                ]),
                left_solver,
                right_solver,
                left_planner,
                right_planner,
                allow_object=True,
            )
            stage_motion_start = robot_qpos.copy()
            stage_motion_goal = trajectory[-1].copy()
            goal = trajectory.pop(0)
            recovery_phase = "slip_regrasp"
            recovery_started_time = now
            recovery_contact_frames = 0
            monitor.record_event(
                "dropped_object_found",
                simulation_time=now,
                target_source="waist_rgbd_6d_pose",
                perceived_object_position=observed_center,
                perceived_object_rpy=latest_detection.rpy,
                recovery_grasp_z=recovery_grasp_z,
                retry_count=slip_retry_count,
            )

        contact_recovery_active = (
            recovery_plan is not None
            and recovery_plan.fault.value in {
                "left_grasp_loss", "right_grasp_loss",
            }
        )
        if (
            contact_recovery_active
            and recovery_phase in {"regrasp", "level"}
            and recovery_started_time is not None
            and now - recovery_started_time >= LOCAL_REGRASP_TIMEOUT
        ):
            drop = max(
                0.0,
                float(data.xpos[object_body, 2] - initial_object_position[2]),
            )
            drop = max(0.0, drop - 0.015)
            trajectory = _plan_dual_goal(
                robot_qpos,
                left_solver.tool_position(data) + [0.0, 0.0, -drop],
                right_solver.tool_position(data) + [0.0, 0.0, -drop],
                left_solver,
                right_solver,
                left_planner,
                right_planner,
                allow_object=True,
            )
            stage_motion_start = robot_qpos.copy()
            stage_motion_goal = trajectory[-1].copy()
            goal = trajectory.pop(0)
            if recovery_plan.fault.value == "left_grasp_loss":
                _set_grippers(execution_backend, False, True)
            else:
                _set_grippers(execution_backend, True, False)
            recovery_phase = "fallback_lower"
            recovery_started_time = now
            recovery_contact_frames = 0
            monitor.record_event(
                "contact_recovery_fallback_started",
                simulation_time=now,
                strategy="support_arm_setdown_and_dual_regrasp",
                drop_distance=drop,
            )

        if (
            contact_recovery_active
            and recovery_phase == "fallback_settle"
            and recovery_settle_until is not None
            and now >= recovery_settle_until
        ):
            visual_center = np.asarray(latest_detection.position, dtype=float)
            observed_center = visual_center.copy()
            observed_center[2] = max(
                observed_center[2], initial_object_position[2],
            )
            object_rotation = _rpy_to_matrix(latest_detection.rpy)
            trajectory = _plan_dual_goal(
                robot_qpos,
                observed_center + object_rotation @ np.array([
                    -grasp_offset_x, 0.0, tool_object_offset_z,
                ]),
                observed_center + object_rotation @ np.array([
                    grasp_offset_x, 0.0, tool_object_offset_z,
                ]),
                left_solver,
                right_solver,
                left_planner,
                right_planner,
                allow_object=True,
            )
            stage_motion_start = robot_qpos.copy()
            stage_motion_goal = trajectory[-1].copy()
            goal = trajectory.pop(0)
            recovery_phase = "fallback_regrasp"
            recovery_settle_until = None
            monitor.record_event(
                "fallback_dual_regrasp_planned",
                simulation_time=now,
                visual_object_position=visual_center,
                perceived_object_position=observed_center,
                perceived_object_rpy=latest_detection.rpy,
            )

        if (
            contact_recovery_active
            and recovery_phase in {
                "fallback_lower", "fallback_settle", "fallback_regrasp",
                "fallback_lift",
            }
            and recovery_started_time is not None
            and now - recovery_started_time >= FALLBACK_RECOVERY_TIMEOUT
        ):
            raise RuntimeError("contact recovery fallback did not converge")
        if stage == "plan_dual_pregrasp" and goal is None and recovery_plan is None:
            if graph.nodes[stage].state.value == "ready":
                graph.start(stage)
            trajectory = _plan_dual_goal(
                robot_qpos,
                left_target + [0.0, 0.0, PREGRASP_HEIGHT],
                right_target + [0.0, 0.0, PREGRASP_HEIGHT],
                left_solver, right_solver, left_planner, right_planner,
                allow_object=True,
                precomputed_joint_goal=right_probe.qpos,
            )
            stage_motion_start = robot_qpos.copy()
            stage_motion_goal = trajectory[-1].copy()
            goal = trajectory.pop(0)
        elif stage == "dual_grasp" and goal is None and recovery_plan is None:
            if graph.nodes[stage].state.value == "ready":
                graph.start(stage)
            trajectory = _plan_dual_goal(
                robot_qpos, left_target, right_target,
                left_solver, right_solver, left_planner, right_planner,
                allow_object=True,
                precomputed_joint_goal=right_grasp_probe.qpos,
            )
            stage_motion_start = robot_qpos.copy()
            stage_motion_goal = trajectory[-1].copy()
            goal = trajectory.pop(0)
        elif stage == "cooperative_lift" and goal is None and recovery_plan is None:
            if graph.nodes[stage].state.value == "ready":
                graph.start(stage)
            remaining_lift = max(
                0.0,
                float(initial_object_position[2] + LIFT_HEIGHT - data.xpos[object_body, 2]),
            )
            trajectory = _plan_dual_goal(
                robot_qpos,
                left_solver.tool_position(data) + [0.0, 0.0, remaining_lift],
                right_solver.tool_position(data) + [0.0, 0.0, remaining_lift],
                left_solver, right_solver, left_planner, right_planner,
                allow_object=True,
            )
            stage_motion_start = robot_qpos.copy()
            stage_motion_goal = trajectory[-1].copy()
            goal = trajectory.pop(0)
        elif stage == "cooperative_transport" and goal is None and recovery_plan is None:
            if graph.nodes[stage].state.value == "ready":
                graph.start(stage)
            target_object_position = initial_object_position + np.array([
                0.0, TRANSPORT_OFFSET_Y, LIFT_HEIGHT,
            ])
            remaining_transport = (
                target_object_position - data.xpos[object_body]
            )
            trajectory = _plan_dual_goal(
                robot_qpos,
                left_solver.tool_position(data) + remaining_transport,
                right_solver.tool_position(data) + remaining_transport,
                left_solver, right_solver, left_planner, right_planner,
                allow_object=True,
            )
            stage_motion_start = robot_qpos.copy()
            stage_motion_goal = trajectory[-1].copy()
            goal = trajectory.pop(0)
        elif stage == "place_object" and goal is None and recovery_plan is None:
            if graph.nodes[stage].state.value == "ready":
                graph.start(stage)
            drop = max(
                0.0,
                float(data.xpos[object_body, 2] - initial_object_position[2]),
            )
            trajectory = _plan_dual_goal(
                robot_qpos,
                left_solver.tool_position(data) + [0.0, 0.0, -drop],
                right_solver.tool_position(data) + [0.0, 0.0, -drop],
                left_solver, right_solver, left_planner, right_planner,
                allow_object=True,
            )
            stage_motion_start = robot_qpos.copy()
            stage_motion_goal = trajectory[-1].copy()
            goal = trajectory.pop(0)
        elif stage == "retreat_arms" and goal is None and recovery_plan is None:
            if graph.nodes[stage].state.value == "ready":
                graph.start(stage)
            trajectory = _plan_dual_goal(
                robot_qpos,
                left_solver.tool_position(data)
                + [-RELEASE_RETREAT_X, 0.0, RELEASE_RETREAT_Z],
                right_solver.tool_position(data)
                + [RELEASE_RETREAT_X, 0.0, RELEASE_RETREAT_Z],
                left_solver,
                right_solver,
                left_planner,
                right_planner,
                allow_object=True,
            )
            stage_motion_start = robot_qpos.copy()
            stage_motion_goal = trajectory[-1].copy()
            goal = trajectory.pop(0)

        frozen_arm = None
        if (
            recovery_plan is None
            and active_directive is not None
            and active_directive.action in {
                "delay_arm", "freeze_left_arm", "freeze_right_arm",
            }
        ):
            frozen_arm = active_directive.parameters.get("arm")
        waiting_for_reobservation = (
            recovery_plan is not None
            and recovery_plan.fault.value == "visual_degradation"
        )
        if (
            goal is not None
            and not waiting_for_reobservation
            and not obstacle_motion_hold
            and now >= diagnostic_hold_until
            and _step_dual_toward(
            robot_qpos,
            goal,
            left_solver.qpos_adrs,
            right_solver.qpos_adrs,
            frozen_arm=frozen_arm,
            )
        ):
            waypoint_can_advance = True
            if recovery_plan is not None:
                contact_recovery = recovery_plan.fault.value in {
                    "left_grasp_loss", "right_grasp_loss", "bimanual_slip",
                }
                recovered_contact = True
                if contact_recovery and trajectory:
                    recovered_contact = False
                elif (
                    recovery_plan.fault.value == "dynamic_obstacle"
                    and trajectory
                ):
                    pass
                elif recovery_plan.fault.value == "left_grasp_loss":
                    if recovery_phase in {"fallback_regrasp", "fallback_lift"}:
                        contact_now = (
                            left_gripper.both_pads_touch(object_geom)
                            and right_gripper.both_pads_touch(object_geom)
                        )
                    else:
                        contact_now = left_gripper.both_pads_touch(object_geom)
                    if recovery_phase == "level":
                        contact_now = (
                            contact_now
                            and right_gripper.both_pads_touch(object_geom)
                        )
                    recovery_contact_frames = (
                        recovery_contact_frames + 1 if contact_now else 0
                    )
                    recovered_contact = recovery_contact_frames >= 25
                elif recovery_plan.fault.value == "right_grasp_loss":
                    if recovery_phase in {"fallback_regrasp", "fallback_lift"}:
                        contact_now = (
                            left_gripper.both_pads_touch(object_geom)
                            and right_gripper.both_pads_touch(object_geom)
                        )
                    else:
                        contact_now = right_gripper.both_pads_touch(object_geom)
                    if recovery_phase == "level":
                        contact_now = (
                            contact_now
                            and left_gripper.both_pads_touch(object_geom)
                        )
                    recovery_contact_frames = (
                        recovery_contact_frames + 1 if contact_now else 0
                    )
                    recovered_contact = recovery_contact_frames >= 25
                elif recovery_plan.fault.value == "bimanual_slip":
                    left_contact = any(left_gripper.contact_flags(object_geom))
                    right_contact = any(right_gripper.contact_flags(object_geom))
                    contact_now = left_contact and right_contact
                    recovery_contact_frames = (
                        recovery_contact_frames + 1 if contact_now else 0
                    )
                    recovered_contact = recovery_contact_frames >= 25
                if (
                    contact_recovery
                    and recovery_phase == "slip_fallback_lower"
                    and not trajectory
                ):
                    _set_grippers(execution_backend, False, False)
                    goal = None
                    recovery_phase = "slip_wait_ground"
                    recovery_started_time = now
                    recovery_contact_frames = 0
                    waypoint_can_advance = False
                    monitor.record_event(
                        "slip_relift_setdown_completed",
                        simulation_time=now,
                        next_action="waist_rgbd_reobserve",
                    )
                elif (
                    contact_recovery
                    and recovery_phase == "fallback_lower"
                    and not trajectory
                ):
                    _set_grippers(execution_backend, False, False)
                    goal = None
                    recovery_phase = "fallback_settle"
                    recovery_settle_until = now + FALLBACK_SETTLE_TIME
                    waypoint_can_advance = False
                    monitor.record_event(
                        "fallback_setdown_completed",
                        simulation_time=now,
                        settle_until=recovery_settle_until,
                    )
                elif contact_recovery and trajectory:
                    pass
                elif (
                    recovery_plan.fault.value == "dynamic_obstacle"
                    and trajectory
                ):
                    pass
                elif not recovered_contact:
                    _set_grippers(execution_backend, True, True)
                    waypoint_can_advance = False
                elif contact_recovery and recovery_phase == "regrasp":
                    left_position = left_solver.tool_position(data)
                    right_position = right_solver.tool_position(data)
                    yaw = float(latest_detection.rpy[2])
                    object_axis = np.array([np.cos(yaw), np.sin(yaw), 0.0])
                    if recovery_plan.fault.value == "left_grasp_loss":
                        level_left_target = left_position
                        level_right_target = (
                            left_position + 2.0 * grasp_offset_x * object_axis
                        )
                        anchor_side = "left"
                    else:
                        level_right_target = right_position
                        level_left_target = (
                            right_position - 2.0 * grasp_offset_x * object_axis
                        )
                        anchor_side = "right"
                    trajectory = _plan_dual_goal(
                        robot_qpos,
                        level_left_target,
                        level_right_target,
                        left_solver,
                        right_solver,
                        left_planner,
                        right_planner,
                        allow_object=True,
                    )
                    stage_motion_start = robot_qpos.copy()
                    stage_motion_goal = trajectory[-1].copy()
                    goal = trajectory.pop(0)
                    recovery_phase = "level"
                    recovery_contact_frames = 0
                    waypoint_can_advance = False
                    monitor.record_event(
                        "object_leveling_started",
                        simulation_time=now,
                        anchor_side=anchor_side,
                        left_target=level_left_target,
                        right_target=level_right_target,
                    )
                elif contact_recovery and recovery_phase == "fallback_regrasp":
                    remaining_lift = max(
                        0.0,
                        float(
                            initial_object_position[2]
                            + LIFT_HEIGHT
                            - data.xpos[object_body, 2]
                        ),
                    )
                    trajectory = _plan_dual_goal(
                        robot_qpos,
                        left_solver.tool_position(data)
                        + [0.0, 0.0, remaining_lift],
                        right_solver.tool_position(data)
                        + [0.0, 0.0, remaining_lift],
                        left_solver,
                        right_solver,
                        left_planner,
                        right_planner,
                        allow_object=True,
                    )
                    stage_motion_start = robot_qpos.copy()
                    stage_motion_goal = trajectory[-1].copy()
                    goal = trajectory.pop(0)
                    recovery_phase = "fallback_lift"
                    recovery_contact_frames = 0
                    waypoint_can_advance = False
                    monitor.record_event(
                        "fallback_cooperative_lift_started",
                        simulation_time=now,
                        lift_distance=remaining_lift,
                    )
                elif contact_recovery and recovery_phase == "slip_regrasp":
                    remaining_lift = max(
                        0.0,
                        float(
                            initial_object_position[2]
                            + LIFT_HEIGHT
                            - data.xpos[object_body, 2]
                        ),
                    )
                    trajectory = _plan_dual_goal(
                        robot_qpos,
                        left_solver.tool_position(data)
                        + [0.0, 0.0, remaining_lift],
                        right_solver.tool_position(data)
                        + [0.0, 0.0, remaining_lift],
                        left_solver,
                        right_solver,
                        left_planner,
                        right_planner,
                        allow_object=True,
                    )
                    stage_motion_start = robot_qpos.copy()
                    stage_motion_goal = trajectory[-1].copy()
                    goal = trajectory.pop(0)
                    recovery_phase = "slip_lift"
                    recovery_started_time = now
                    recovery_contact_frames = 0
                    waypoint_can_advance = False
                    monitor.record_event(
                        "dropped_object_relift_started",
                        simulation_time=now,
                        lift_distance=remaining_lift,
                    )
                elif (
                    recovery_plan.fault.value == "dynamic_obstacle"
                    and recovery_phase == "obstacle_retreat"
                ):
                    target_object_position = initial_object_position + np.array([
                        0.0, TRANSPORT_OFFSET_Y, LIFT_HEIGHT,
                    ])
                    remaining_transport = (
                        target_object_position - data.xpos[object_body]
                    )
                    trajectory = _plan_dual_goal(
                        robot_qpos,
                        left_solver.tool_position(data) + remaining_transport,
                        right_solver.tool_position(data) + remaining_transport,
                        left_solver,
                        right_solver,
                        left_planner,
                        right_planner,
                        allow_object=True,
                    )
                    stage_motion_start = robot_qpos.copy()
                    stage_motion_goal = trajectory[-1].copy()
                    goal = trajectory.pop(0)
                    recovery_phase = "obstacle_replan"
                    waypoint_can_advance = False
                    monitor.record_event(
                        "dynamic_obstacle_coupled_replan_started",
                        simulation_time=now,
                        remaining_transport=remaining_transport,
                        waypoint_count=len(trajectory) + 1,
                    )
                else:
                    completed_fault = recovery_plan.fault.value
                    graph.resume_after_recovery(
                        recovery_plan.recovery_node, stage
                    )
                    monitor.record_event(
                        "recovery_completed",
                        simulation_time=now,
                        stage=stage,
                        recovery_node=recovery_plan.recovery_node,
                        successful=True,
                        evidence="cooperative_constraints_restored",
                    )
                    experience_node = supervisor.mark_recovery_complete(
                        True, timestamp=float(data.time)
                    )
                    if experience_node is not None:
                        monitor.record_event(
                            "recovery_experience_updated",
                            simulation_time=float(data.time),
                            strategy_id=experience_node.strategy_id,
                            successes=experience_node.successes,
                            failures=experience_node.failures,
                            observed_cost=experience_node.costs[-1],
                        )
                    recovery_plan = None
                    recovery_started_time = None
                    recovery_contact_frames = 0
                    recovery_phase = None
                    if completed_fault == "dynamic_obstacle":
                        trajectory_invalidated = False
                    if completed_fault in {
                        "left_grasp_loss", "right_grasp_loss", "bimanual_slip",
                    }:
                        trajectory = []
                        goal = None
                        waypoint_can_advance = False
            if waypoint_can_advance:
                if trajectory:
                    goal = trajectory.pop(0)
                else:
                    goal = None
                    graph.complete(stage)
                    if stage == "plan_dual_pregrasp":
                        stage = "dual_grasp"
                        stage_start = now
                    elif stage == "dual_grasp":
                        stage = "verify_grasp"
                        stage_start = now
                    elif stage == "cooperative_lift":
                        stage = "cooperative_transport"
                        stage_start = now
                    elif stage == "cooperative_transport":
                        stage = "place_object"
                        stage_start = now
                    elif stage == "place_object":
                        _set_grippers(execution_backend, False, False)
                        stage = "retreat_arms"
                        stage_start = now
                    elif stage == "retreat_arms":
                        stage = "verify_completion"
                        stage_start = now

        if stage == "verify_grasp":
            if graph.nodes[stage].state.value == "ready":
                graph.start(stage)
            _set_grippers(execution_backend, True, True)
            all_contacts = (
                left_gripper.both_pads_touch(object_geom)
                and right_gripper.both_pads_touch(object_geom)
            )
            contact_frames = contact_frames + 1 if all_contacts else 0
            if contact_frames >= 12:
                graph.complete(stage)
                stage = "cooperative_lift"
                stage_start = now
                goal = None
            elif now - stage_start > 5.0:
                graph.fail(stage, "dual contact not established")
                raise RuntimeError("dual grasp verification failed")
        elif stage == "verify_completion":
            if graph.nodes[stage].state.value == "ready":
                graph.start(stage)
            if now - stage_start > 1.0:
                position = data.xpos[object_body]
                expected_y = initial_object_position[1] + TRANSPORT_OFFSET_Y
                success = (
                    abs(float(position[1] - expected_y)) < 0.07
                    and abs(float(position[2] - initial_object_position[2])) < 0.05
                )
                final_rpy = _matrix_to_rpy(data.xmat[object_body])
                final_tilt = max(abs(final_rpy[0]), abs(final_rpy[1]))
                success = success and final_tilt < 0.25
                success = success and not obstacle_safety_violation
                target_position = initial_object_position.copy()
                target_position[1] = expected_y
                final_position_error = float(
                    np.linalg.norm(position - target_position)
                )
                if success:
                    graph.complete(stage)
                    print(
                        "RESULT: SUCCESS dual-arm transport, object_position="
                        f"{position.round(4).tolist()}"
                    )
                else:
                    graph.fail(stage, "final object pose outside goal region")
                    print(
                        "RESULT: FAILURE final pose, object_position="
                        f"{position.round(4).tolist()}"
                    )
                camera.close()
                if viewer is not None:
                    viewer.close()
                monitor.record_summary(
                    success=success,
                    terminal_stage=stage,
                    simulation_duration=now - run_start,
                    recovery_count=supervisor.recovery_attempts,
                    diagnosis_count=diagnosis_count,
                    maximum_synchronization_error=maximum_sync_error,
                    maximum_object_tilt=maximum_object_tilt,
                    final_object_position=position,
                    final_position_error=final_position_error,
                    final_object_rpy=final_rpy,
                    final_object_tilt=final_tilt,
                    obstacle_safety_violation=obstacle_safety_violation,
                    expected_recovery_cost=selected_risk.expected_cost,
                    cvar=selected_risk.cvar,
                    graph_rollback_depth=max(
                        (
                            graph.rollback_depth(item[1])
                            for item in graph.history
                            if item[0] == "fail"
                        ),
                        default=0,
                    ),
                    graph_rollback_cost=(
                        graph.last_repair_plan.rollback_cost
                        if graph.last_repair_plan is not None else 0.0
                    ),
                    preserved_task_nodes=(
                        len(graph.last_repair_plan.preserved_succeeded_nodes)
                        if graph.last_repair_plan is not None else 0
                    ),
                )
                if experience_path is not None and update_experience:
                    experience_path.parent.mkdir(parents=True, exist_ok=True)
                    experience_graph.save(experience_path)
                monitor.close()
                return success

        data.qpos[kinematic_qpos] = robot_qpos[kinematic_qpos]
        data.qvel[kinematic_dofs] = 0.0
        left_gripper.follow_wrist()
        right_gripper.follow_wrist()
        mujoco.mj_step(model, data)

        if obstacle_position is not None and not obstacle_safety_violation:
            dynamic_reasons = tuple(
                reason
                for planner in (left_planner, right_planner)
                for reason in planner.configuration_collision_reasons(
                    robot_qpos, allow_cube=True
                )
                if (
                    "dynamic_obstacle" in reason
                    and not reason.startswith("bounds:")
                )
            )
            if dynamic_reasons:
                obstacle_safety_violation = True
                monitor.record_event(
                    "dynamic_obstacle_safety_violation",
                    simulation_time=float(data.time),
                    reasons=dynamic_reasons,
                )

        if data.time >= next_camera_time:
            latest = camera.render(data)
            if latest is not None:
                latest_detection = latest
                graph.update_belief("verify_completion", latest.confidence)
                if (
                    recovery_plan is not None
                    and recovery_plan.fault.value == "visual_degradation"
                    and recovery_started_time is not None
                    and float(data.time) > recovery_started_time
                ):
                    graph.update_belief(stage, latest.confidence)
                    graph.resume_after_recovery(
                        recovery_plan.recovery_node, stage
                    )
                    monitor.record_event(
                        "recovery_completed",
                        simulation_time=float(data.time),
                        stage=stage,
                        recovery_node=recovery_plan.recovery_node,
                        successful=True,
                        evidence="valid_rgbd_frame",
                        visual_confidence=float(latest.confidence),
                    )
                    experience_node = supervisor.mark_recovery_complete(
                        True, timestamp=float(data.time)
                    )
                    if experience_node is not None:
                        monitor.record_event(
                            "recovery_experience_updated",
                            simulation_time=float(data.time),
                            strategy_id=experience_node.strategy_id,
                            successes=experience_node.successes,
                            failures=experience_node.failures,
                            observed_cost=experience_node.costs[-1],
                        )
                    recovery_plan = None
                    recovery_started_time = None
            if viewer is not None:
                glfw.make_context_current(viewer.window)
            next_camera_time = data.time + CAMERA_RENDER_DT

        if data.time >= next_observation_time:
            portable_observation = execution_backend.observe()
            left_tracking = _tracking_error(
                robot_qpos, goal, left_solver.qpos_adrs
            )
            right_tracking = _tracking_error(
                robot_qpos, goal, right_solver.qpos_adrs
            )
            left_phase = _trajectory_phase(
                robot_qpos,
                stage_motion_start,
                stage_motion_goal,
                left_solver.qpos_adrs,
            )
            right_phase = _trajectory_phase(
                robot_qpos,
                stage_motion_start,
                stage_motion_goal,
                right_solver.qpos_adrs,
            )
            synchronization_error = abs(left_phase - right_phase)
            perceived_object_pose = portable_observation.object_pose
            object_rpy = (
                perceived_object_pose.rpy
                if perceived_object_pose is not None
                else _matrix_to_rpy(data.xmat[object_body])
            )
            if stage in {
                "dual_grasp",
                "cooperative_lift",
                "cooperative_transport",
                "place_object",
            }:
                maximum_sync_error = max(
                    maximum_sync_error, synchronization_error
                )
            maximum_object_tilt = max(
                maximum_object_tilt, abs(object_rpy[0]), abs(object_rpy[1])
            )
            vision_masked = (
                active_directive is not None
                and active_directive.action in {
                    "mask_vision", "invalidate_sensors",
                }
            )
            observation = DualArmObservation(
                timestamp=portable_observation.timestamp,
                stage=stage,
                object_position=(
                    perceived_object_pose.position
                    if perceived_object_pose is not None
                    else tuple(float(value) for value in data.xpos[object_body])
                ),
                object_rpy=object_rpy,
                left_tool_position=portable_observation.left.tool_pose.position,
                right_tool_position=portable_observation.right.tool_pose.position,
                left_contacts=portable_observation.left.pad_contacts,
                right_contacts=portable_observation.right.pad_contacts,
                left_force=float(portable_observation.left.force[2]),
                right_force=float(portable_observation.right.force[2]),
                visual_confidence=(
                    0.0 if vision_masked else float(latest_detection.confidence)
                ),
                visual_valid=not vision_masked,
                object_vertical_velocity=portable_observation.object_linear_velocity[2],
                synchronization_error=synchronization_error,
                left_tracking_error=left_tracking,
                right_tracking_error=right_tracking,
                collision=trajectory_invalidated,
                event=(active_directive.action if active_directive else None),
                metadata={
                    "left_trajectory_phase": left_phase,
                    "right_trajectory_phase": right_phase,
                },
            )
            monitor.record_observation(observation)
            if (
                recovery_enabled
                and recovery_plan is None
                and graph.nodes[stage].state.value == "running"
            ):
                if pending_probe is not None and now >= probe_complete_time:
                    probe_positive = evaluate_probe_outcome(
                        pending_probe, pending_probe_observation, observation
                    )
                    probe_estimate = supervisor.incorporate_probe_result(
                        pending_probe, probe_positive
                    )
                    monitor.record_event(
                        "active_probe_completed",
                        simulation_time=now,
                        stage=stage,
                        probe=pending_probe,
                        positive=probe_positive,
                        posterior=(
                            {
                                fault.value: probability
                                for fault, probability
                                in probe_estimate.posterior.items()
                            }
                            if probe_estimate is not None else None
                        ),
                        entropy=(
                            probe_estimate.entropy
                            if probe_estimate is not None else None
                        ),
                    )
                    pending_probe = None
                    pending_probe_observation = None
                decision = supervisor.observe(observation)
                if decision.fault_distribution is not None:
                    monitor.record_event(
                        "temporal_fault_belief_updated",
                        simulation_time=float(data.time),
                        stage=stage,
                        confirmed_fault=decision.confirmed_fault.value,
                        posterior=decision.fault_distribution,
                        leading_probability=decision.fault_posterior,
                        entropy=decision.belief_entropy,
                        selected_probe=decision.selected_probe,
                        expected_information_gain=(
                            decision.expected_information_gain
                        ),
                    )
                if decision.report.anomalous:
                    diagnosis_count += 1
                    monitor.record_diagnosis(decision.report)
                if (
                    diagnosis_ablation == "full"
                    and decision.selected_probe is not None
                    and decision.recovery_plan is None
                    and pending_probe is None
                ):
                    try:
                        specification = probe_specification(
                            decision.selected_probe
                        )
                    except ValueError:
                        specification = None
                    if specification is not None:
                        pending_probe = specification.name
                        pending_probe_observation = observation
                        probe_complete_time = now + specification.hold_seconds
                        diagnostic_hold_until = max(
                            diagnostic_hold_until, probe_complete_time
                        )
                        if specification.camera_view is not None:
                            view_result = execution_backend.set_camera_view(
                                specification.camera_view
                            )
                            if view_result.accepted:
                                refreshed = camera.render(data)
                                if refreshed is not None:
                                    latest_detection = refreshed
                        monitor.record_event(
                            "active_probe_started",
                            simulation_time=now,
                            stage=stage,
                            probe=pending_probe,
                            hold_seconds=specification.hold_seconds,
                            camera_view=specification.camera_view,
                            expected_information_gain=(
                                decision.expected_information_gain
                            ),
                            prior_entropy=decision.belief_entropy,
                        )
                if decision.recovery_plan is not None:
                    recovery_plan = decision.recovery_plan
                    if recovery_planner.last_llm_audit is not None:
                        monitor.record_event(
                            "llm_candidate_audit",
                            simulation_time=now,
                            **recovery_planner.last_llm_audit.__dict__,
                        )
                    recovery_started_time = float(data.time)
                    recovery_contact_frames = 0
                    recovery_phase = None
                    monitor.record_recovery_plan(recovery_plan)
                    graph.fail(
                        stage,
                        recovery_plan.fault.value,
                        recovery_node_id=recovery_plan.recovery_node,
                    )
                    graph.start(recovery_plan.recovery_node)
                    if active_directive is not None:
                        if active_directive.action == "disable_left_contact":
                            left_gripper.enable_cube_contacts(object_geom, True)
                        elif active_directive.action == "disable_right_contact":
                            right_gripper.enable_cube_contacts(object_geom, True)
                        elif active_directive.action == "release_object":
                            left_gripper.enable_cube_contacts(object_geom, True)
                            right_gripper.enable_cube_contacts(object_geom, True)
                        active_directive = None
                    if recovery_plan.fault.value in {
                        "left_grasp_loss", "right_grasp_loss",
                    }:
                        recovery_phase = "regrasp"
                        observed_center = np.asarray(
                            latest_detection.position, dtype=float
                        )
                        object_rotation = _rpy_to_matrix(latest_detection.rpy)
                        left_recovery_target = left_solver.tool_position(data)
                        right_recovery_target = right_solver.tool_position(data)
                        if recovery_plan.fault.value == "left_grasp_loss":
                            failed_side = "left"
                            support_position = right_recovery_target
                        else:
                            failed_side = "right"
                            support_position = left_recovery_target
                        (
                            trajectory,
                            left_recovery_target,
                            right_recovery_target,
                            observed_center,
                            recovery_grasp_span,
                        ) = _plan_contact_anchored_regrasp(
                            robot_qpos,
                            failed_side,
                            support_position,
                            object_rotation,
                            observed_center,
                            grasp_offset_x,
                            tool_object_offset_z,
                            left_solver,
                            right_solver,
                            left_planner,
                            right_planner,
                        )
                        stage_motion_start = robot_qpos.copy()
                        stage_motion_goal = trajectory[-1].copy()
                        goal = trajectory.pop(0)
                        _set_grippers(execution_backend, True, True)
                        monitor.record_event(
                            "regrasp_planned",
                            simulation_time=float(data.time),
                            failed_side=(
                                "left"
                                if recovery_plan.fault.value == "left_grasp_loss"
                                else "right"
                            ),
                            target_source="rgbd_orientation_plus_support_contact",
                            perceived_object_rpy=latest_detection.rpy,
                            fused_object_center=observed_center,
                            left_target=left_recovery_target,
                            right_target=right_recovery_target,
                            recovery_grasp_span=recovery_grasp_span,
                        )
                        if recovery_plan.strategy_id == "setdown_dual_regrasp":
                            # Trigger the verified support-arm setdown transaction
                            # on the next control cycle instead of retrying in air.
                            recovery_started_time = (
                                now - LOCAL_REGRASP_TIMEOUT
                            )
                            monitor.record_event(
                                "recovery_strategy_activated",
                                simulation_time=now,
                                strategy_id=recovery_plan.strategy_id,
                                execution_mode="immediate_support_arm_setdown",
                            )
                    elif recovery_plan.fault.value == "bimanual_slip":
                        trajectory = []
                        goal = None
                        _set_grippers(execution_backend, False, False)
                        camera_result = execution_backend.set_camera_view(
                            "waist_search"
                        )
                        if not camera_result.accepted:
                            raise RuntimeError(camera_result.reason)
                        recovery_phase = "slip_wait_ground"
                        slip_retry_count = (
                            1 if recovery_plan.strategy_id == "safe_setdown_regrasp"
                            else 0
                        )
                        monitor.record_event(
                            "recovery_strategy_activated",
                            simulation_time=now,
                            strategy_id=recovery_plan.strategy_id,
                            execution_mode=(
                                "lower_conservative_regrasp"
                                if slip_retry_count else "direct_visual_regrasp"
                            ),
                        )
                        monitor.record_event(
                            "dropped_object_search_started",
                            simulation_time=float(data.time),
                            camera="waist_rgbd",
                            search_mode="look_down_then_visual_6d_regrasp",
                        )
                    elif recovery_plan.fault.value == "dynamic_obstacle":
                        if obstacle_position is None:
                            raise RuntimeError(
                                "dynamic obstacle recovery has no obstacle pose"
                            )
                        tool_center = 0.5 * (
                            left_solver.tool_position(data)
                            + right_solver.tool_position(data)
                        )
                        retreat_direction = tool_center - obstacle_position
                        retreat_norm = float(np.linalg.norm(retreat_direction))
                        if retreat_norm < 1e-8:
                            retreat_direction = np.array([0.0, 1.0, 0.0])
                        else:
                            retreat_direction /= retreat_norm
                        horizontal_direction = retreat_direction.copy()
                        horizontal_direction[2] = 0.0
                        horizontal_norm = float(
                            np.linalg.norm(horizontal_direction)
                        )
                        if horizontal_norm < 1e-8:
                            horizontal_direction = np.array([0.0, 1.0, 0.0])
                        else:
                            horizontal_direction /= horizontal_norm
                        tangent_direction = np.array([
                            -horizontal_direction[1],
                            horizontal_direction[0],
                            0.0,
                        ])
                        retreat_directions = (
                            retreat_direction,
                            horizontal_direction,
                            tangent_direction,
                            -tangent_direction,
                        )
                        retreat_failures = []
                        trajectory = None
                        retreat_delta = None
                        for candidate_direction in retreat_directions:
                            for retreat_distance in OBSTACLE_RETREAT_DISTANCES:
                                candidate_delta = (
                                    retreat_distance * candidate_direction
                                )
                                candidate_delta[2] = np.clip(
                                    candidate_delta[2],
                                    -OBSTACLE_MAX_VERTICAL_RETREAT,
                                    OBSTACLE_MAX_VERTICAL_RETREAT,
                                )
                                try:
                                    candidate_trajectory = _plan_dual_goal(
                                        robot_qpos,
                                        left_solver.tool_position(data)
                                        + candidate_delta,
                                        right_solver.tool_position(data)
                                        + candidate_delta,
                                        left_solver,
                                        right_solver,
                                        left_planner,
                                        right_planner,
                                        allow_object=True,
                                        allow_start_in_collision=True,
                                    )
                                except RuntimeError as error:
                                    retreat_failures.append(str(error))
                                    continue
                                hard_collision_waypoints = []
                                previous_waypoint = robot_qpos
                                for waypoint_index, waypoint in enumerate(
                                    candidate_trajectory
                                ):
                                    joint_distance = float(np.max(np.abs(
                                        waypoint - previous_waypoint
                                    )))
                                    sample_count = max(
                                        2,
                                        int(np.ceil(
                                            joint_distance / MAX_JOINT_DELTA
                                        )) + 1,
                                    )
                                    for alpha in np.linspace(
                                        0.0, 1.0, sample_count
                                    ):
                                        sample = (
                                            previous_waypoint
                                            + alpha
                                            * (waypoint - previous_waypoint)
                                        )
                                        hard_reasons = tuple(
                                            reason
                                            for planner in (
                                                left_planner, right_planner,
                                            )
                                            for reason in planner.configuration_collision_reasons(
                                                sample, allow_cube=True
                                            )
                                            if (
                                                "dynamic_obstacle" in reason
                                                and not reason.startswith("bounds:")
                                            )
                                        )
                                        if hard_reasons:
                                            hard_collision_waypoints.append(
                                                (
                                                    waypoint_index,
                                                    round(float(alpha), 3),
                                                    hard_reasons,
                                                )
                                            )
                                            break
                                    previous_waypoint = waypoint
                                if hard_collision_waypoints:
                                    retreat_failures.append(
                                        "retreat enters hard obstacle collision: "
                                        f"{hard_collision_waypoints[:3]}"
                                    )
                                    continue
                                retreat_delta = candidate_delta
                                trajectory = candidate_trajectory
                                break
                            if trajectory is not None:
                                break
                        if trajectory is None:
                            raise RuntimeError(
                                "no bounded dynamic-obstacle retreat is feasible: "
                                f"{retreat_failures}"
                            )
                        stage_motion_start = robot_qpos.copy()
                        stage_motion_goal = trajectory[-1].copy()
                        goal = trajectory.pop(0)
                        recovery_phase = "obstacle_retreat"
                        trajectory_invalidated = False
                        obstacle_motion_hold = False
                        monitor.record_event(
                            "dynamic_obstacle_retreat_started",
                            simulation_time=float(data.time),
                            obstacle_position=obstacle_position,
                            retreat_delta=retreat_delta,
                        )
                    monitor.record_event(
                        "recovery_started",
                        simulation_time=float(data.time),
                        stage=stage,
                        recovery_node=recovery_plan.recovery_node,
                        actions=recovery_plan.actions,
                    )
            next_observation_time = data.time + OBSERVATION_DT
        if viewer is not None and data.time >= next_render_time:
            glfw.make_context_current(viewer.window)
            viewer.render()
            glfw.poll_events()
            next_render_time = data.time + VIEWER_RENDER_DT
            if not viewer.is_alive or glfw.window_should_close(viewer.window):
                break

    camera.close()
    if viewer is not None:
        viewer.close()
    failed_experience = supervisor.mark_recovery_complete(
        False, timestamp=float(data.time)
    )
    if failed_experience is not None:
        monitor.record_event(
            "recovery_experience_updated",
            simulation_time=float(data.time),
            strategy_id=failed_experience.strategy_id,
            successes=failed_experience.successes,
            failures=failed_experience.failures,
            observed_cost=failed_experience.costs[-1],
        )
    print(f"RESULT: TIMEOUT stage={stage}")
    monitor.record_summary(
        success=False,
        terminal_stage=stage,
        simulation_duration=float(data.time) - run_start,
        recovery_count=supervisor.recovery_attempts,
        diagnosis_count=diagnosis_count,
        maximum_synchronization_error=maximum_sync_error,
        maximum_object_tilt=maximum_object_tilt,
        obstacle_safety_violation=obstacle_safety_violation,
        expected_recovery_cost=selected_risk.expected_cost,
        cvar=selected_risk.cvar,
        graph_rollback_depth=max(
            (
                graph.rollback_depth(item[1])
                for item in graph.history
                if item[0] == "fail"
            ),
            default=0,
        ),
        graph_rollback_cost=(
            graph.last_repair_plan.rollback_cost
            if graph.last_repair_plan is not None else 0.0
        ),
        preserved_task_nodes=(
            len(graph.last_repair_plan.preserved_succeeded_nodes)
            if graph.last_repair_plan is not None else 0
        ),
        failure_reason="timeout",
    )
    if experience_path is not None and update_experience:
        experience_path.parent.mkdir(parents=True, exist_ok=True)
        experience_graph.save(experience_path)
    monitor.close()
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument(
        "--fault",
        choices=tuple(item.value for item in DualArmFaultScenario),
        default="none",
    )
    parser.add_argument("--fault-severity", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--policy",
        choices=tuple(item.value for item in ExperimentPolicy),
        default="full",
    )
    parser.add_argument("--log", type=Path)
    parser.add_argument("--scene-jitter", type=float, default=0.0)
    parser.add_argument("--experience-graph", type=Path)
    parser.add_argument(
        "--experience-mode",
        choices=("off", "online", "train", "frozen"),
        default="online",
    )
    parser.add_argument(
        "--experience-ablation",
        choices=("success_only", "no_cvar", "full"),
        default="full",
    )
    parser.add_argument(
        "--diagnosis-ablation",
        choices=("full", "no_active_probe", "no_temporal"),
        default="full",
    )
    parser.add_argument("--workpiece-domain", choices=("train", "unseen"))
    parser.add_argument("--detector", choices=("color", "yolo-seg"), default="color")
    parser.add_argument("--detector-weights", type=Path)
    parser.add_argument("--qwen-model", default="qwen-plus")
    parser.add_argument(
        "--qwen-base-url",
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    parser.add_argument("--qwen-timeout", type=float, default=12.0)
    parser.add_argument("--llm-replay-response", type=Path)
    args = parser.parse_args()
    try:
        success = run(
            headless=args.headless,
            timeout=args.timeout,
            fault=args.fault,
            fault_severity=args.fault_severity,
            seed=args.seed,
            policy=args.policy,
            log_path=args.log,
            scene_jitter=args.scene_jitter,
            experience_path=args.experience_graph,
            workpiece_domain=args.workpiece_domain,
            detector_kind=args.detector,
            detector_weights=args.detector_weights,
            experience_mode=args.experience_mode,
            experience_ablation=args.experience_ablation,
            diagnosis_ablation=args.diagnosis_ablation,
            qwen_model=args.qwen_model,
            qwen_base_url=args.qwen_base_url,
            qwen_timeout=args.qwen_timeout,
            llm_replay_response=args.llm_replay_response,
        )
    except Exception as error:
        failure_monitor = OperationMonitor(args.log, append=True)
        failure_monitor.record_summary(
            success=False,
            terminal_stage="exception",
            failure_reason=f"{type(error).__name__}: {error}",
        )
        failure_monitor.close()
        raise
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
