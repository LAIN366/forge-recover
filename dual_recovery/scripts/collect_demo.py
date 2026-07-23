#!/usr/bin/env python3
"""Cruzr S2 MuJoCo teleoperation demo with a dual-table scene.

The script converts the URDF to MJCF, then injects scene assets and bodies
directly into that MJCF. It intentionally does not include a second robot XML.
"""

from pathlib import Path
import time

import glfw
import mujoco
import mujoco_viewer
import numpy as np

from cruzr_sim.perception.head_camera import HeadCameraWindow


ROBOT_DIR = Path("/home/lain/robosuite_ws/dual_recovery")
URDF_FILE = ROBOT_DIR / "robots" / "cruzr_s2_v1.urdf"
SAVE_DIR = Path("/home/lain/robosuite_ws/datasets/test_task")
BASE_MJCF = Path("/tmp/cruzr_s2_base.xml")
SCENE_MJCF = Path("/tmp/cruzr_s2_dual_table_scene.xml")
ROBOT_BASE_LIFT = 0.14
ROBOT_ROOT_CANDIDATES = ("base_link", "base_footprint", "base", "robot_base")
KINEMATIC_TELEOP = True
BASE_SPEED = 0.25
TURN_SPEED = 0.45
# Manual teleoperation is a kinematic visualization/recording tool.  The
# imported robot contains detailed STL collision meshes, which are needlessly
# expensive to test against the floor and tables on every render frame.
# Keep this False for smooth teleoperation. The autonomous demo has its own
# contact-enabled scene for collision checks.
TELEOP_ENABLE_CONTACTS = False
TELEOP_FRAME_DT = 1.0 / 50.0
TELEOP_RENDER_DT = 1.0 / 50.0
HEAD_CAMERA_RENDER_DT = 1.0 / 20.0
# The cube table center is (-1.25, 0.57). Its long side is parallel to X.
# Start alongside the long Y-negative edge, facing the table along +Y.
BASE_INITIAL_X = -1.25
BASE_INITIAL_Y = 0.0
BASE_INITIAL_YAW = np.pi / 2

# Each entry is (MuJoCo joint name, label shown in the terminal).
CONTROL_JOINTS = [
    ("driving_wheel_left_joint", "left wheel"),
    ("driving_wheel_right_joint", "right wheel"),
    ("L_shoulder_pitch_joint", "left shoulder pitch"),
    ("L_shoulder_roll_joint", "left shoulder roll"),
    ("L_shoulder_yaw_joint", "left shoulder yaw"),
    ("L_elbow_roll_joint", "left elbow roll"),
    ("L_elbow_yaw_joint", "left elbow yaw"),
    ("L_wrist_pitch_joint", "left wrist pitch"),
    ("L_wrist_roll_joint", "left wrist roll"),
    ("L_finger1_joint", "left finger 1"),
    ("L_finger2_joint", "left finger 2"),
    ("R_shoulder_pitch_joint", "right shoulder pitch"),
    ("R_shoulder_roll_joint", "right shoulder roll"),
    ("R_shoulder_yaw_joint", "right shoulder yaw"),
    ("R_elbow_roll_joint", "right elbow roll"),
    ("R_elbow_yaw_joint", "right elbow yaw"),
    ("R_wrist_pitch_joint", "right wrist pitch"),
    ("R_wrist_roll_joint", "right wrist roll"),
    ("R_finger1_joint", "right finger 1"),
    ("R_finger2_joint", "right finger 2"),
]

LOCKED_JOINTS = [
    "lifter_pitch_1_joint",
    "lifter_pitch_2_joint",
    "lifter_pitch_3_joint",
    "waist_yaw_joint",
    "head_yaw_joint",
    "head_pitch_joint",
]

ASSET_XML = """
<texture name="checker_tex" type="2d" builtin="checker"
         rgb1="0.09 0.26 0.43" rgb2="0.20 0.46 0.66"
         width="512" height="512"/>
<texture name="sky_tex" type="skybox" builtin="gradient"
         rgb1="0.38 0.62 0.84" rgb2="0.08 0.16 0.28"
         width="512" height="512"/>
<material name="checker_floor" texture="checker_tex" texrepeat="10 10"
          texuniform="true" reflectance="0.04"/>
<material name="table_top" rgba="0.30 0.24 0.17 1"/>
<material name="table_leg" rgba="0.13 0.09 0.06 1"/>
<material name="cube_blue" rgba="0.04 0.20 0.95 1"/>
"""

# The robot stays at the MJCF origin. Move table positions to match the
# actual facing direction of the converted Cruzr S2 model if needed.
WORLD_XML = """
<light name="key_light" pos="0 -2 5.5" dir="0 0 -1"
       directional="true" diffuse="0.34 0.34 0.34" specular="0.03 0.03 0.03"/>
<light name="fill_light" pos="-3 2 3" dir="1 -0.5 -0.6"
       directional="true" diffuse="0.20 0.23 0.28"/>
<light name="rim_light" pos="3 1 4" dir="-1 -0.3 -0.8"
       directional="true" diffuse="0.12 0.14 0.18"/>
<!-- A plane has infinite collision extent; size controls its rendered area. -->
<geom name="floor" type="plane" size="500 500 0.1" material="checker_floor"
      friction="1 0.005 0.0001"/>
<camera name="overview" pos="4.8 -5.6 3.7"
        xyaxes="0.76 0.65 0 -0.30 0.35 0.89" fovy="48"/>

<body name="left_table" pos="-1.25 0.57 0">
  <geom name="left_table_top" type="box" size="0.40 0.30 0.02"
        pos="0 0 0.75" material="table_top"/>
  <geom type="box" size="0.025 0.025 0.375" pos=" 0.34  0.24 0.375" material="table_leg"/>
  <geom type="box" size="0.025 0.025 0.375" pos=" 0.34 -0.24 0.375" material="table_leg"/>
  <geom type="box" size="0.025 0.025 0.375" pos="-0.34  0.24 0.375" material="table_leg"/>
  <geom type="box" size="0.025 0.025 0.375" pos="-0.34 -0.24 0.375" material="table_leg"/>
</body>

<body name="right_table" pos="1.25 0.75 0">
  <geom name="right_table_top" type="box" size="0.40 0.30 0.02"
        pos="0 0 0.75" material="table_top"/>
  <geom type="box" size="0.025 0.025 0.375" pos=" 0.34  0.24 0.375" material="table_leg"/>
  <geom type="box" size="0.025 0.025 0.375" pos=" 0.34 -0.24 0.375" material="table_leg"/>
  <geom type="box" size="0.025 0.025 0.375" pos="-0.34  0.24 0.375" material="table_leg"/>
  <geom type="box" size="0.025 0.025 0.375" pos="-0.34 -0.24 0.375" material="table_leg"/>
</body>

<!-- Table top is at z = 0.77. Cube center is 0.03 m above it. -->
<body name="blue_cube" pos="-1.25 0.57 0.80">
  <freejoint/>
  <geom name="blue_cube_geom" type="box" size="0.03 0.03 0.03"
        mass="0.1" material="cube_blue" friction="1 0.005 0.0001"/>
</body>
"""


def insert_before_close(xml_text: str, tag: str, content: str) -> str:
    closing_tag = f"</{tag}>"
    if closing_tag not in xml_text:
        raise RuntimeError(f"Converted MJCF has no {closing_tag} section.")
    return xml_text.replace(closing_tag, f"{content}\n{closing_tag}", 1)


def joint_id(model: mujoco.MjModel, name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)


def prepare_robot_scene(model: mujoco.MjModel, data: mujoco.MjData, robot_geom_count: int,
                        enable_contacts: bool = True, robot_geom_ids=None):
    """Translate the complete fixed-base URDF import and disable self contacts.

    MuJoCo omits a fixed URDF base_link body and puts its mesh directly in
    worldbody. Its immediate joint children are also worldbody children. To
    move the robot without breaking the chassis-wheel alignment, translate
    both these child bodies and all original world-attached robot geoms.
    """
    scene_bodies = {
        "left_table", "right_table", "blue_cube", "dynamic_obstacle",
    }
    robot_roots = [
        body_id for body_id in range(1, model.nbody)
        if model.body_parentid[body_id] == 0
        and mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) not in scene_bodies
    ]
    if not robot_roots:
        raise RuntimeError("Could not identify robot bodies under worldbody.")

    for body_id in robot_roots:
        model.body_pos[body_id, 2] += ROBOT_BASE_LIFT

    if robot_geom_ids is None:
        robot_geom_ids = np.arange(robot_geom_count, dtype=np.int32)
    robot_geom_ids = np.asarray(robot_geom_ids, dtype=np.int32)
    robot_geom_set = set(robot_geom_ids.tolist())

    # Geoms attached to body 0 are the fixed base_link shell and must move
    # with the robot during one-time scene placement.
    moved_world_geoms = 0
    for geom_id in robot_geom_ids:
        if model.geom_bodyid[geom_id] == 0:
            model.geom_pos[geom_id, 2] += ROBOT_BASE_LIFT
            moved_world_geoms += 1
    mujoco.mj_setConst(model, data)

    if enable_contacts:
        # Disable robot self collision but retain robot-vs-environment
        # contacts for autonomous collision validation.
        for geom_id in robot_geom_ids:
            model.geom_contype[geom_id] = 1
            model.geom_conaffinity[geom_id] = 2
        for geom_id in range(model.ngeom):
            if geom_id in robot_geom_set:
                continue
            model.geom_contype[geom_id] = 2
            model.geom_conaffinity[geom_id] = 1
        contact_status = "Enabled robot-environment collision checks"
    else:
        # Kinematic teleop has no physical response to contacts. Suppressing
        # contact generation avoids repeated detailed mesh intersections.
        for geom_id in robot_geom_ids:
            model.geom_contype[geom_id] = 0
            model.geom_conaffinity[geom_id] = 0
        contact_status = "Disabled collision checks for lightweight teleoperation"

    mujoco.mj_forward(model, data)
    root_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in robot_roots]
    print(f"Raised {len(robot_roots)} robot world roots by {ROBOT_BASE_LIFT:.3f} m.")
    print(f"Raised {moved_world_geoms} fixed-base robot geoms by the same amount.")
    print("Robot roots:", ", ".join(root_names))
    print(f"{contact_status} for {len(robot_geom_ids)} robot geoms.")
    world_geoms = [i for i in robot_geom_ids if model.geom_bodyid[i] == 0]
    return (robot_roots, model.body_pos[robot_roots].copy(), model.body_quat[robot_roots].copy(),
            world_geoms, model.geom_pos[world_geoms].copy(), model.geom_quat[world_geoms].copy())


def move_base(model, scene_state, x, y, yaw):
    roots, root_pos0, root_quat0, world_geoms, geom_pos0, geom_quat0 = scene_state
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[c, -s], [s, c]])
    yaw_quat = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
    def multiply(a, b):
        return np.array([a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3],
                         a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2],
                         a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1],
                         a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0]])
    for i, body_id in enumerate(roots):
        model.body_pos[body_id, :2] = rotation @ root_pos0[i, :2] + (x, y)
        model.body_pos[body_id, 2] = root_pos0[i, 2]
        model.body_quat[body_id] = multiply(yaw_quat, root_quat0[i])
    for i, geom_id in enumerate(world_geoms):
        model.geom_pos[geom_id, :2] = rotation @ geom_pos0[i, :2] + (x, y)
        model.geom_pos[geom_id, 2] = geom_pos0[i, 2]
        model.geom_quat[geom_id] = multiply(yaw_quat, geom_quat0[i])


def robot_body_ids(model: mujoco.MjModel):
    """Return all imported robot bodies, excluding injected scene objects."""
    scene_bodies = {
        "left_table", "right_table", "blue_cube", "dynamic_obstacle",
    }
    roots = {
        body_id for body_id in range(1, model.nbody)
        if model.body_parentid[body_id] == 0
        and mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) not in scene_bodies
    }
    result = []
    for body_id in range(1, model.nbody):
        ancestor = body_id
        while model.body_parentid[ancestor] != 0:
            ancestor = model.body_parentid[ancestor]
        if ancestor in roots:
            result.append(body_id)
    return np.asarray(result, dtype=np.int32)


def robot_geom_ids(model: mujoco.MjModel):
    """Identify robot geoms by excluding the injected environment tree."""
    scene_roots = {"left_table", "right_table", "blue_cube"}
    result = []
    for geom_id in range(model.ngeom):
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if geom_name == "floor":
            continue

        body_id = model.geom_bodyid[geom_id]
        root_body = body_id
        while root_body != 0 and model.body_parentid[root_body] != 0:
            root_body = model.body_parentid[root_body]
        root_name = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, root_body)
            if root_body != 0 else None
        )
        if root_name in scene_roots:
            continue
        result.append(geom_id)
    return np.asarray(result, dtype=np.int32)


def apply_visual_base_pose(data, body_ids, geom_ids, x, y, yaw):
    """Move only rendered robot transforms after normal forward kinematics.

    The converted URDF has multiple fixed roots. Updating model.body_pos for
    them during every keyboard event causes an invalid high-frequency model
    mutation. MuJoCo's renderer reads these derived data transforms, which
    lets the complete robot move visually without changing the model.
    """
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    translation = np.array([x, y, 0.0])

    data.xpos[body_ids] = data.xpos[body_ids] @ rotation.T + translation
    body_rotations = data.xmat[body_ids].reshape(-1, 3, 3).copy()
    data.xmat[body_ids] = (rotation @ body_rotations).reshape(-1, 9)

    data.geom_xpos[geom_ids] = data.geom_xpos[geom_ids] @ rotation.T + translation
    geom_rotations = data.geom_xmat[geom_ids].reshape(-1, 3, 3).copy()
    data.geom_xmat[geom_ids] = (rotation @ geom_rotations).reshape(-1, 9)


def set_target(model, target_qpos, jid, delta):
    if jid < 0:
        return
    qadr = model.jnt_qposadr[jid]
    value = target_qpos[qadr] + delta
    if model.jnt_limited[jid]:
        lo, hi = model.jnt_range[jid]
        value = float(np.clip(value, lo, hi))
    target_qpos[qadr] = value


def teleop_key_callback(window, key, scancode, action, mods):
    """Own the keyboard so viewer shortcuts cannot consume teleop keys."""
    del scancode, mods
    if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
        glfw.set_window_should_close(window, True)


def main():
    if not URDF_FILE.is_file():
        raise FileNotFoundError(f"URDF not found: {URDF_FILE}")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # Convert the URDF once. The generated MJCF already contains the robot.
    urdf_model = mujoco.MjModel.from_xml_path(str(URDF_FILE))
    robot_geom_count = urdf_model.ngeom
    mujoco.mj_saveLastXML(str(BASE_MJCF), urdf_model)
    mjcf_text = BASE_MJCF.read_text(encoding="utf-8")
    mjcf_text = insert_before_close(mjcf_text, "asset", ASSET_XML)
    mjcf_text = insert_before_close(mjcf_text, "worldbody", WORLD_XML)
    SCENE_MJCF.write_text(mjcf_text, encoding="utf-8")

    model = mujoco.MjModel.from_xml_path(str(SCENE_MJCF))
    data = mujoco.MjData(model)
    visual_geom_ids = robot_geom_ids(model)
    if not len(visual_geom_ids):
        raise RuntimeError("Could not identify imported robot geoms.")
    prepare_robot_scene(
        model, data, robot_geom_count, enable_contacts=TELEOP_ENABLE_CONTACTS,
        robot_geom_ids=visual_geom_ids,
    )
    model.opt.iterations = 100
    model.opt.ls_iterations = 50

    control = [(joint_id(model, name), label) for name, label in CONTROL_JOINTS]
    control = [(jid, label) for jid, label in control if jid >= 0]
    if not control:
        raise RuntimeError("None of the configured control joint names exist in the converted model.")

    missing = [name for name, _ in CONTROL_JOINTS if joint_id(model, name) < 0]
    if missing:
        print("Warning: missing joints:", ", ".join(missing))

    locked = [joint_id(model, name) for name in LOCKED_JOINTS]
    locked = [jid for jid in locked if jid >= 0]
    for jid, _ in control:
        model.dof_damping[model.jnt_dofadr[jid]] = 20.0

    left_fingers = [joint_id(model, name) for name in ("L_finger1_joint", "L_finger2_joint")]
    right_fingers = [joint_id(model, name) for name in ("R_finger1_joint", "R_finger2_joint")]
    left_fingers = [jid for jid in left_fingers if jid >= 0]
    right_fingers = [jid for jid in right_fingers if jid >= 0]

    # Preserve the imported URDF pose. Forcing the folding/lifter joints to
    # zero at t=0 can create an impulse large enough to destabilize the model.
    target_qpos = data.qpos.copy()
    visual_body_ids = robot_body_ids(model)
    if not len(visual_body_ids):
        raise RuntimeError("Could not identify imported robot bodies for visual teleoperation.")

    base_x = BASE_INITIAL_X
    base_y = BASE_INITIAL_Y
    base_yaw = BASE_INITIAL_YAW

    viewer = mujoco_viewer.MujocoViewer(model, data)
    # mujoco_viewer binds right-arrow to single-step and number keys to geom
    # visibility. Those bindings conflict with this script's base and arm
    # controls, so replace only its keyboard callback; mouse camera controls
    # remain untouched.
    glfw.set_key_callback(viewer.window, teleop_key_callback)
    viewer.cam.distance = 4.0
    viewer.cam.azimuth = 35.0
    viewer.cam.elevation = -22.0
    model.vis.headlight.active = 1
    model.vis.headlight.ambient = [0.30, 0.30, 0.30]
    model.vis.headlight.diffuse = [0.38, 0.38, 0.38]
    model.vis.headlight.specular = [0.03, 0.03, 0.03]
    head_camera = HeadCameraWindow(model, viewer.window)

    selected = 0
    recording = False
    frames = []
    episode_idx = 0
    previous = {key: False for key in "1290"}
    hold_count = {key: 0 for key in "345678"}

    arm_step = 0.05
    finger_step = 0.005
    hold_interval = 2
    kp, kd = 120.0, 24.0
    locked_kp, locked_kd = 800.0, 80.0
    last_control_time = time.perf_counter()
    last_render_time = last_control_time
    last_head_camera_time = last_control_time

    print(f"Loaded model: {model.njnt} joints, {model.nbody} bodies")
    print("Mode:", "kinematic teleoperation" if KINEMATIC_TELEOP else "physics")
    print("Arrows: up/down move, left/right turn")
    print("1/2 select joint, 3/4 move selected joint, 5/6 left gripper, 7/8 right gripper")
    print("9 toggle recording, 0 save recording, ESC close")

    while viewer.is_alive:
        frame_start = time.perf_counter()
        control_dt = min(frame_start - last_control_time, 0.05)
        last_control_time = frame_start
        glfw.poll_events()
        if glfw.window_should_close(viewer.window):
            break

        pressed = {
            key: glfw.get_key(viewer.window, getattr(glfw, f"KEY_{key}")) == glfw.PRESS
            for key in "1234567890"
        }
        forward = glfw.get_key(viewer.window, glfw.KEY_UP) == glfw.PRESS
        backward = glfw.get_key(viewer.window, glfw.KEY_DOWN) == glfw.PRESS
        turn_left = glfw.get_key(viewer.window, glfw.KEY_LEFT) == glfw.PRESS
        turn_right = glfw.get_key(viewer.window, glfw.KEY_RIGHT) == glfw.PRESS
        linear = BASE_SPEED * (float(forward) - float(backward))
        angular = TURN_SPEED * (float(turn_left) - float(turn_right))
        base_yaw += angular * control_dt
        # In this URDF, the longitudinal chassis direction is +X.
        base_x += linear * np.cos(base_yaw) * control_dt
        base_y += linear * np.sin(base_yaw) * control_dt

        if pressed["1"] and not previous["1"]:
            selected = (selected - 1) % len(control)
            print("Selected:", control[selected][1])
        if pressed["2"] and not previous["2"]:
            selected = (selected + 1) % len(control)
            print("Selected:", control[selected][1])
        if pressed["9"] and not previous["9"]:
            recording = not recording
            print("Recording:", recording)
        if pressed["0"] and not previous["0"] and frames:
            file_name = SAVE_DIR / f"ep_{episode_idx:03d}.npy"
            np.save(file_name, np.asarray(frames, dtype=np.float64))
            print(f"Saved {file_name} ({len(frames)} frames)")
            episode_idx += 1
            frames = []
            recording = False

        for key in "345678":
            if pressed[key]:
                hold_count[key] += 1
                if hold_count[key] < hold_interval:
                    continue
                hold_count[key] = 0
                if key == "3":
                    set_target(model, target_qpos, control[selected][0], arm_step)
                elif key == "4":
                    set_target(model, target_qpos, control[selected][0], -arm_step)
                elif key in ("5", "6"):
                    delta = -finger_step if key == "5" else finger_step
                    for jid in left_fingers:
                        set_target(model, target_qpos, jid, delta)
                elif key in ("7", "8"):
                    delta = -finger_step if key == "7" else finger_step
                    for jid in right_fingers:
                        set_target(model, target_qpos, jid, delta)
            else:
                hold_count[key] = 0

        for key in previous:
            previous[key] = pressed[key]

        if KINEMATIC_TELEOP:
            # The imported URDF is split into multiple independent roots, so
            # direct forward kinematics is the stable choice for collecting
            # pose demonstrations until a proper MJCF is built.
            data.qpos[:] = target_qpos
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            apply_visual_base_pose(
                data, visual_body_ids, visual_geom_ids, base_x, base_y, base_yaw
            )
        else:
            data.qfrc_applied[:] = 0.0
            for jid in locked:
                qadr, dadr = model.jnt_qposadr[jid], model.jnt_dofadr[jid]
                data.qfrc_applied[dadr] = locked_kp * (target_qpos[qadr] - data.qpos[qadr]) - locked_kd * data.qvel[dadr]
            for jid, _ in control:
                qadr, dadr = model.jnt_qposadr[jid], model.jnt_dofadr[jid]
                data.qfrc_applied[dadr] = kp * (target_qpos[qadr] - data.qpos[qadr]) - kd * data.qvel[dadr]
            mujoco.mj_step(model, data)

        if recording:
            frames.append(data.qpos.copy())
        # The complete robot uses detailed STL meshes. Updating the GLFW
        # scene at the 50 Hz control rate can stall VMware while the entire
        # base is moving. Keep control responsive but redraw only at 10 Hz.
        if frame_start - last_render_time >= TELEOP_RENDER_DT:
            # mujoco.Renderer used by the head-camera window owns a separate
            # OpenGL context. Restore the viewer context before drawing the
            # main scene; otherwise only the camera window keeps updating.
            glfw.make_context_current(viewer.window)
            viewer.render()
            last_render_time = frame_start
        if frame_start - last_head_camera_time >= HEAD_CAMERA_RENDER_DT:
            head_camera.render(data)
            glfw.make_context_current(viewer.window)
            last_head_camera_time = frame_start

        # Bound both CPU usage and frame rate. Movement above is based on the
        # measured elapsed time, so its physical speed stays stable.
        remaining = TELEOP_FRAME_DT - (time.perf_counter() - frame_start)
        if remaining > 0:
            time.sleep(remaining)

    head_camera.close()
    viewer.close()


if __name__ == "__main__":
    main()
