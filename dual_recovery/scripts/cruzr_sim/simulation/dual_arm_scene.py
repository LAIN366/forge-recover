"""Build the MuJoCo scene used by the cooperative transport task."""

from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco

from .robot_model import (
    ASSET_XML,
    BASE_INITIAL_X,
    BASE_INITIAL_Y,
    BASE_INITIAL_YAW,
    BASE_MJCF,
    URDF_FILE,
    WORLD_XML,
    ROBOT_VISUAL_GEOM_COUNT,
    add_cruzr_s2_visual_details,
    apply_cruzr_s2_appearance,
    insert_before_close,
    move_base,
    prepare_robot_scene,
)
from cruzr_sim.control.dynamic_gripper import (
    build_gripper_model_xml,
    build_gripper_worldbody_xml,
)


COLLISION_OBJECT = 2
COLLISION_GRIPPER = 4
COLLISION_ENVIRONMENT = 8
COLLISION_LEFT_ARM = 16
COLLISION_RIGHT_ARM = 32

DUAL_SCENE_MJCF = "/tmp/cruzr_s2_dual_arm_transport.xml"
SCENE_DEFINITION = (
    Path(__file__).resolve().parents[1]
    / "scenes"
    / "dual_arm_transport_scene.xml"
)
LEFT_GRIPPER_PREFIX = "dual_left"
RIGHT_GRIPPER_PREFIX = "dual_right"
REPLACED_FINGER_BODIES = {
    "L_finger1_link", "L_finger2_link",
    "R_finger1_link", "R_finger2_link",
}

def mujoco_id(model, object_type, name):
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise ValueError(f"MuJoCo object not found: {name}")
    return object_id


def _task_world_xml(workpiece=None):
    scene_root = ET.parse(SCENE_DEFINITION).getroot()
    scene_worldbody = scene_root.find("worldbody")
    if scene_worldbody is None:
        raise RuntimeError(f"scene has no worldbody: {SCENE_DEFINITION}")
    if workpiece is not None:
        body = scene_worldbody.find("body[@name='blue_cube']")
        geom = body.find("geom[@name='blue_cube_geom']")
        ballast = body.find("geom[@name='blue_cube_ballast']")
        body.set("pos", " ".join(f"{value:.6f}" for value in workpiece.position))
        body.set("quat", " ".join(f"{value:.8f}" for value in workpiece.quaternion))
        geom.set("size", " ".join(f"{value:.6f}" for value in workpiece.half_size))
        geom.set("mass", f"{0.60 * workpiece.mass:.6f}")
        geom.set("friction", f"{workpiece.friction:.6f} 0.05 0.02")
        ballast_radius = 0.35 * min(workpiece.half_size[1:])
        ballast.set("pos", f"0 0 {-0.70 * workpiece.half_size[2]:.6f}")
        ballast.set("size", f"{ballast_radius:.6f}")
        ballast.set("mass", f"{0.40 * workpiece.mass:.6f}")
    scene_xml = "\n".join(
        ET.tostring(element, encoding="unicode")
        for element in scene_worldbody
    )
    return WORLD_XML + scene_xml


def build_dual_arm_scene(workpiece=None):
    urdf_model = mujoco.MjModel.from_xml_path(str(URDF_FILE))
    robot_geom_count = urdf_model.ngeom
    mujoco.mj_saveLastXML(str(BASE_MJCF), urdf_model)
    text = BASE_MJCF.read_text(encoding="utf-8")
    text = insert_before_close(text, "asset", ASSET_XML)
    text = add_cruzr_s2_visual_details(text)
    robot_geom_count += ROBOT_VISUAL_GEOM_COUNT
    gripper_bodies = (
        build_gripper_worldbody_xml(LEFT_GRIPPER_PREFIX)
        + build_gripper_worldbody_xml(RIGHT_GRIPPER_PREFIX)
    )
    text = insert_before_close(
        text, "worldbody", _task_world_xml(workpiece) + gripper_bodies
    )
    gripper_models = (
        build_gripper_model_xml(LEFT_GRIPPER_PREFIX)
        + build_gripper_model_xml(RIGHT_GRIPPER_PREFIX)
    )
    text = text.replace("</mujoco>", gripper_models + "\n</mujoco>", 1)
    Path(DUAL_SCENE_MJCF).write_text(text, encoding="utf-8")

    model = mujoco.MjModel.from_xml_path(DUAL_SCENE_MJCF)
    data = mujoco.MjData(model)
    apply_cruzr_s2_appearance(model, robot_geom_count)
    scene_state = prepare_robot_scene(model, data, robot_geom_count)
    move_base(model, scene_state, BASE_INITIAL_X, BASE_INITIAL_Y, BASE_INITIAL_YAW)

    object_geom = mujoco_id(model, mujoco.mjtObj.mjOBJ_GEOM, "blue_cube_geom")
    left_arm_geoms = []
    right_arm_geoms = []
    environment_geoms = []
    gripper_geoms = []
    for geom_id in range(model.ngeom):
        geom_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
        ) or ""
        body_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[geom_id]
        ) or ""
        if geom_name == "blue_cube_ballast":
            continue
        if geom_name.startswith((LEFT_GRIPPER_PREFIX, RIGHT_GRIPPER_PREFIX)):
            gripper_geoms.append(geom_id)
        elif geom_id >= robot_geom_count and geom_id != object_geom:
            if geom_name != "transport_goal":
                environment_geoms.append(geom_id)
        elif body_name in REPLACED_FINGER_BODIES:
            continue
        elif body_name.startswith("L_"):
            left_arm_geoms.append(geom_id)
        elif body_name.startswith("R_"):
            right_arm_geoms.append(geom_id)

    if not left_arm_geoms or not right_arm_geoms:
        raise RuntimeError("could not identify both Cruzr arm collision groups")

    for geom_id in range(model.ngeom):
        model.geom_contype[geom_id] = 0
        model.geom_conaffinity[geom_id] = 0
    for geom_id in left_arm_geoms:
        model.geom_contype[geom_id] = COLLISION_LEFT_ARM
        model.geom_conaffinity[geom_id] = (
            COLLISION_OBJECT | COLLISION_ENVIRONMENT | COLLISION_RIGHT_ARM
        )
    for geom_id in right_arm_geoms:
        model.geom_contype[geom_id] = COLLISION_RIGHT_ARM
        model.geom_conaffinity[geom_id] = (
            COLLISION_OBJECT | COLLISION_ENVIRONMENT | COLLISION_LEFT_ARM
        )
    for geom_id in gripper_geoms:
        model.geom_contype[geom_id] = COLLISION_GRIPPER
        model.geom_conaffinity[geom_id] = COLLISION_OBJECT | COLLISION_ENVIRONMENT
    model.geom_contype[object_geom] = COLLISION_OBJECT
    model.geom_conaffinity[object_geom] = (
        COLLISION_GRIPPER | COLLISION_ENVIRONMENT
        | COLLISION_LEFT_ARM | COLLISION_RIGHT_ARM
    )
    for geom_id in environment_geoms:
        model.geom_contype[geom_id] = COLLISION_ENVIRONMENT
        model.geom_conaffinity[geom_id] = (
            COLLISION_OBJECT | COLLISION_GRIPPER
            | COLLISION_LEFT_ARM | COLLISION_RIGHT_ARM
        )

    model.opt.timestep = 0.002
    model.opt.iterations = 120
    model.opt.noslip_iterations = 12
    mujoco.mj_forward(model, data)
    return (
        model, data, object_geom,
        left_arm_geoms, right_arm_geoms, environment_geoms,
    )
