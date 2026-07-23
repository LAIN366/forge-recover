"""Shared Cruzr S2 URDF-to-MuJoCo scene utilities."""

from pathlib import Path

import mujoco
import numpy as np
import xml.etree.ElementTree as ET


PROJECT_ROOT = Path(__file__).resolve().parents[3]
URDF_FILE = PROJECT_ROOT / "robots" / "cruzr_s2_v1.urdf"
BASE_MJCF = Path("/tmp/cruzr_s2_base.xml")
ROBOT_BASE_LIFT = 0.14
BASE_INITIAL_X = -1.25
BASE_INITIAL_Y = 0.0
BASE_INITIAL_YAW = np.pi / 2

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
<material name="cruzr_silver" rgba="0.72 0.74 0.78 1"
          specular="0.28" shininess="0.45" metallic="0.18" emission="0.12"/>
<material name="cruzr_base_metal" rgba="0.50 0.53 0.58 1"
          specular="0.24" shininess="0.38" metallic="0.16" emission="0.06"/>
<material name="cruzr_joint_gray" rgba="0.16 0.18 0.21 1"
          specular="0.16" shininess="0.28" emission="0.02"/>
<material name="cruzr_status_light" rgba="0.62 0.92 1.00 1"
          emission="0.45" specular="0.55" shininess="0.80"/>
"""

WORLD_XML = """
<light name="key_light" pos="0 -2 5.5" dir="0 0 -1"
       directional="true" diffuse="0.34 0.34 0.34" specular="0.03 0.03 0.03"/>
<light name="fill_light" pos="-3 2 3" dir="1 -0.5 -0.6"
       directional="true" diffuse="0.20 0.23 0.28"/>
<light name="rim_light" pos="3 1 4" dir="-1 -0.3 -0.8"
       directional="true" diffuse="0.12 0.14 0.18"/>
<geom name="floor" type="plane" size="500 500 0.1" material="checker_floor"
      friction="1 0.005 0.0001"/>
<camera name="overview" pos="4.8 -5.6 3.7"
        xyaxes="0.76 0.65 0 -0.30 0.35 0.89" fovy="48"/>
<body name="left_table" pos="-1.25 0.57 0">
  <geom name="left_table_top" type="box" size="0.40 0.30 0.02"
        pos="0 0 0.75" material="table_top"/>
  <geom type="box" size="0.025 0.025 0.375" pos="0.34 0.24 0.375"
        material="table_leg"/>
  <geom type="box" size="0.025 0.025 0.375" pos="0.34 -0.24 0.375"
        material="table_leg"/>
  <geom type="box" size="0.025 0.025 0.375" pos="-0.34 0.24 0.375"
        material="table_leg"/>
  <geom type="box" size="0.025 0.025 0.375" pos="-0.34 -0.24 0.375"
        material="table_leg"/>
</body>
"""


ROBOT_VISUAL_GEOM_COUNT = 1


def add_cruzr_s2_visual_details(xml_text):
    """Add the documented, non-colliding base status light."""
    root = ET.fromstring(xml_text)
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("converted robot MJCF has no worldbody")
    ET.SubElement(worldbody, "geom", {
        "name": "cruzr_base_status_light",
        "type": "capsule",
        "fromto": "-0.334 -0.132 0.116 -0.334 0.132 0.116",
        "size": "0.004",
        "material": "cruzr_status_light",
        "contype": "0",
        "conaffinity": "0",
        "group": "1",
        "mass": "0",
    })
    return ET.tostring(root, encoding="unicode")


def apply_cruzr_s2_appearance(model, robot_geom_count):
    """Apply body-only materials while leaving all scene geoms untouched."""
    material_ids = {
        name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, name)
        for name in (
            "cruzr_silver", "cruzr_base_metal", "cruzr_joint_gray",
            "cruzr_status_light",
        )
    }
    if min(material_ids.values()) < 0:
        raise RuntimeError("Cruzr appearance materials are missing from MJCF")

    for geom_id in range(int(robot_geom_count)):
        geom_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
        ) or ""
        if geom_name == "cruzr_base_status_light":
            model.geom_matid[geom_id] = material_ids["cruzr_status_light"]
            continue
        mesh_id = int(model.geom_dataid[geom_id])
        if int(model.geom_group[geom_id]) != 1:
            # URDF conversion emits a coincident collision mesh for each
            # visual mesh. Scene primitives have no mesh and must stay visible.
            if mesh_id >= 0:
                model.geom_rgba[geom_id, 3] = 0.0
            continue

        body_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id])
        ) or "world"
        mesh_name = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id) or ""
            if mesh_id >= 0 else ""
        )
        name = f"{body_name} {mesh_name}".lower()
        if mesh_name.lower() == "base_link":
            material = "cruzr_base_metal"
        elif any(token in name for token in (
            "wheel", "protection_", "lifter_", "waist_yaw",
            "head_yaw", "shoulder_roll", "elbow_roll", "wrist_pitch",
            "wrist_roll", "sixforce",
        )):
            material = "cruzr_joint_gray"
        else:
            material = "cruzr_silver"
        model.geom_matid[geom_id] = material_ids[material]


def insert_before_close(xml_text: str, tag: str, content: str) -> str:
    closing_tag = f"</{tag}>"
    if closing_tag not in xml_text:
        raise RuntimeError(f"converted MJCF has no {closing_tag} section")
    return xml_text.replace(closing_tag, f"{content}\n{closing_tag}", 1)


def prepare_robot_scene(model, data, robot_geom_count):
    """Raise all fixed URDF roots and configure robot/environment contacts."""
    scene_bodies = {
        "left_table", "right_table", "blue_cube", "dynamic_obstacle",
    }
    robot_roots = [
        body_id for body_id in range(1, model.nbody)
        if model.body_parentid[body_id] == 0
        and mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, body_id
        ) not in scene_bodies
    ]
    if not robot_roots:
        raise RuntimeError("could not identify robot bodies under worldbody")

    for body_id in robot_roots:
        model.body_pos[body_id, 2] += ROBOT_BASE_LIFT
    robot_geom_ids = np.arange(robot_geom_count, dtype=np.int32)
    for geom_id in robot_geom_ids:
        if model.geom_bodyid[geom_id] == 0:
            model.geom_pos[geom_id, 2] += ROBOT_BASE_LIFT
    mujoco.mj_setConst(model, data)

    for geom_id in robot_geom_ids:
        model.geom_contype[geom_id] = 1
        model.geom_conaffinity[geom_id] = 2
    robot_geom_set = set(robot_geom_ids.tolist())
    for geom_id in range(model.ngeom):
        if geom_id not in robot_geom_set:
            model.geom_contype[geom_id] = 2
            model.geom_conaffinity[geom_id] = 1
    mujoco.mj_forward(model, data)

    world_geoms = [
        geom_id for geom_id in robot_geom_ids
        if model.geom_bodyid[geom_id] == 0
    ]
    return (
        robot_roots,
        model.body_pos[robot_roots].copy(),
        model.body_quat[robot_roots].copy(),
        world_geoms,
        model.geom_pos[world_geoms].copy(),
        model.geom_quat[world_geoms].copy(),
    )


def _multiply_quaternions(a, b):
    return np.array([
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ])


def move_base(model, scene_state, x, y, yaw):
    roots, root_pos, root_quat, world_geoms, geom_pos, geom_quat = scene_state
    cosine, sine = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    yaw_quaternion = np.array([
        np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)
    ])
    for index, body_id in enumerate(roots):
        model.body_pos[body_id, :2] = (
            rotation @ root_pos[index, :2] + (x, y)
        )
        model.body_pos[body_id, 2] = root_pos[index, 2]
        model.body_quat[body_id] = _multiply_quaternions(
            yaw_quaternion, root_quat[index]
        )
    for index, geom_id in enumerate(world_geoms):
        model.geom_pos[geom_id, :2] = (
            rotation @ geom_pos[index, :2] + (x, y)
        )
        model.geom_pos[geom_id, 2] = geom_pos[index, 2]
        model.geom_quat[geom_id] = _multiply_quaternions(
            yaw_quaternion, geom_quat[index]
        )
