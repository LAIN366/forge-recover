"""Adapter from physical grasp runtime signals to diagnosis observations."""

import math

from cruzr_sim.diagnosis.types import ManipulationObservation


def build_grasp_observation(
    *, timestamp, stage, object_position, tool_position, contact_flags,
    contact_diagnostics, object_vertical_velocity, planned_object_position=None,
    tracking_error=0.0, collision=False, sensor_valid=True, event=None,
    sensor_packet=None, visual_detection=None, ik_success=True,
    planning_success=True,
):
    positions = tuple(float(value) for value in object_position)
    tool = tuple(float(value) for value in tool_position)
    target = (
        tuple(float(value) for value in planned_object_position)
        if planned_object_position is not None else None
    )
    finite_values = positions + tool + (
        float(object_vertical_velocity), float(tracking_error),
        float(contact_diagnostics.get("normal_force", 0.0)),
        float(contact_diagnostics.get("tangent_force", 0.0)),
    )
    valid = bool(sensor_valid and all(math.isfinite(value) for value in finite_values))
    return ManipulationObservation(
        timestamp=float(timestamp),
        stage=str(stage),
        object_position=positions,
        tool_position=tool,
        left_contact=bool(contact_flags[0]),
        right_contact=bool(contact_flags[1]),
        normal_force=float(contact_diagnostics.get("normal_force", 0.0)),
        tangent_force=float(contact_diagnostics.get("tangent_force", 0.0)),
        vertical_force=float(contact_diagnostics.get("vertical_force", 0.0)),
        object_vertical_velocity=float(object_vertical_velocity),
        tracking_error=float(tracking_error),
        target_position=target,
        collision=bool(collision),
        sensor_valid=valid,
        ik_success=bool(ik_success),
        planning_success=bool(planning_success),
        event=event,
        metadata={
            "contact_count": int(contact_diagnostics.get("contacts", 0)),
            "minimum_contact_distance": float(
                contact_diagnostics.get("minimum_distance", 0.0)
            ),
            **({
                "joint_position": sensor_packet.joint_position,
                "joint_velocity": sensor_packet.joint_velocity,
                "actuator_force": sensor_packet.actuator_force,
                "wrist_force": sensor_packet.wrist_force,
                "wrist_torque": sensor_packet.wrist_torque,
            } if sensor_packet is not None else {}),
            **({
                "visual_label": visual_detection.label,
                "visual_confidence": visual_detection.confidence,
                "visual_bbox": visual_detection.bbox,
                "visual_position": visual_detection.position,
                "visual_rpy": visual_detection.rpy,
            } if visual_detection is not None else {}),
        },
    )
