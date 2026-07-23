"""Export structured JSONL traces as flat supervised-learning datasets."""

import csv
import json
from pathlib import Path


def _records(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def observation_rows(path):
    """Yield flat observation rows with the injected scenario as label."""
    path = Path(path)
    active_fault = "normal"
    for record in _records(path):
        payload = record.get("payload", {})
        if (
            record.get("record_type") == "event"
            and payload.get("name") == "fault_injected"
        ):
            active_fault = payload.get("scenario", "unknown")
            continue
        if record.get("record_type") != "observation":
            continue
        metadata = payload.get("metadata", {})
        object_position = payload.get("object_position", [0.0, 0.0, 0.0])
        tool_position = payload.get("tool_position", [0.0, 0.0, 0.0])
        visual_position = metadata.get("visual_position", [None, None, None])
        visual_rpy = metadata.get("visual_rpy", [None, None, None])
        visual_bbox = metadata.get("visual_bbox", [None, None, None, None])
        row = {
            "episode": path.stem,
            "timestamp": payload.get("timestamp", 0.0),
            "stage": payload.get("stage", "unknown"),
            "label": active_fault,
            "object_x": object_position[0],
            "object_y": object_position[1],
            "object_z": object_position[2],
            "tool_x": tool_position[0],
            "tool_y": tool_position[1],
            "tool_z": tool_position[2],
            "left_contact": int(bool(payload.get("left_contact"))),
            "right_contact": int(bool(payload.get("right_contact"))),
            "normal_force": payload.get("normal_force", 0.0),
            "tangent_force": payload.get("tangent_force", 0.0),
            "vertical_force": payload.get("vertical_force", 0.0),
            "object_vertical_velocity": payload.get("object_vertical_velocity", 0.0),
            "tracking_error": payload.get("tracking_error", 0.0),
            "ik_success": int(bool(payload.get("ik_success", True))),
            "planning_success": int(bool(payload.get("planning_success", True))),
            "collision": int(bool(payload.get("collision", False))),
            "sensor_valid": int(bool(payload.get("sensor_valid", True))),
            "visual_detected": int("visual_position" in metadata),
            "visual_confidence": metadata.get("visual_confidence"),
            "visual_x": visual_position[0],
            "visual_y": visual_position[1],
            "visual_z": visual_position[2],
            "visual_roll": visual_rpy[0],
            "visual_pitch": visual_rpy[1],
            "visual_yaw": visual_rpy[2],
            "visual_bbox_x1": visual_bbox[0],
            "visual_bbox_y1": visual_bbox[1],
            "visual_bbox_x2": visual_bbox[2],
            "visual_bbox_y2": visual_bbox[3],
        }
        for prefix in ("joint_position", "joint_velocity"):
            for index, value in enumerate(metadata.get(prefix, [])):
                row[f"{prefix}_{index}"] = value
        yield row


def export_dataset(input_paths, output_path):
    rows = []
    for path in input_paths:
        rows.extend(observation_rows(path))
    if not rows:
        raise ValueError("no observation records were found")
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with Path(output_path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)
