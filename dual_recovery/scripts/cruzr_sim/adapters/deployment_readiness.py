"""Fail-closed validation for Cruzr S2 deployment configuration and records."""

from dataclasses import dataclass
import json
import math
from pathlib import Path


PRE_MOTION_STAGES = (
    "dry_run_logging",
    "simulator_replay",
    "motors_off_telemetry",
)


@dataclass(frozen=True)
class ReadinessReport:
    ready_for_first_motion: bool
    errors: tuple[str, ...]


def _finite_vector(value, length, label, errors):
    if not isinstance(value, list) or len(value) != length:
        errors.append(f"{label} must contain {length} values")
        return
    if not all(isinstance(item, (int, float)) and math.isfinite(item) for item in value):
        errors.append(f"{label} must contain only finite numbers")


def validate_deployment_config(payload) -> tuple[str, ...]:
    errors = []
    if payload.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if payload.get("real_motion_enabled") is not False:
        errors.append("real_motion_enabled must remain false during readiness checks")
    if payload.get("calibrated") is not True:
        errors.append("calibration has not been signed off")

    transforms = payload.get("transforms", {})
    for name in ("base_to_left_arm", "base_to_right_arm", "base_to_head_camera"):
        transform = transforms.get(name, {})
        _finite_vector(transform.get("translation_m"), 3, f"{name}.translation_m", errors)
        quaternion = transform.get("quaternion_wxyz")
        _finite_vector(quaternion, 4, f"{name}.quaternion_wxyz", errors)
        if isinstance(quaternion, list) and len(quaternion) == 4 and all(
            isinstance(item, (int, float)) and math.isfinite(item) for item in quaternion
        ):
            norm = math.sqrt(sum(float(item) ** 2 for item in quaternion))
            if abs(norm - 1.0) > 1e-3:
                errors.append(f"{name}.quaternion_wxyz must be normalized")

    arms = payload.get("arms", {})
    for side in ("left", "right"):
        arm = arms.get(side, {})
        names = arm.get("joint_names")
        lower = arm.get("joint_lower_rad")
        upper = arm.get("joint_upper_rad")
        if not isinstance(names, list) or not names or not all(isinstance(x, str) and x for x in names):
            errors.append(f"arms.{side}.joint_names must be a non-empty string list")
            continue
        _finite_vector(lower, len(names), f"arms.{side}.joint_lower_rad", errors)
        _finite_vector(upper, len(names), f"arms.{side}.joint_upper_rad", errors)
        if isinstance(lower, list) and isinstance(upper, list) and len(lower) == len(upper) == len(names):
            if any(float(lo) >= float(hi) for lo, hi in zip(lower, upper)):
                errors.append(f"arms.{side} contains invalid joint limits")

    interfaces = payload.get("interfaces", {})
    required = ("joint_telemetry", "joint_command", "emergency_stop", "rgbd_stream")
    for name in required:
        if not isinstance(interfaces.get(name), str) or not interfaces[name].strip():
            errors.append(f"interfaces.{name} mapping is required")
    return tuple(errors)


def check_deployment_readiness(config_path, acceptance_path) -> ReadinessReport:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    acceptance = json.loads(Path(acceptance_path).read_text(encoding="utf-8"))
    errors = list(validate_deployment_config(config))
    records = acceptance.get("stages", {})
    for stage in PRE_MOTION_STAGES:
        record = records.get(stage, {})
        if record.get("passed") is not True:
            errors.append(f"stage {stage} has not passed")
        if not str(record.get("signed_by", "")).strip():
            errors.append(f"stage {stage} lacks operator sign-off")
        if not str(record.get("evidence", "")).strip():
            errors.append(f"stage {stage} lacks evidence")
    return ReadinessReport(not errors, tuple(errors))
