import json

from cruzr_sim.adapters.deployment_readiness import (
    check_deployment_readiness,
    validate_deployment_config,
)


def valid_config():
    transform = {"translation_m": [0.0, 0.0, 0.0], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]}
    arm = {"joint_names": ["j1", "j2"], "joint_lower_rad": [-1.0, -1.0], "joint_upper_rad": [1.0, 1.0]}
    return {
        "schema_version": 1,
        "real_motion_enabled": False,
        "calibrated": True,
        "transforms": {
            "base_to_left_arm": transform,
            "base_to_right_arm": transform,
            "base_to_head_camera": transform,
        },
        "arms": {"left": arm, "right": arm},
        "interfaces": {
            "joint_telemetry": "sdk.read_joints",
            "joint_command": "sdk.command_joints",
            "emergency_stop": "gpio.estop",
            "rgbd_stream": "/camera/rgbd",
        },
    }


def test_template_state_is_rejected():
    payload = valid_config()
    payload["calibrated"] = False
    payload["real_motion_enabled"] = True
    errors = validate_deployment_config(payload)
    assert "calibration has not been signed off" in errors
    assert "real_motion_enabled must remain false during readiness checks" in errors


def test_non_normalized_transform_is_rejected():
    payload = valid_config()
    payload["transforms"]["base_to_head_camera"] = {
        "translation_m": [0.0, 0.0, 0.0],
        "quaternion_wxyz": [2.0, 0.0, 0.0, 0.0],
    }
    assert any("must be normalized" in error for error in validate_deployment_config(payload))


def test_readiness_requires_all_pre_motion_evidence(tmp_path):
    config_path = tmp_path / "config.json"
    acceptance_path = tmp_path / "acceptance.json"
    config_path.write_text(json.dumps(valid_config()), encoding="utf-8")
    acceptance = {
        "stages": {
            name: {"passed": True, "signed_by": "operator", "evidence": f"results/{name}.json"}
            for name in ("dry_run_logging", "simulator_replay", "motors_off_telemetry")
        }
    }
    acceptance_path.write_text(json.dumps(acceptance), encoding="utf-8")
    assert check_deployment_readiness(config_path, acceptance_path).ready_for_first_motion
    acceptance["stages"]["motors_off_telemetry"]["evidence"] = ""
    acceptance_path.write_text(json.dumps(acceptance), encoding="utf-8")
    report = check_deployment_readiness(config_path, acceptance_path)
    assert not report.ready_for_first_motion
    assert "stage motors_off_telemetry lacks evidence" in report.errors
