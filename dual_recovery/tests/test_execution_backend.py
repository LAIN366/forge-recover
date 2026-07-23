"""Safety and dependency tests for the real-robot execution boundary."""

import ast
from pathlib import Path

import pytest

from cruzr_sim.adapters.cruzr_sdk import CruzrSdkBackend
from cruzr_sim.adapters.real_robot import (
    DryRunRobotDriver,
    RobotSafetyLimits,
    SafetyGatedCommandPort,
)
from cruzr_sim.simulation.execution_backend import MuJoCoDualArmBackend
from cruzr_sim.tasks.execution_backend import Pose6D


class FakeDriver:
    def __init__(self):
        self.joints = {"left": (0.0, 0.0), "right": (0.0, 0.0)}
        self.sent = []
        self.stops = []

    def read_joint_positions(self, side):
        return self.joints[side]

    def send_joint_command(self, command):
        self.sent.append(command)
        return True

    def stop(self, reason):
        self.stops.append(reason)


def make_backend():
    driver = FakeDriver()
    limits = RobotSafetyLimits(
        joint_lower=(-1.0, -1.0), joint_upper=(1.0, 1.0),
        maximum_joint_step=0.1,
    )
    port = SafetyGatedCommandPort(driver, limits)
    port.heartbeat()
    return CruzrSdkBackend(driver, port), driver


def test_operator_enable_is_required():
    backend, driver = make_backend()
    result = backend.execute_waypoint((0.01, 0.02), (0.01, 0.02), 0.1, source_node="lift")
    assert not result.accepted
    assert not driver.sent
    assert driver.stops == ["operator enable is false"]


def test_bilateral_command_is_validated_before_any_send():
    backend, driver = make_backend()
    backend.set_operator_enabled(True)
    result = backend.execute_waypoint((0.01, 0.02), (0.5, 0.02), 0.1, source_node="lift")
    assert not result.accepted
    assert result.reason == "joint step violation"
    assert not driver.sent
    assert driver.stops[-1] == "joint step violation"


def test_valid_bilateral_command_is_sent_in_order():
    backend, driver = make_backend()
    backend.set_operator_enabled(True)
    result = backend.execute_waypoint((0.01, 0.02), (-0.01, -0.02), 0.1, source_node="lift")
    assert result.accepted
    assert [command.side for command in driver.sent] == ["left", "right"]


def test_watchdog_rejects_motion_without_fresh_heartbeat():
    driver = FakeDriver()
    now = [1.0]
    limits = RobotSafetyLimits((-1.0, -1.0), (1.0, 1.0), watchdog_timeout=0.5)
    port = SafetyGatedCommandPort(driver, limits, clock=lambda: now[0])
    backend = CruzrSdkBackend(driver, port)
    backend.set_operator_enabled(True)
    result = backend.execute_waypoint((0.01, 0.02), (-0.01, -0.02), 0.1, source_node="lift")
    assert not result.accepted
    assert result.reason == "watchdog heartbeat is missing"

    port.heartbeat()
    now[0] = 1.6
    result = backend.execute_waypoint((0.01, 0.02), (-0.01, -0.02), 0.1, source_node="lift")
    assert not result.accepted
    assert result.reason == "watchdog heartbeat expired"


def test_emergency_stop_latches_until_port_is_recreated():
    driver = FakeDriver()
    estop = [False]
    limits = RobotSafetyLimits((-1.0, -1.0), (1.0, 1.0))
    port = SafetyGatedCommandPort(driver, limits, emergency_stop=lambda: estop[0])
    port.heartbeat()
    backend = CruzrSdkBackend(driver, port)
    backend.set_operator_enabled(True)
    estop[0] = True
    result = backend.execute_waypoint((0.01, 0.02), (-0.01, -0.02), 0.1, source_node="lift")
    assert not result.accepted
    estop[0] = False
    port.heartbeat()
    result = backend.execute_waypoint((0.01, 0.02), (-0.01, -0.02), 0.1, source_node="lift")
    assert not result.accepted
    assert result.reason == "emergency stop is latched"


def test_dry_run_records_validated_commands_without_using_sdk_motion():
    telemetry = FakeDriver()
    dry_run = DryRunRobotDriver(telemetry)
    limits = RobotSafetyLimits((-1.0, -1.0), (1.0, 1.0))
    port = SafetyGatedCommandPort(dry_run, limits)
    port.heartbeat()
    backend = CruzrSdkBackend(telemetry, port)
    backend.set_operator_enabled(True)
    result = backend.execute_waypoint((0.01, 0.02), (-0.01, -0.02), 0.1, source_node="lift")
    assert result.accepted
    assert not telemetry.sent
    assert [command.side for command in dry_run.commands] == ["left", "right"]


def test_sdk_rejection_stops_second_arm_transmission():
    class RejectingDriver(FakeDriver):
        def send_joint_command(self, command):
            self.sent.append(command)
            return command.side != "left"

    driver = RejectingDriver()
    limits = RobotSafetyLimits((-1.0, -1.0), (1.0, 1.0))
    port = SafetyGatedCommandPort(driver, limits)
    port.heartbeat()
    backend = CruzrSdkBackend(driver, port)
    backend.set_operator_enabled(True)
    result = backend.execute_waypoint((0.01, 0.02), (-0.01, -0.02), 0.1, source_node="lift")
    assert not result.accepted
    assert result.reason == "SDK rejected left arm command"
    assert [command.side for command in driver.sent] == ["left"]


def test_task_backend_contract_does_not_import_mujoco():
    source = Path("scripts/cruzr_sim/tasks/execution_backend.py").read_text(encoding="utf-8")
    imports = [node for node in ast.walk(ast.parse(source)) if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert all("mujoco" not in ast.unparse(node).lower() for node in imports)


def test_mujoco_backend_converts_identity_rpy_to_identity_quaternion():
    assert MuJoCoDualArmBackend._rpy_to_quaternion((0.0, 0.0, 0.0)) == (
        1.0, 0.0, 0.0, 0.0,
    )


def test_portable_pose_preserves_rpy_through_quaternion_boundary():
    expected = (0.2, -0.1, 0.4)
    quaternion = MuJoCoDualArmBackend._rpy_to_quaternion(expected)
    actual = Pose6D((0.0, 0.0, 0.0), quaternion).rpy
    assert actual == pytest.approx(expected)


def test_mujoco_backend_owns_waist_search_camera_configuration():
    backend = object.__new__(MuJoCoDualArmBackend)
    backend.camera = type("Camera", (), {
        "views": {"waist": {"pitch_down": 0.0}},
        "active_view": "head",
    })()
    result = backend.set_camera_view("waist_search")
    assert result.accepted
    assert backend.camera.active_view == "waist"
    assert backend.camera.views["waist"]["pitch_down"] == 35.0


def test_mujoco_backend_delegates_both_grippers():
    class Gripper:
        def __init__(self):
            self.state = None

        def open(self):
            self.state = "open"

        def close(self):
            self.state = "closed"

    backend = object.__new__(MuJoCoDualArmBackend)
    backend.left_gripper = Gripper()
    backend.right_gripper = Gripper()
    result = backend.set_grippers(True, False)
    assert result.accepted
    assert backend.left_gripper.state == "closed"
    assert backend.right_gripper.state == "open"
