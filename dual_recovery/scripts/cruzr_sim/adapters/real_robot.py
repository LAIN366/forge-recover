"""Safety-gated hardware boundary shared by simulation and Cruzr SDK drivers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
import time
from typing import Callable, Iterable


@dataclass(frozen=True)
class JointCommand:
    side: str
    positions: tuple[float, ...]
    duration: float
    source_node: str


@dataclass(frozen=True)
class RobotSafetyLimits:
    joint_lower: tuple[float, ...]
    joint_upper: tuple[float, ...]
    maximum_joint_step: float = 0.08
    minimum_duration: float = 0.05
    watchdog_timeout: float = 0.5


class RobotCommandPort(ABC):
    """Only this boundary may transmit a motion command to a real robot."""

    @abstractmethod
    def read_joint_positions(self, side):
        raise NotImplementedError

    @abstractmethod
    def send_joint_command(self, command):
        raise NotImplementedError

    @abstractmethod
    def stop(self, reason):
        raise NotImplementedError


class SafetyGatedCommandPort:
    def __init__(
        self,
        driver,
        limits,
        *,
        emergency_stop: Callable[[], bool] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.driver = driver
        self.limits = limits
        self._emergency_stop = emergency_stop or (lambda: False)
        self._clock = clock
        self._last_heartbeat: float | None = None
        self._estop_latched = False

    def heartbeat(self) -> None:
        """Record fresh control-loop telemetry using a monotonic clock."""
        self._last_heartbeat = self._clock()

    def _interlock_reason(self) -> str | None:
        self._estop_latched = self._estop_latched or bool(self._emergency_stop())
        if self._estop_latched:
            return "emergency stop is latched"
        if self._last_heartbeat is None:
            return "watchdog heartbeat is missing"
        if self._clock() - self._last_heartbeat > self.limits.watchdog_timeout:
            return "watchdog heartbeat expired"
        return None

    def validate(self, command):
        interlock = self._interlock_reason()
        if interlock:
            return False, interlock
        if command.side not in {"left", "right"}:
            return False, "invalid arm side"
        if command.duration < self.limits.minimum_duration:
            return False, "command duration below safety limit"
        if len(command.positions) != len(self.limits.joint_lower):
            return False, "joint vector length mismatch"
        if not all(math.isfinite(value) for value in command.positions):
            return False, "non-finite joint target"
        current = tuple(self.driver.read_joint_positions(command.side))
        if len(current) != len(command.positions):
            return False, "telemetry joint vector length mismatch"
        for target, now, lower, upper in zip(
            command.positions,
            current,
            self.limits.joint_lower,
            self.limits.joint_upper,
        ):
            if not lower <= target <= upper:
                return False, "joint limit violation"
            if abs(target - now) > self.limits.maximum_joint_step:
                return False, "joint step violation"
        return True, "validated"

    def validate_batch(self, commands: Iterable[JointCommand]):
        commands = tuple(commands)
        if not commands:
            return False, "empty command batch"
        if len({command.side for command in commands}) != len(commands):
            return False, "duplicate arm command"
        for command in commands:
            valid, reason = self.validate(command)
            if not valid:
                return False, reason
        return True, "validated"

    def execute_batch(self, commands, *, operator_enabled=False):
        commands = tuple(commands)
        if not operator_enabled:
            self.driver.stop("operator enable is false")
            return False, "operator enable is false"
        valid, reason = self.validate_batch(commands)
        if not valid:
            self.driver.stop(reason)
            return False, reason
        for command in commands:
            accepted = self.driver.send_joint_command(command)
            if accepted is not True:
                reason = f"SDK rejected {command.side} arm command"
                self.driver.stop(reason)
                return False, reason
        return True, "accepted"

    def execute(self, command, *, operator_enabled=False):
        accepted, _ = self.execute_batch((command,), operator_enabled=operator_enabled)
        return accepted


class DryRunRobotDriver:
    """Audit hardware commands without forwarding them to the vendor SDK."""

    def __init__(self, telemetry_driver):
        self.telemetry_driver = telemetry_driver
        self.commands: list[JointCommand] = []
        self.stop_reasons: list[str] = []

    def read_joint_positions(self, side):
        return self.telemetry_driver.read_joint_positions(side)

    def send_joint_command(self, command):
        self.commands.append(command)
        return True

    def stop(self, reason):
        self.stop_reasons.append(str(reason))
