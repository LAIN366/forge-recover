"""Stage-triggered, seeded fault injector used by MuJoCo experiments."""

from dataclasses import dataclass
import math
import random

from .types import FaultDirective, FaultScenario


SLIP_MIN_HEIGHT_GAIN = 0.075
SLIP_MIN_LIFT_TIME = 2.0


@dataclass(frozen=True)
class FaultInjectionConfig:
    scenario: FaultScenario = FaultScenario.NONE
    severity: float = 1.0
    seed: int = 7


class FaultInjector:
    def __init__(self, config: FaultInjectionConfig | None = None):
        self.config = config or FaultInjectionConfig()
        self.rng = random.Random(self.config.seed)
        self.injected = False

    def reset(self) -> None:
        self.injected = False
        self.rng = random.Random(self.config.seed)

    def poll(self, stage: str, object_height_gain: float,
             stage_elapsed: float) -> FaultDirective | None:
        if self.injected or self.config.scenario == FaultScenario.NONE:
            return None
        severity = float(min(2.0, max(0.1, self.config.severity)))
        scenario = self.config.scenario
        directive = None
        if (
            scenario == FaultScenario.SLIP
            and stage == "lift"
            and object_height_gain >= SLIP_MIN_HEIGHT_GAIN
            and stage_elapsed >= SLIP_MIN_LIFT_TIME
        ):
            angle = self.rng.uniform(-math.pi, math.pi)
            lateral_speed = self.rng.uniform(0.24, 0.42) * severity
            angular_speed = self.rng.uniform(1.2, 2.8) * severity
            directive = FaultDirective(scenario, "release_grasp", {
                "velocity_x": float(math.cos(angle) * lateral_speed),
                "velocity_y": float(math.sin(angle) * lateral_speed),
                "angular_velocity": float(angular_speed),
            })
        elif scenario == FaultScenario.MISSED_GRASP and stage == "close" and stage_elapsed >= 0.15:
            directive = FaultDirective(scenario, "disable_gripper_contact")
        elif scenario == FaultScenario.TARGET_SHIFT and stage == "approach" and stage_elapsed >= 0.10:
            angle = self.rng.uniform(-math.pi, math.pi)
            distance = 0.055 * severity
            directive = FaultDirective(scenario, "shift_object", {
                "dx": float(math.cos(angle) * distance),
                "dy": float(math.sin(angle) * distance),
            })
        elif scenario == FaultScenario.CONTACT_NOISE and stage == "lift":
            directive = FaultDirective(
                scenario, "mask_contact_sensor", {"duration": 0.8 * severity}
            )
        elif (
            scenario == FaultScenario.COLLISION_EVENT
            and stage == "right_detour"
            and stage_elapsed >= 0.10
        ):
            directive = FaultDirective(
                scenario, "move_dynamic_obstacle", {
                    "duration": 0.4 * severity,
                    "path_fraction": self.rng.uniform(0.22, 0.32),
                }
            )
        elif scenario == FaultScenario.IK_FAILURE and stage == "pregrasp":
            directive = FaultDirective(
                scenario, "report_ik_failure", {"duration": 0.4 * severity}
            )
        elif scenario == FaultScenario.PLANNING_FAILURE and stage == "right_detour":
            directive = FaultDirective(
                scenario, "report_planning_failure", {"duration": 0.4 * severity}
            )
        elif scenario == FaultScenario.SENSOR_DROPOUT and stage == "approach":
            directive = FaultDirective(
                scenario, "invalidate_sensors", {"duration": 0.6 * severity}
            )
        if directive is not None:
            self.injected = True
        return directive
