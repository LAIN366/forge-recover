"""Seeded fault scenarios for cooperative dual-arm experiments."""

from dataclasses import dataclass, field
from enum import Enum
import random


class DualArmFaultScenario(str, Enum):
    NONE = "none"
    LEFT_CONTACT_LOSS = "left_contact_loss"
    RIGHT_CONTACT_LOSS = "right_contact_loss"
    SYNCHRONIZATION_DELAY = "synchronization_delay"
    VISION_DROPOUT = "vision_dropout"
    DYNAMIC_OBSTACLE = "dynamic_obstacle"
    TRANSPORT_SLIP = "transport_slip"
    VISION_OCCLUSION = "vision_occlusion"
    TARGET_POSE_SHIFT = "target_pose_shift"
    SENSOR_DROPOUT = "sensor_dropout"
    LEFT_ARM_FAILURE = "left_arm_failure"
    RIGHT_ARM_FAILURE = "right_arm_failure"


@dataclass(frozen=True)
class DualArmFaultDirective:
    scenario: DualArmFaultScenario
    action: str
    parameters: dict = field(default_factory=dict)


class DualArmFaultInjector:
    def __init__(self, scenario="none", severity=1.0, seed=7):
        self.scenario = DualArmFaultScenario(scenario)
        self.severity = min(2.0, max(0.1, float(severity)))
        self.seed = int(seed)
        self.rng = random.Random(self.seed)
        self.injected = False

    def poll(self, stage, stage_elapsed):
        if self.injected or self.scenario == DualArmFaultScenario.NONE:
            return None
        directive = None
        if stage == "cooperative_lift" and stage_elapsed >= 0.10:
            if self.scenario == DualArmFaultScenario.LEFT_CONTACT_LOSS:
                directive = DualArmFaultDirective(
                    self.scenario, "disable_left_contact",
                    {"duration": 0.7 * self.severity},
                )
            elif self.scenario == DualArmFaultScenario.RIGHT_CONTACT_LOSS:
                directive = DualArmFaultDirective(
                    self.scenario, "disable_right_contact",
                    {"duration": 0.7 * self.severity},
                )
        elif stage == "cooperative_transport":
            if (
                self.scenario == DualArmFaultScenario.SYNCHRONIZATION_DELAY
                and stage_elapsed >= 0.10
            ):
                delayed_arm = self.rng.choice(("left", "right"))
                directive = DualArmFaultDirective(
                    self.scenario, "delay_arm",
                    {
                        "arm": delayed_arm,
                        "duration": self.rng.uniform(0.35, 0.60)
                        * self.severity,
                    },
                )
            elif (
                self.scenario == DualArmFaultScenario.DYNAMIC_OBSTACLE
                and stage_elapsed >= 0.002
            ):
                directive = DualArmFaultDirective(
                    self.scenario, "insert_obstacle",
                    {
                        "path_fraction": self.rng.uniform(0.25, 0.45),
                        "side": self.rng.choice(("left", "right")),
                    },
                )
            elif (
                self.scenario == DualArmFaultScenario.TRANSPORT_SLIP
                and stage_elapsed >= 0.12
            ):
                angle = self.rng.uniform(-3.141592653589793, 3.141592653589793)
                directive = DualArmFaultDirective(
                    self.scenario,
                    "release_object",
                    {
                        "duration": 1.0 * self.severity,
                        "angle": angle,
                        "lateral_speed": self.rng.uniform(0.035, 0.075)
                        * self.severity,
                        "angular_speed": 0.0,
                        "drop_offset_x": self.rng.uniform(-0.08, 0.08),
                        "drop_offset_y": self.rng.uniform(-0.10, 0.10),
                    },
                )
            elif (
                self.scenario == DualArmFaultScenario.VISION_OCCLUSION
                and stage_elapsed >= 0.08
            ):
                directive = DualArmFaultDirective(
                    self.scenario, "mask_vision",
                    {"duration": 0.6 * self.severity},
                )
            elif (
                self.scenario == DualArmFaultScenario.SENSOR_DROPOUT
                and stage_elapsed >= 0.08
            ):
                directive = DualArmFaultDirective(
                    self.scenario, "invalidate_sensors",
                    {"duration": 0.5 * self.severity},
                )
            elif (
                self.scenario in {
                    DualArmFaultScenario.LEFT_ARM_FAILURE,
                    DualArmFaultScenario.RIGHT_ARM_FAILURE,
                }
                and stage_elapsed >= 0.08
            ):
                side = (
                    "left"
                    if self.scenario == DualArmFaultScenario.LEFT_ARM_FAILURE
                    else "right"
                )
                directive = DualArmFaultDirective(
                    self.scenario, f"freeze_{side}_arm",
                    {"arm": side, "duration": 1.0 * self.severity},
                )
        elif (
            self.scenario == DualArmFaultScenario.VISION_DROPOUT
            and stage == "plan_dual_pregrasp"
            and stage_elapsed >= 0.10
        ):
            directive = DualArmFaultDirective(
                self.scenario, "mask_vision",
                {"duration": 0.8 * self.severity},
            )
        elif (
            self.scenario == DualArmFaultScenario.TARGET_POSE_SHIFT
            and stage == "plan_dual_pregrasp"
            and stage_elapsed >= 0.05
        ):
            directive = DualArmFaultDirective(
                self.scenario, "shift_target",
                {
                    "dx": self.rng.uniform(-0.05, 0.05) * self.severity,
                    "dy": self.rng.uniform(-0.05, 0.05) * self.severity,
                    "yaw": self.rng.uniform(-0.25, 0.25) * self.severity,
                },
            )
        if directive is not None:
            self.injected = True
        return directive
