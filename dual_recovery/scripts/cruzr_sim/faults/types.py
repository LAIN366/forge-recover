"""Fault injection contracts for reproducible simulation experiments."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class FaultScenario(str, Enum):
    NONE = "none"
    SLIP = "slip"
    MISSED_GRASP = "missed_grasp"
    TARGET_SHIFT = "target_shift"
    CONTACT_NOISE = "contact_noise"
    COLLISION_EVENT = "collision_event"
    IK_FAILURE = "ik_failure"
    PLANNING_FAILURE = "planning_failure"
    SENSOR_DROPOUT = "sensor_dropout"


@dataclass(frozen=True)
class FaultDirective:
    scenario: FaultScenario
    action: str
    parameters: dict[str, Any] = field(default_factory=dict)
