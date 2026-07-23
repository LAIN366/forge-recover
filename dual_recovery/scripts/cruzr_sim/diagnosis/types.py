"""Shared data structures for manipulation monitoring and diagnosis."""

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class FaultType(str, Enum):
    NORMAL = "normal"
    MISSED_GRASP = "missed_grasp"
    GRASP_SLIP = "grasp_slip"
    UNSTABLE_GRASP = "unstable_grasp"
    TARGET_DISPLACEMENT = "target_displacement"
    COLLISION = "collision"
    IK_FAILURE = "ik_failure"
    PLANNING_FAILURE = "planning_failure"
    SENSOR_FAULT = "sensor_fault"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ManipulationObservation:
    """One synchronized observation from the manipulation system."""

    timestamp: float
    stage: str
    object_position: tuple[float, float, float]
    tool_position: tuple[float, float, float]
    left_contact: bool = False
    right_contact: bool = False
    normal_force: float = 0.0
    tangent_force: float = 0.0
    vertical_force: float = 0.0
    object_vertical_velocity: float = 0.0
    tracking_error: float = 0.0
    target_position: tuple[float, float, float] | None = None
    ik_success: bool = True
    planning_success: bool = True
    collision: bool = False
    sensor_valid: bool = True
    event: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def both_contacts(self) -> bool:
        return self.left_contact and self.right_contact

    @property
    def any_contact(self) -> bool:
        return self.left_contact or self.right_contact

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DiagnosticEvidence:
    fault: FaultType
    confidence: float
    reasons: tuple[str, ...]
    recommended_probes: tuple[str, ...] = ()


@dataclass(frozen=True)
class DiagnosisReport:
    timestamp: float
    stage: str
    anomalous: bool
    primary_fault: FaultType
    confidence: float
    hypotheses: tuple[DiagnosticEvidence, ...]
    needs_active_probe: bool = False
    selected_probe: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["primary_fault"] = self.primary_fault.value
        for item in result["hypotheses"]:
            item["fault"] = item["fault"].value
        return result
