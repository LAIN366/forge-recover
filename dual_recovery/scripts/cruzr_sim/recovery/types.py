"""Typed recovery plans shared by simulation and robot adapters."""

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from cruzr_sim.diagnosis.types import FaultType


class RecoveryActionType(str, Enum):
    STOP = "stop"
    REOBSERVE = "reobserve"
    RETREAT = "retreat"
    OPEN_GRIPPER = "open_gripper"
    ADJUST_GRIP_FORCE = "adjust_grip_force"
    MOVE_BASE = "move_base"
    MOVE_TO_CLEARANCE = "move_to_clearance"
    RESELECT_GRASP = "reselect_grasp"
    REPLAN_ARM = "replan_arm"
    REGRASP = "regrasp"
    PLACE_BACK = "place_back"
    RESUME_TASK = "resume_task"
    SAFE_ABORT = "safe_abort"


@dataclass(frozen=True)
class RecoveryAction:
    action: RecoveryActionType
    parameters: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""


@dataclass(frozen=True)
class RecoveryPlan:
    plan_id: str
    fault: FaultType
    source: str
    confidence: float
    actions: tuple[RecoveryAction, ...]
    max_attempts: int = 2
    expected_outcome: str = "task can safely resume"

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["fault"] = self.fault.value
        for action in result["actions"]:
            action["action"] = action["action"].value
        return result


@dataclass(frozen=True)
class PlanValidation:
    valid: bool
    errors: tuple[str, ...] = ()
