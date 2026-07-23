"""Small, inspectable recovery case base used for retrieval-augmented planning."""

from dataclasses import dataclass

from cruzr_sim.diagnosis.types import FaultType

from .types import RecoveryAction, RecoveryActionType


@dataclass(frozen=True)
class RecoveryCase:
    case_id: str
    fault: FaultType
    applicable_stages: tuple[str, ...]
    actions: tuple[RecoveryAction, ...]
    tags: tuple[str, ...] = ()
    success_rate: float = 0.5


DEFAULT_CASES = (
    RecoveryCase(
        "missed-grasp-reobserve",
        FaultType.MISSED_GRASP,
        ("close", "lift"),
        (
            RecoveryAction(RecoveryActionType.OPEN_GRIPPER, rationale="release partial contact"),
            RecoveryAction(RecoveryActionType.RETREAT, {"distance": 0.12}),
            RecoveryAction(RecoveryActionType.REOBSERVE, {"target": "object"}),
            RecoveryAction(RecoveryActionType.RESELECT_GRASP),
            RecoveryAction(RecoveryActionType.REPLAN_ARM),
            RecoveryAction(RecoveryActionType.REGRASP),
            RecoveryAction(RecoveryActionType.RESUME_TASK),
        ),
        ("vision", "grasp"),
        0.82,
    ),
    RecoveryCase(
        "slip-place-and-regrasp",
        FaultType.GRASP_SLIP,
        ("lift", "hold", "transport", "fault_slip"),
        (
            RecoveryAction(RecoveryActionType.STOP),
            RecoveryAction(RecoveryActionType.PLACE_BACK, {"surface": "nearest_safe"}),
            RecoveryAction(RecoveryActionType.OPEN_GRIPPER),
            RecoveryAction(RecoveryActionType.MOVE_TO_CLEARANCE),
            RecoveryAction(RecoveryActionType.REOBSERVE, {"target": "object"}),
            RecoveryAction(RecoveryActionType.ADJUST_GRIP_FORCE, {"scale": 1.15}),
            RecoveryAction(RecoveryActionType.REPLAN_ARM),
            RecoveryAction(RecoveryActionType.REGRASP),
            RecoveryAction(RecoveryActionType.RESUME_TASK),
        ),
        ("force", "contact", "grasp"),
        0.86,
    ),
    RecoveryCase(
        "unstable-grasp-reset",
        FaultType.UNSTABLE_GRASP,
        ("close", "lift", "hold"),
        (
            RecoveryAction(RecoveryActionType.STOP),
            RecoveryAction(RecoveryActionType.PLACE_BACK, {"surface": "source"}),
            RecoveryAction(RecoveryActionType.OPEN_GRIPPER),
            RecoveryAction(RecoveryActionType.RESELECT_GRASP, {"prefer_centered": True}),
            RecoveryAction(RecoveryActionType.REGRASP),
            RecoveryAction(RecoveryActionType.RESUME_TASK),
        ),
        ("asymmetric_contact", "force"),
        0.79,
    ),
    RecoveryCase(
        "target-moved-replan",
        FaultType.TARGET_DISPLACEMENT,
        ("right_detour", "pregrasp", "approach", "close"),
        (
            RecoveryAction(RecoveryActionType.STOP),
            RecoveryAction(RecoveryActionType.RETREAT, {"distance": 0.10}),
            RecoveryAction(RecoveryActionType.REOBSERVE, {"target": "object"}),
            RecoveryAction(RecoveryActionType.REPLAN_ARM),
            RecoveryAction(RecoveryActionType.RESUME_TASK, {"stage": "pregrasp"}),
        ),
        ("vision", "dynamic_target"),
        0.88,
    ),
    RecoveryCase(
        "collision-retreat-replan",
        FaultType.COLLISION,
        ("clearance", "right_detour", "pregrasp", "approach", "transport"),
        (
            RecoveryAction(RecoveryActionType.STOP),
            RecoveryAction(RecoveryActionType.RETREAT, {"distance": 0.08}),
            RecoveryAction(RecoveryActionType.REOBSERVE, {"target": "obstacles"}),
            RecoveryAction(RecoveryActionType.REPLAN_ARM, {"avoid_previous_path": True}),
            RecoveryAction(RecoveryActionType.RESUME_TASK),
        ),
        ("collision", "safety"),
        0.74,
    ),
    RecoveryCase(
        "ik-base-reconfiguration",
        FaultType.IK_FAILURE,
        ("clearance", "right_detour", "pregrasp", "approach"),
        (
            RecoveryAction(RecoveryActionType.REOBSERVE, {"target": "object"}),
            RecoveryAction(RecoveryActionType.MOVE_BASE, {"strategy": "increase_reachability"}),
            RecoveryAction(RecoveryActionType.REPLAN_ARM),
            RecoveryAction(RecoveryActionType.RESUME_TASK),
        ),
        ("reachability", "mobile_manipulator"),
        0.72,
    ),
    RecoveryCase(
        "planning-refresh-and-retry",
        FaultType.PLANNING_FAILURE,
        ("clearance", "right_detour", "pregrasp", "approach"),
        (
            RecoveryAction(RecoveryActionType.STOP),
            RecoveryAction(RecoveryActionType.REOBSERVE, {"target": "obstacles"}),
            RecoveryAction(RecoveryActionType.MOVE_TO_CLEARANCE),
            RecoveryAction(RecoveryActionType.REPLAN_ARM, {"new_seed": True}),
            RecoveryAction(RecoveryActionType.RESUME_TASK),
        ),
        ("planning", "dynamic_obstacle"),
        0.76,
    ),
    RecoveryCase(
        "sensor-reset-reobserve",
        FaultType.SENSOR_FAULT,
        ("clearance", "right_detour", "pregrasp", "approach", "close", "lift"),
        (
            RecoveryAction(RecoveryActionType.STOP),
            RecoveryAction(RecoveryActionType.REOBSERVE, {"target": "all_sensors"}),
            RecoveryAction(RecoveryActionType.MOVE_TO_CLEARANCE),
            RecoveryAction(RecoveryActionType.RESUME_TASK),
        ),
        ("sensor", "safety"),
        0.81,
    ),
)


class RecoveryCaseLibrary:
    def __init__(self, cases=DEFAULT_CASES):
        self.cases = tuple(cases)

    def retrieve(self, fault: FaultType, stage: str, tags=(), limit: int = 3):
        requested_tags = set(tags)

        def score(case: RecoveryCase) -> float:
            value = 3.0 if case.fault == fault else 0.0
            value += 1.5 if stage in case.applicable_stages else 0.0
            value += 0.25 * len(requested_tags.intersection(case.tags))
            value += case.success_rate
            return value

        candidates = [case for case in self.cases if case.fault == fault]
        return tuple(sorted(candidates, key=score, reverse=True)[:limit])
