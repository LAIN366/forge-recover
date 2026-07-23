"""Stage-aware multimodal diagnosis for cooperative dual-arm manipulation."""

from dataclasses import dataclass, field
from enum import Enum
import math


class DualArmFaultType(str, Enum):
    NORMAL = "normal"
    VISUAL_DEGRADATION = "visual_degradation"
    LEFT_GRASP_LOSS = "left_grasp_loss"
    RIGHT_GRASP_LOSS = "right_grasp_loss"
    BIMANUAL_SLIP = "bimanual_slip"
    SYNCHRONIZATION_ERROR = "synchronization_error"
    DYNAMIC_OBSTACLE = "dynamic_obstacle"
    IK_FAILURE = "ik_failure"
    PLANNING_FAILURE = "planning_failure"


@dataclass(frozen=True)
class DualArmObservation:
    timestamp: float
    stage: str
    object_position: tuple[float, float, float]
    object_rpy: tuple[float, float, float]
    left_tool_position: tuple[float, float, float]
    right_tool_position: tuple[float, float, float]
    left_contacts: tuple[bool, bool]
    right_contacts: tuple[bool, bool]
    left_force: float = 0.0
    right_force: float = 0.0
    visual_confidence: float = 1.0
    visual_valid: bool = True
    object_vertical_velocity: float = 0.0
    synchronization_error: float = 0.0
    left_tracking_error: float = 0.0
    right_tracking_error: float = 0.0
    collision: bool = False
    ik_success: bool = True
    planning_success: bool = True
    event: str | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def left_grasp(self):
        return all(self.left_contacts)

    @property
    def right_grasp(self):
        return all(self.right_contacts)


@dataclass(frozen=True)
class DualArmDiagnosticEvidence:
    fault: DualArmFaultType
    confidence: float
    reasons: tuple[str, ...]
    recommended_probes: tuple[str, ...] = ()


@dataclass(frozen=True)
class DualArmDiagnosisReport:
    timestamp: float
    stage: str
    anomalous: bool
    primary_fault: DualArmFaultType
    confidence: float
    hypotheses: tuple[DualArmDiagnosticEvidence, ...]
    selected_probe: str | None = None


@dataclass(frozen=True)
class DualArmDetectorThresholds:
    minimum_confidence: float = 0.60
    minimum_visual_confidence: float = 0.42
    maximum_synchronization_error: float = 0.035
    maximum_tracking_error: float = 0.045
    maximum_object_tilt: float = math.radians(12.0)
    slip_velocity: float = -0.025


class DualArmAnomalyDetector:
    """Diagnose cooperative failures using stage and cross-arm consistency."""

    def __init__(self, thresholds=None):
        self.thresholds = thresholds or DualArmDetectorThresholds()

    def update(self, observation):
        evidence = self._collect_evidence(observation)
        evidence.sort(key=lambda item: item.confidence, reverse=True)
        primary = evidence[0] if evidence else DualArmDiagnosticEvidence(
            DualArmFaultType.NORMAL, 1.0, ("multimodal state is consistent",)
        )
        anomalous = (
            primary.fault != DualArmFaultType.NORMAL
            and primary.confidence >= self.thresholds.minimum_confidence
        )
        probe = (
            primary.recommended_probes[0]
            if anomalous and primary.recommended_probes else None
        )
        return DualArmDiagnosisReport(
            timestamp=observation.timestamp,
            stage=observation.stage,
            anomalous=anomalous,
            primary_fault=(
                primary.fault if anomalous else DualArmFaultType.NORMAL
            ),
            confidence=primary.confidence if anomalous else 1.0,
            hypotheses=tuple(evidence),
            selected_probe=probe,
        )

    def _collect_evidence(self, observation):
        result = []
        if not observation.visual_valid or (
            observation.visual_confidence
            < self.thresholds.minimum_visual_confidence
        ):
            result.append(DualArmDiagnosticEvidence(
                DualArmFaultType.VISUAL_DEGRADATION,
                0.94,
                ("RGB-D observation is missing or low confidence",),
                ("active_reobserve",),
            ))
        if observation.collision:
            result.append(DualArmDiagnosticEvidence(
                DualArmFaultType.DYNAMIC_OBSTACLE,
                0.99,
                ("new obstacle invalidated the active trajectory",),
                ("retreat_and_reobserve",),
            ))
        if not observation.ik_success:
            result.append(DualArmDiagnosticEvidence(
                DualArmFaultType.IK_FAILURE,
                0.97,
                ("one or both arm IK problems are infeasible",),
                ("reassign_roles",),
            ))
        if not observation.planning_success:
            result.append(DualArmDiagnosticEvidence(
                DualArmFaultType.PLANNING_FAILURE,
                0.97,
                ("coupled collision planner returned no path",),
                ("reassign_roles",),
            ))

        cooperative_stages = {
            "verify_grasp", "cooperative_lift", "cooperative_transport",
            "place_object",
        }
        if observation.stage in cooperative_stages:
            left_grasp = observation.left_grasp
            right_grasp = observation.right_grasp
            if not left_grasp and right_grasp:
                result.append(DualArmDiagnosticEvidence(
                    DualArmFaultType.LEFT_GRASP_LOSS,
                    0.92,
                    ("left bilateral contact was lost",),
                    ("pause_and_micro_lift", "left_regrasp"),
                ))
            elif left_grasp and not right_grasp:
                result.append(DualArmDiagnosticEvidence(
                    DualArmFaultType.RIGHT_GRASP_LOSS,
                    0.92,
                    ("right bilateral contact was lost",),
                    ("pause_and_micro_lift", "right_regrasp"),
                ))
            elif (
                not left_grasp
                and not right_grasp
                and observation.object_vertical_velocity
                <= self.thresholds.slip_velocity
            ):
                result.append(DualArmDiagnosticEvidence(
                    DualArmFaultType.BIMANUAL_SLIP,
                    0.98,
                    ("both grasps were lost while the object moved downward",),
                    ("pause_and_hold",),
                ))

            tilt = max(abs(observation.object_rpy[0]), abs(observation.object_rpy[1]))
            sync_excess = (
                observation.synchronization_error
                / self.thresholds.maximum_synchronization_error
            )
            tracking_excess = max(
                observation.left_tracking_error,
                observation.right_tracking_error,
            ) / self.thresholds.maximum_tracking_error
            tilt_excess = tilt / self.thresholds.maximum_object_tilt
            if (
                sync_excess >= 1.0
                or tracking_excess >= 1.0
                or tilt_excess >= 1.0
            ):
                confidence = min(
                    0.96,
                    0.58
                    + 0.18 * max(sync_excess, tracking_excess, tilt_excess),
                )
                reasons = []
                if sync_excess >= 1.0 or tracking_excess >= 1.0:
                    reasons.append(
                        "left-right trajectory synchronization diverged"
                    )
                if tilt >= self.thresholds.maximum_object_tilt:
                    reasons.append("object tilt exceeded the cooperative limit")
                    confidence = max(confidence, 0.90)
                result.append(DualArmDiagnosticEvidence(
                    DualArmFaultType.SYNCHRONIZATION_ERROR,
                    confidence,
                    tuple(reasons),
                    ("pause_leading_arm", "resynchronize"),
                ))
        return result
