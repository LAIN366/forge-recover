"""Interpretable, stage-aware baseline for manipulation anomalies."""

from dataclasses import dataclass
import math

from .feature_extractor import TemporalFeatureExtractor, TemporalFeatures
from .types import DiagnosisReport, DiagnosticEvidence, FaultType, ManipulationObservation


@dataclass(frozen=True)
class DetectorThresholds:
    minimum_confidence: float = 0.55
    active_probe_confidence: float = 0.78
    missed_grasp_delay: float = 1.8
    slip_velocity: float = -0.025
    slip_height_change: float = -0.012
    target_displacement: float = 0.045
    tracking_error: float = 0.045
    minimum_grasp_force: float = 0.8
    high_tangent_ratio: float = 0.75


class StageAwareAnomalyDetector:
    """Combine immediate safety events with short-horizon temporal evidence."""

    def __init__(self, thresholds: DetectorThresholds | None = None):
        self.thresholds = thresholds or DetectorThresholds()
        self.features = TemporalFeatureExtractor()
        self.stage_started_at: dict[str, float] = {}
        self.last_stage: str | None = None

    def reset(self) -> None:
        self.features.reset()
        self.stage_started_at.clear()
        self.last_stage = None

    def update(self, observation: ManipulationObservation) -> DiagnosisReport:
        if observation.stage != self.last_stage:
            self.stage_started_at[observation.stage] = observation.timestamp
            self.features.reset()
            self.last_stage = observation.stage
        temporal = self.features.update(observation)
        evidence = self._collect_evidence(observation, temporal)
        evidence.sort(key=lambda item: item.confidence, reverse=True)
        primary = evidence[0] if evidence else DiagnosticEvidence(
            FaultType.NORMAL, 1.0, ("no anomaly threshold exceeded",)
        )
        anomalous = (
            primary.fault != FaultType.NORMAL
            and primary.confidence >= self.thresholds.minimum_confidence
        )
        uncertain = anomalous and primary.confidence < self.thresholds.active_probe_confidence
        selected_probe = (
            primary.recommended_probes[0]
            if uncertain and primary.recommended_probes else None
        )
        return DiagnosisReport(
            timestamp=observation.timestamp,
            stage=observation.stage,
            anomalous=anomalous,
            primary_fault=primary.fault if anomalous else FaultType.NORMAL,
            confidence=primary.confidence if anomalous else 1.0,
            hypotheses=tuple(evidence[:4]),
            needs_active_probe=bool(selected_probe),
            selected_probe=selected_probe,
        )

    def _collect_evidence(
        self, observation: ManipulationObservation, temporal: TemporalFeatures
    ) -> list[DiagnosticEvidence]:
        result: list[DiagnosticEvidence] = []
        if not observation.sensor_valid:
            result.append(DiagnosticEvidence(
                FaultType.SENSOR_FAULT, 0.99,
                ("invalid or non-finite sensor sample",), ("reobserve",),
            ))
        if observation.collision:
            result.append(DiagnosticEvidence(
                FaultType.COLLISION, 0.99,
                ("forbidden collision reported",), ("retreat", "reobserve"),
            ))
        if not observation.ik_success:
            result.append(DiagnosticEvidence(
                FaultType.IK_FAILURE, 0.97,
                ("inverse kinematics did not converge",),
                ("change_configuration", "move_base"),
            ))
        if not observation.planning_success:
            result.append(DiagnosticEvidence(
                FaultType.PLANNING_FAILURE, 0.97,
                ("motion planner returned no path",), ("reobserve", "move_base"),
            ))

        elapsed = observation.timestamp - self.stage_started_at.get(
            observation.stage, observation.timestamp
        )
        stable_grasp_stages = {"lift", "hold", "transport"}
        if observation.stage == "close" and elapsed >= self.thresholds.missed_grasp_delay:
            if (
                not observation.any_contact
                and temporal.mean_normal_force < self.thresholds.minimum_grasp_force
            ):
                result.append(DiagnosticEvidence(
                    FaultType.MISSED_GRASP, 0.92,
                    ("gripper closed without object contact", "normal force remained low"),
                    ("reobserve", "small_gripper_close"),
                ))

        if observation.stage in {"lift", "hold", "transport", "fault_slip"}:
            slip_score = 0.0
            reasons = []
            if not observation.both_contacts:
                slip_score += 0.34
                reasons.append("bilateral contact was lost")
            if temporal.mean_vertical_velocity <= self.thresholds.slip_velocity:
                slip_score += 0.34
                reasons.append("object is moving downward")
            if temporal.height_change <= self.thresholds.slip_height_change:
                slip_score += 0.28
                reasons.append("object height decreased in the temporal window")
            if slip_score:
                result.append(DiagnosticEvidence(
                    FaultType.GRASP_SLIP, min(0.98, 0.30 + slip_score),
                    tuple(reasons), ("pause_and_hold", "reobserve"),
                ))

        if observation.stage in stable_grasp_stages and observation.any_contact:
            unstable_score = 0.0
            reasons = []
            if observation.left_contact != observation.right_contact:
                unstable_score += 0.35
                reasons.append("contact is present on only one finger")
            if temporal.asymmetric_contact_ratio > 0.45:
                unstable_score += 0.20
                reasons.append("contact remained asymmetric")
            if temporal.mean_tangent_ratio > self.thresholds.high_tangent_ratio:
                unstable_score += 0.25
                reasons.append("tangential-to-normal force ratio is high")
            if unstable_score >= 0.40:
                result.append(DiagnosticEvidence(
                    FaultType.UNSTABLE_GRASP, min(0.90, 0.30 + unstable_score),
                    tuple(reasons), ("micro_lift", "adjust_grip_force"),
                ))

        if (
            observation.target_position is not None
            and not observation.any_contact
            and observation.stage in {"right_detour", "pregrasp", "approach", "close"}
        ):
            displacement = float(math.dist(
                observation.target_position, observation.object_position
            ))
            if displacement >= self.thresholds.target_displacement:
                result.append(DiagnosticEvidence(
                    FaultType.TARGET_DISPLACEMENT,
                    min(0.96, 0.55 + displacement * 4.0),
                    (f"target moved {displacement:.3f} m from the planned pose",),
                    ("reobserve",),
                ))

        if observation.tracking_error >= self.thresholds.tracking_error:
            result.append(DiagnosticEvidence(
                FaultType.UNKNOWN,
                min(0.85, 0.5 + observation.tracking_error * 3.0),
                (f"tool tracking error is {observation.tracking_error:.3f} m",),
                ("pause_and_hold", "reobserve"),
            ))
        return result
