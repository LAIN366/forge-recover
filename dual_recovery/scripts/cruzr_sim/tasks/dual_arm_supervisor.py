"""Closed-loop fault confirmation and recovery planning for dual-arm tasks."""

from dataclasses import dataclass

from cruzr_sim.diagnosis.dual_arm import (
    DualArmAnomalyDetector,
    DualArmDiagnosisReport,
    DualArmFaultType,
    DualArmObservation,
)
from cruzr_sim.diagnosis.temporal_belief import StageConditionedFaultFilter
from cruzr_sim.recovery.dual_arm import DualArmRecoveryPlan, DualArmRecoveryPlanner
from cruzr_sim.recovery.experience_graph import RecoveryContext


@dataclass(frozen=True)
class DualArmSupervisorDecision:
    report: DualArmDiagnosisReport
    confirmed_fault: DualArmFaultType
    recovery_plan: DualArmRecoveryPlan | None = None
    fault_posterior: float = 0.0
    belief_entropy: float = 0.0
    selected_probe: str | None = None
    expected_information_gain: float = 0.0
    fault_distribution: dict[str, float] | None = None


class DualArmSupervisor:
    """Suppress transient evidence and issue one bounded graph recovery plan."""

    IMMEDIATE_FAULTS = {
        DualArmFaultType.DYNAMIC_OBSTACLE,
        DualArmFaultType.IK_FAILURE,
        DualArmFaultType.PLANNING_FAILURE,
        DualArmFaultType.BIMANUAL_SLIP,
    }

    def __init__(
        self,
        detector=None,
        recovery_planner=None,
        confirmation_frames=3,
        maximum_recoveries=3,
        use_temporal_belief=True,
        belief_threshold=0.68,
        use_experience_graph=True,
        update_experience_graph=True,
    ):
        self.detector = detector or DualArmAnomalyDetector()
        self.recovery_planner = recovery_planner or DualArmRecoveryPlanner()
        self.confirmation_frames = max(1, int(confirmation_frames))
        self.maximum_recoveries = max(0, int(maximum_recoveries))
        self.use_temporal_belief = bool(use_temporal_belief)
        self.belief_threshold = float(belief_threshold)
        self.belief_filter = StageConditionedFaultFilter()
        self.use_experience_graph = bool(use_experience_graph)
        self.update_experience_graph = bool(update_experience_graph)
        self.reset()

    def reset(self):
        self.consecutive_fault = DualArmFaultType.NORMAL
        self.consecutive_count = 0
        self.recovery_attempts = 0
        self.active_plan = None
        self.belief_filter.reset()
        self.active_context = None
        self.recovery_started_at = None
        self.latest_timestamp = None

    def observe(self, observation: DualArmObservation):
        self.latest_timestamp = float(observation.timestamp)
        report = self.detector.update(observation)
        estimate = (
            self.belief_filter.update(report)
            if self.use_temporal_belief else None
        )
        if not report.anomalous:
            self.consecutive_fault = DualArmFaultType.NORMAL
            self.consecutive_count = 0
            return self._decision(report, DualArmFaultType.NORMAL, estimate)

        if report.primary_fault == self.consecutive_fault:
            self.consecutive_count += 1
        else:
            self.consecutive_fault = report.primary_fault
            self.consecutive_count = 1

        confirmed = report.primary_fault in self.IMMEDIATE_FAULTS
        if self.use_temporal_belief:
            confirmed = confirmed or (
                estimate.fault == report.primary_fault
                and estimate.confidence >= self.belief_threshold
            )
        else:
            confirmed = confirmed or self.consecutive_count >= self.confirmation_frames
        if not confirmed:
            return self._decision(report, DualArmFaultType.NORMAL, estimate)
        if self.active_plan is not None:
            return self._decision(
                report, report.primary_fault, estimate, self.active_plan
            )
        if self.recovery_attempts >= self.maximum_recoveries:
            return self._decision(report, report.primary_fault, estimate)

        self.recovery_attempts += 1
        self.active_context = RecoveryContext(
            fault=report.primary_fault.value,
            stage=observation.stage,
            left_contact=observation.left_grasp,
            right_contact=observation.right_grasp,
            visual_reliable=(
                observation.visual_valid and observation.visual_confidence >= 0.42
            ),
            severity_bin=self._severity_bin(report.confidence),
        )
        self.recovery_started_at = float(observation.timestamp)
        self.active_plan = self.recovery_planner.plan(
            report,
            observation.stage,
            context=(self.active_context if self.use_experience_graph else None),
        )
        return self._decision(
            report, report.primary_fault, estimate, self.active_plan
        )

    @staticmethod
    def _decision(report, confirmed_fault, estimate, recovery_plan=None):
        return DualArmSupervisorDecision(
            report,
            confirmed_fault,
            recovery_plan,
            fault_posterior=(estimate.confidence if estimate else 0.0),
            belief_entropy=(estimate.entropy if estimate else 0.0),
            selected_probe=(estimate.selected_probe if estimate else None),
            expected_information_gain=(
                estimate.expected_information_gain if estimate else 0.0
            ),
            fault_distribution=(
                {fault.value: value for fault, value in estimate.posterior.items()}
                if estimate else None
            ),
        )

    @staticmethod
    def _severity_bin(confidence):
        if confidence >= 0.9:
            return "high"
        if confidence >= 0.75:
            return "medium"
        return "low"

    def mark_recovery_complete(self, successful, timestamp=None):
        completed_plan = self.active_plan
        completed_context = self.active_context
        completion_time = (
            float(timestamp) if timestamp is not None
            else float(self.latest_timestamp or 0.0)
        )
        duration = max(
            0.0,
            completion_time
            - float(self.recovery_started_at or self.latest_timestamp or 0.0),
        )
        experience_node = None
        if (
            self.update_experience_graph
            and completed_plan is not None
            and completed_context is not None
        ):
            experience_node = self.recovery_planner.record_outcome(
                completed_context, completed_plan, bool(successful), duration
            )
        self.active_plan = None
        self.active_context = None
        self.recovery_started_at = None
        self.consecutive_fault = DualArmFaultType.NORMAL
        self.consecutive_count = 0
        if not successful:
            return experience_node
        self.belief_filter.reset()
        return experience_node

    def incorporate_probe_result(self, probe, positive):
        if not self.use_temporal_belief:
            return None
        return self.belief_filter.incorporate_probe_result(probe, positive)
