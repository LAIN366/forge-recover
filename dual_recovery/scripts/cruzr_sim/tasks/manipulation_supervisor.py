"""Closed-loop coordination of diagnosis, active probes, and recovery plans."""

from dataclasses import dataclass, replace

from cruzr_sim.diagnosis import ActiveDiagnoser, StageAwareAnomalyDetector
from cruzr_sim.diagnosis.types import DiagnosisReport, FaultType, ManipulationObservation
from cruzr_sim.recovery import RecoveryPlan, RecoveryPlanner


@dataclass(frozen=True)
class SupervisorDecision:
    report: DiagnosisReport
    confirmed_fault: FaultType
    confidence: float
    probe: str | None = None
    recovery_plan: RecoveryPlan | None = None


class ManipulationSupervisor:
    """Stateful high-level safety supervisor for one manipulation episode."""

    def __init__(self, detector=None, active_diagnoser=None, recovery_planner=None,
                 confirmation_frames: int = 3, maximum_recoveries: int = 3):
        self.detector = detector or StageAwareAnomalyDetector()
        self.active_diagnoser = active_diagnoser or ActiveDiagnoser()
        self.recovery_planner = recovery_planner or RecoveryPlanner()
        self.confirmation_frames = int(confirmation_frames)
        self.maximum_recoveries = int(maximum_recoveries)
        self.reset()

    def reset(self) -> None:
        self.detector.reset()
        self.active_diagnoser.reset()
        self.consecutive_fault = FaultType.NORMAL
        self.consecutive_count = 0
        self.recovery_attempts = 0
        self.active_plan = None
        self.last_planned_fault = FaultType.NORMAL

    def observe(self, observation: ManipulationObservation) -> SupervisorDecision:
        report = self.detector.update(observation)
        if not report.anomalous:
            self.consecutive_fault = FaultType.NORMAL
            self.consecutive_count = 0
            return SupervisorDecision(report, FaultType.NORMAL, 1.0)
        if report.primary_fault == self.consecutive_fault:
            self.consecutive_count += 1
        else:
            self.consecutive_fault = report.primary_fault
            self.consecutive_count = 1
        fault, confidence, probe = self.active_diagnoser.update(report)
        immediate = fault in {
            FaultType.COLLISION, FaultType.IK_FAILURE, FaultType.PLANNING_FAILURE,
            FaultType.SENSOR_FAULT,
        }
        if immediate:
            fault = report.primary_fault
            confidence = report.confidence
        confidence_confirmed = confidence >= self.active_diagnoser.decision_threshold
        if not (
            immediate
            or (
                self.consecutive_count >= self.confirmation_frames
                and confidence_confirmed
            )
        ):
            return SupervisorDecision(report, fault, confidence, probe=probe)
        if self.recovery_attempts >= self.maximum_recoveries:
            return SupervisorDecision(report, fault, confidence)
        if self.active_plan is None or fault != self.last_planned_fault:
            self.recovery_attempts += 1
            self.active_plan = self.recovery_planner.plan(
                report, observation.stage, attempt=self.recovery_attempts
            )
            self.last_planned_fault = fault
        return SupervisorDecision(
            report, fault, confidence, probe=probe, recovery_plan=self.active_plan
        )

    def mark_recovery_complete(self, successful: bool) -> None:
        self.active_plan = None
        self.consecutive_fault = FaultType.NORMAL
        self.consecutive_count = 0
        self.active_diagnoser.reset()
        self.detector.reset()
        if successful:
            self.last_planned_fault = FaultType.NORMAL

    def complete_probe(
        self,
        report: DiagnosisReport,
        stage: str,
        probe: str,
        positive: bool,
    ) -> SupervisorDecision:
        """Fuse a probe outcome and plan only if the updated belief is decisive."""
        self.active_diagnoser.incorporate_probe_result(probe, positive)
        fault, confidence = self.active_diagnoser.leading_hypothesis()
        if self.active_plan is not None:
            return SupervisorDecision(
                report,
                self.active_plan.fault,
                confidence,
                recovery_plan=self.active_plan,
            )

        plan = None
        if (
            positive
            and fault != FaultType.NORMAL
            and confidence >= self.active_diagnoser.decision_threshold
            and self.recovery_attempts < self.maximum_recoveries
        ):
            confirmed_report = replace(
                report,
                anomalous=True,
                primary_fault=fault,
                confidence=confidence,
            )
            self.recovery_attempts += 1
            plan = self.recovery_planner.plan(
                confirmed_report, stage, attempt=self.recovery_attempts
            )
            self.active_plan = plan
            self.last_planned_fault = fault
        return SupervisorDecision(report, fault, confidence, recovery_plan=plan)
