"""Belief update and information-gathering probes for active diagnosis."""

from dataclasses import dataclass, field

from .types import DiagnosisReport, FaultType


PROBE_EXPECTATIONS = {
    "reobserve": {
        FaultType.TARGET_DISPLACEMENT: 0.90,
        FaultType.SENSOR_FAULT: 0.85,
        FaultType.MISSED_GRASP: 0.55,
    },
    "small_gripper_close": {
        FaultType.MISSED_GRASP: 0.85,
        FaultType.UNSTABLE_GRASP: 0.70,
    },
    "micro_lift": {
        FaultType.GRASP_SLIP: 0.90,
        FaultType.UNSTABLE_GRASP: 0.85,
    },
    "pause_and_hold": {
        FaultType.GRASP_SLIP: 0.75,
        FaultType.COLLISION: 0.40,
    },
}


@dataclass
class ActiveDiagnosisState:
    beliefs: dict[FaultType, float] = field(default_factory=dict)
    probes_attempted: list[str] = field(default_factory=list)


class ActiveDiagnoser:
    """Maintain smoothed beliefs and choose a low-risk discriminating probe."""

    def __init__(self, decision_threshold: float = 0.78, smoothing: float = 0.65):
        self.decision_threshold = float(decision_threshold)
        self.smoothing = float(smoothing)
        self.state = ActiveDiagnosisState()

    def reset(self) -> None:
        self.state = ActiveDiagnosisState()

    def update(self, report: DiagnosisReport) -> tuple[FaultType, float, str | None]:
        observed = {item.fault: item.confidence for item in report.hypotheses}
        faults = set(self.state.beliefs) | set(observed)
        for fault in faults:
            previous = self.state.beliefs.get(fault, 0.0)
            current = observed.get(fault, 0.0)
            self.state.beliefs[fault] = (
                self.smoothing * previous + (1.0 - self.smoothing) * current
            )
        if not self.state.beliefs:
            return FaultType.NORMAL, 1.0, None
        fault, confidence = max(self.state.beliefs.items(), key=lambda item: item[1])
        if confidence >= self.decision_threshold:
            return fault, confidence, None
        probe = self._select_probe(fault)
        if probe:
            self.state.probes_attempted.append(probe)
        return fault, confidence, probe

    def incorporate_probe_result(self, probe: str, positive: bool) -> None:
        for fault, likelihood in PROBE_EXPECTATIONS.get(probe, {}).items():
            prior = self.state.beliefs.get(fault, 0.15)
            evidence = likelihood if positive else 1.0 - likelihood
            self.state.beliefs[fault] = min(
                0.99, max(0.0, 0.15 * prior + 0.85 * evidence)
            )

    def leading_hypothesis(self) -> tuple[FaultType, float]:
        if not self.state.beliefs:
            return FaultType.NORMAL, 1.0
        return max(self.state.beliefs.items(), key=lambda item: item[1])

    def _select_probe(self, leading_fault: FaultType) -> str | None:
        candidates = [
            probe for probe, expectations in PROBE_EXPECTATIONS.items()
            if leading_fault in expectations and probe not in self.state.probes_attempted
        ]
        if not candidates:
            return None
        return max(
            candidates, key=lambda probe: PROBE_EXPECTATIONS[probe][leading_fault]
        )
