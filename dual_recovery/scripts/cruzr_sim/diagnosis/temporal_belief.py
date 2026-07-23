"""Stage-conditioned temporal fault inference and information-gain probes."""

from dataclasses import dataclass
import math

from .dual_arm import DualArmFaultType


FAULTS = tuple(DualArmFaultType)
_EPSILON = 1e-9

STAGE_FAULT_PRIORS = {
    "dual_grasp": {
        DualArmFaultType.LEFT_GRASP_LOSS: 0.10,
        DualArmFaultType.RIGHT_GRASP_LOSS: 0.10,
        DualArmFaultType.VISUAL_DEGRADATION: 0.05,
    },
    "verify_grasp": {
        DualArmFaultType.LEFT_GRASP_LOSS: 0.12,
        DualArmFaultType.RIGHT_GRASP_LOSS: 0.12,
        DualArmFaultType.BIMANUAL_SLIP: 0.06,
    },
    "cooperative_lift": {
        DualArmFaultType.LEFT_GRASP_LOSS: 0.08,
        DualArmFaultType.RIGHT_GRASP_LOSS: 0.08,
        DualArmFaultType.BIMANUAL_SLIP: 0.10,
        DualArmFaultType.SYNCHRONIZATION_ERROR: 0.08,
    },
    "cooperative_transport": {
        DualArmFaultType.BIMANUAL_SLIP: 0.10,
        DualArmFaultType.SYNCHRONIZATION_ERROR: 0.10,
        DualArmFaultType.DYNAMIC_OBSTACLE: 0.08,
        DualArmFaultType.VISUAL_DEGRADATION: 0.06,
    },
}

PROBE_POSITIVE_LIKELIHOODS = {
    "active_reobserve": {
        DualArmFaultType.VISUAL_DEGRADATION: 0.88,
        DualArmFaultType.DYNAMIC_OBSTACLE: 0.62,
    },
    "pause_and_contact_test": {
        DualArmFaultType.LEFT_GRASP_LOSS: 0.82,
        DualArmFaultType.RIGHT_GRASP_LOSS: 0.82,
        DualArmFaultType.BIMANUAL_SLIP: 0.68,
    },
    "pause_and_hold": {
        DualArmFaultType.BIMANUAL_SLIP: 0.88,
        DualArmFaultType.SYNCHRONIZATION_ERROR: 0.38,
    },
    "pause_and_resample_sync": {
        DualArmFaultType.SYNCHRONIZATION_ERROR: 0.84,
        DualArmFaultType.BIMANUAL_SLIP: 0.30,
    },
    "retreat_and_reobserve": {
        DualArmFaultType.DYNAMIC_OBSTACLE: 0.90,
        DualArmFaultType.VISUAL_DEGRADATION: 0.35,
    },
    "reassign_roles": {
        DualArmFaultType.IK_FAILURE: 0.78,
        DualArmFaultType.PLANNING_FAILURE: 0.72,
    },
}


def _normalize(values):
    total = sum(max(0.0, value) for value in values.values())
    if total <= _EPSILON:
        return {fault: 1.0 / len(FAULTS) for fault in FAULTS}
    return {fault: max(0.0, values.get(fault, 0.0)) / total for fault in FAULTS}


def entropy(distribution):
    return -sum(
        probability * math.log(probability + _EPSILON, 2)
        for probability in distribution.values()
        if probability > 0.0
    )


@dataclass(frozen=True)
class TemporalFaultEstimate:
    fault: DualArmFaultType
    confidence: float
    entropy: float
    selected_probe: str | None
    expected_information_gain: float
    posterior: dict[DualArmFaultType, float]


class StageConditionedFaultFilter:
    """A compact HMM-like filter with task-stage priors and active sensing."""

    def __init__(self, persistence=0.82, probe_cost=0.03):
        self.persistence = min(0.999, max(0.0, float(persistence)))
        self.probe_cost = max(0.0, float(probe_cost))
        self.reset()

    def reset(self):
        self.posterior = {fault: 0.0 for fault in FAULTS}
        self.posterior[DualArmFaultType.NORMAL] = 1.0
        self.trace = []
        self.attempted_probes = set()

    @staticmethod
    def _stage_prior(stage):
        assigned = STAGE_FAULT_PRIORS.get(stage, {})
        values = {fault: assigned.get(fault, 0.015) for fault in FAULTS}
        values[DualArmFaultType.NORMAL] = max(
            0.45, 1.0 - sum(value for fault, value in values.items()
                            if fault != DualArmFaultType.NORMAL)
        )
        return _normalize(values)

    def _predict(self, stage):
        stage_prior = self._stage_prior(stage)
        return {
            fault: (
                self.persistence * self.posterior[fault]
                + (1.0 - self.persistence) * stage_prior[fault]
            )
            for fault in FAULTS
        }

    @staticmethod
    def _likelihoods(report):
        observed = {item.fault: item.confidence for item in report.hypotheses}
        values = {}
        for fault in FAULTS:
            if fault == DualArmFaultType.NORMAL:
                values[fault] = 0.94 if not report.anomalous else 0.04
            elif fault in observed:
                values[fault] = max(0.05, min(0.99, observed[fault]))
            else:
                values[fault] = 0.06
        return values

    def update(self, report):
        predicted = self._predict(report.stage)
        likelihoods = self._likelihoods(report)
        self.posterior = _normalize({
            fault: predicted[fault] * likelihoods[fault] for fault in FAULTS
        })
        fault, confidence = max(self.posterior.items(), key=lambda item: item[1])
        probes = {
            probe
            for hypothesis in report.hypotheses
            for probe in hypothesis.recommended_probes
        }
        selected_probe, gain = self.select_probe(probes)
        estimate = TemporalFaultEstimate(
            fault, confidence, entropy(self.posterior), selected_probe, gain,
            dict(self.posterior),
        )
        self.trace.append(estimate)
        return estimate

    def select_probe(self, candidates):
        current_entropy = entropy(self.posterior)
        best_probe, best_value, best_gain = None, 0.0, 0.0
        for probe in candidates:
            if probe in self.attempted_probes:
                continue
            model = PROBE_POSITIVE_LIKELIHOODS.get(probe)
            if not model:
                continue
            positive = {
                fault: model.get(fault, 0.12) for fault in FAULTS
            }
            probability_positive = sum(
                self.posterior[fault] * positive[fault] for fault in FAULTS
            )
            posterior_positive = _normalize({
                fault: self.posterior[fault] * positive[fault] for fault in FAULTS
            })
            posterior_negative = _normalize({
                fault: self.posterior[fault] * (1.0 - positive[fault])
                for fault in FAULTS
            })
            expected_entropy = (
                probability_positive * entropy(posterior_positive)
                + (1.0 - probability_positive) * entropy(posterior_negative)
            )
            gain = max(0.0, current_entropy - expected_entropy)
            value = gain - self.probe_cost
            if value > best_value:
                best_probe, best_value, best_gain = probe, value, gain
        return best_probe, best_gain

    def incorporate_probe_result(self, probe, positive):
        """Apply an executed probe outcome as Bayesian evidence."""
        model = PROBE_POSITIVE_LIKELIHOODS.get(probe)
        if model is None:
            raise ValueError(f"unknown active diagnosis probe: {probe}")
        self.attempted_probes.add(probe)
        likelihoods = {
            fault: model.get(fault, 0.12) for fault in FAULTS
        }
        if not bool(positive):
            likelihoods = {
                fault: 1.0 - value for fault, value in likelihoods.items()
            }
        self.posterior = _normalize({
            fault: self.posterior[fault] * likelihoods[fault]
            for fault in FAULTS
        })
        fault, confidence = max(self.posterior.items(), key=lambda item: item[1])
        estimate = TemporalFaultEstimate(
            fault=fault,
            confidence=confidence,
            entropy=entropy(self.posterior),
            selected_probe=None,
            expected_information_gain=0.0,
            posterior=dict(self.posterior),
        )
        self.trace.append(estimate)
        return estimate
