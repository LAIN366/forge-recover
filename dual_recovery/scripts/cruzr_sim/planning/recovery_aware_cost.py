"""Risk-sensitive scoring for recovery-aware manipulation plans."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PlanRiskProfile:
    plan_id: str
    nominal_cost: float
    failure_probability: float
    recovery_costs: tuple[float, ...]
    minimum_clearance: float
    visibility: float
    grasp_stability: float


@dataclass(frozen=True)
class PlanRiskEvaluation:
    profile: PlanRiskProfile
    expected_cost: float
    cvar: float
    clearance_penalty: float
    visibility_penalty: float
    stability_penalty: float
    total_cost: float


class RecoveryAwareEvaluator:
    """Combine nominal efficiency, expected recovery cost, and tail risk."""

    def __init__(
        self,
        risk_weight=0.35,
        clearance_weight=0.025,
        visibility_weight=0.8,
        stability_weight=0.7,
        cvar_alpha=0.8,
    ):
        self.risk_weight = float(risk_weight)
        self.clearance_weight = float(clearance_weight)
        self.visibility_weight = float(visibility_weight)
        self.stability_weight = float(stability_weight)
        self.cvar_alpha = float(cvar_alpha)
        if not 0.0 <= self.cvar_alpha < 1.0:
            raise ValueError("cvar_alpha must be in [0, 1)")

    @staticmethod
    def _validate(profile):
        if not 0.0 <= profile.failure_probability <= 1.0:
            raise ValueError("failure_probability must be in [0, 1]")
        if not profile.recovery_costs:
            raise ValueError("at least one recovery cost is required")
        if profile.minimum_clearance <= 0.0:
            raise ValueError("minimum_clearance must be positive")

    @staticmethod
    def _outcomes(profile):
        failure_probability = float(profile.failure_probability)
        count = len(profile.recovery_costs)
        outcomes = [(float(profile.nominal_cost), 1.0 - failure_probability)]
        outcomes.extend(
            (
                float(profile.nominal_cost + recovery_cost),
                failure_probability / count,
            )
            for recovery_cost in profile.recovery_costs
        )
        return sorted(outcomes, key=lambda item: item[0])

    def _cvar(self, outcomes):
        tail_probability = 1.0 - self.cvar_alpha
        remaining = tail_probability
        weighted_sum = 0.0
        for cost, probability in reversed(outcomes):
            take = min(remaining, probability)
            weighted_sum += take * cost
            remaining -= take
            if remaining <= 1e-12:
                break
        if remaining > 1e-12:
            weighted_sum += remaining * outcomes[0][0]
        return weighted_sum / tail_probability

    def evaluate(self, profile):
        self._validate(profile)
        outcomes = self._outcomes(profile)
        expected_cost = sum(cost * probability for cost, probability in outcomes)
        cvar = self._cvar(outcomes)
        clearance_penalty = self.clearance_weight / profile.minimum_clearance
        visibility_penalty = self.visibility_weight * (
            1.0 - min(1.0, max(0.0, profile.visibility))
        )
        stability_penalty = self.stability_weight * (
            1.0 - min(1.0, max(0.0, profile.grasp_stability))
        )
        total_cost = (
            expected_cost
            + self.risk_weight * cvar
            + clearance_penalty
            + visibility_penalty
            + stability_penalty
        )
        return PlanRiskEvaluation(
            profile=profile,
            expected_cost=expected_cost,
            cvar=cvar,
            clearance_penalty=clearance_penalty,
            visibility_penalty=visibility_penalty,
            stability_penalty=stability_penalty,
            total_cost=total_cost,
        )

    def select(self, profiles):
        evaluations = [self.evaluate(profile) for profile in profiles]
        if not evaluations:
            raise ValueError("at least one plan profile is required")
        return min(evaluations, key=lambda item: item.total_cost), evaluations
