"""Task, coordination, and motion-layer planning contracts."""

from dataclasses import dataclass

from cruzr_sim.planning.recovery_aware_cost import (
    PlanRiskProfile,
    RecoveryAwareEvaluator,
)


@dataclass(frozen=True)
class HierarchicalCandidate:
    candidate_id: str
    task_node: str
    primary_arm: str
    support_arm: str
    motion_cost: float
    failure_probability: float
    recovery_costs: tuple[float, ...]
    minimum_clearance: float
    visibility: float
    grasp_stability: float


@dataclass(frozen=True)
class HierarchicalDecision:
    candidate: HierarchicalCandidate
    expected_cost: float
    cvar: float
    total_cost: float
    evaluated_candidates: int


class RecoveryAwareHierarchicalPlanner:
    """Rank role and motion candidates using expected recovery cost and CVaR."""

    def __init__(self, evaluator=None):
        self.evaluator = evaluator or RecoveryAwareEvaluator()

    def select(self, candidates):
        candidates = tuple(candidates)
        profiles = tuple(
            PlanRiskProfile(
                plan_id=item.candidate_id,
                nominal_cost=item.motion_cost,
                failure_probability=item.failure_probability,
                recovery_costs=item.recovery_costs,
                minimum_clearance=item.minimum_clearance,
                visibility=item.visibility,
                grasp_stability=item.grasp_stability,
            )
            for item in candidates
        )
        selected, evaluations = self.evaluator.select(profiles)
        by_id = {item.candidate_id: item for item in candidates}
        return HierarchicalDecision(
            candidate=by_id[selected.profile.plan_id],
            expected_cost=selected.expected_cost,
            cvar=selected.cvar,
            total_cost=selected.total_cost,
            evaluated_candidates=len(evaluations),
        )
