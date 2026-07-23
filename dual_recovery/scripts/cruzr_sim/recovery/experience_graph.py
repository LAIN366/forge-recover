"""Contextual Bayesian experience graph for risk-aware recovery selection."""

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class RecoveryContext:
    fault: str
    stage: str
    left_contact: bool
    right_contact: bool
    visual_reliable: bool
    severity_bin: str = "nominal"


@dataclass
class RecoveryExperienceNode:
    context: RecoveryContext
    strategy_id: str
    successes: int = 0
    failures: int = 0
    costs: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class RecoveryStrategyEstimate:
    strategy_id: str
    success_probability: float
    success_variance: float
    expected_cost: float
    cost_cvar: float
    effective_samples: float
    total_score: float


class ContextualRecoveryExperienceGraph:
    """Transfer experience between similar contexts without unsafe exploration."""

    def __init__(
        self, prior_success=0.7, prior_strength=4.0, failure_penalty=6.0,
        uncertainty_weight=1.0, tail_risk_weight=0.35, cvar_alpha=0.8,
        selection_mode="full",
    ):
        self.prior_success = float(prior_success)
        self.prior_strength = float(prior_strength)
        self.failure_penalty = float(failure_penalty)
        self.uncertainty_weight = float(uncertainty_weight)
        self.tail_risk_weight = float(tail_risk_weight)
        self.cvar_alpha = float(cvar_alpha)
        if selection_mode not in {"full", "success_only", "no_cvar"}:
            raise ValueError(f"invalid experience selection mode: {selection_mode}")
        self.selection_mode = selection_mode
        self.nodes = {}

    @staticmethod
    def _key(context, strategy_id):
        return (
            context.fault, context.stage, context.left_contact,
            context.right_contact, context.visual_reliable,
            context.severity_bin, strategy_id,
        )

    @staticmethod
    def context_similarity(query, stored):
        if query.fault != stored.fault:
            return 0.0
        similarity = 1.0
        similarity *= 1.0 if query.stage == stored.stage else 0.55
        similarity *= 1.0 if query.left_contact == stored.left_contact else 0.65
        similarity *= 1.0 if query.right_contact == stored.right_contact else 0.65
        similarity *= 1.0 if query.visual_reliable == stored.visual_reliable else 0.60
        similarity *= 1.0 if query.severity_bin == stored.severity_bin else 0.75
        return similarity

    def record(self, context, strategy_id, successful, cost):
        key = self._key(context, strategy_id)
        node = self.nodes.setdefault(
            key, RecoveryExperienceNode(context, str(strategy_id))
        )
        if successful:
            node.successes += 1
        else:
            node.failures += 1
        node.costs.append(max(0.0, float(cost)))
        return node

    def estimate(self, context, strategy_id, nominal_cost=1.0):
        alpha = self.prior_success * self.prior_strength
        beta = (1.0 - self.prior_success) * self.prior_strength
        weighted_costs = []
        effective_samples = 0.0
        for node in self.nodes.values():
            if node.strategy_id != strategy_id:
                continue
            weight = self.context_similarity(context, node.context)
            if weight <= 0.0:
                continue
            alpha += weight * node.successes
            beta += weight * node.failures
            effective_samples += weight * (node.successes + node.failures)
            weighted_costs.extend((cost, weight) for cost in node.costs)
        success_probability = alpha / (alpha + beta)
        success_variance = alpha * beta / (
            (alpha + beta) ** 2 * (alpha + beta + 1.0)
        )
        expected_cost = self._weighted_mean(weighted_costs, nominal_cost)
        cost_cvar = self._weighted_cvar(weighted_costs, nominal_cost)
        total_score = expected_cost + self.failure_penalty * (
            1.0 - success_probability
        )
        if self.selection_mode != "success_only":
            total_score += self.uncertainty_weight * math.sqrt(success_variance)
        if self.selection_mode == "full":
            total_score += self.tail_risk_weight * cost_cvar
        return RecoveryStrategyEstimate(
            strategy_id, success_probability, success_variance,
            expected_cost, cost_cvar, effective_samples, total_score,
        )

    @staticmethod
    def _weighted_mean(values, fallback):
        total_weight = sum(weight for _, weight in values)
        if total_weight <= 0.0:
            return float(fallback)
        return sum(cost * weight for cost, weight in values) / total_weight

    def _weighted_cvar(self, values, fallback):
        if not values:
            return float(fallback)
        ordered = sorted(values, key=lambda item: item[0], reverse=True)
        tail_weight = (1.0 - self.cvar_alpha) * sum(w for _, w in ordered)
        remaining = max(tail_weight, 1e-9)
        total = 0.0
        for cost, weight in ordered:
            take = min(remaining, weight)
            total += cost * take
            remaining -= take
            if remaining <= 1e-9:
                break
        return total / max(tail_weight, 1e-9)

    def select(self, context, candidates):
        estimates = tuple(
            self.estimate(context, strategy_id, nominal_cost)
            for strategy_id, nominal_cost in candidates
        )
        if not estimates:
            raise ValueError("at least one recovery strategy is required")
        return min(estimates, key=lambda item: item.total_score), estimates

    def save(self, path):
        payload = [
            {
                "context": asdict(node.context),
                "strategy_id": node.strategy_id,
                "successes": node.successes,
                "failures": node.failures,
                "costs": node.costs,
            }
            for node in self.nodes.values()
        ]
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path, **kwargs):
        graph = cls(**kwargs)
        for item in json.loads(Path(path).read_text(encoding="utf-8")):
            context = RecoveryContext(**item["context"])
            node = RecoveryExperienceNode(
                context, item["strategy_id"], item["successes"],
                item["failures"], list(item["costs"]),
            )
            graph.nodes[graph._key(context, node.strategy_id)] = node
        return graph
