"""Tests for contextual Bayesian recovery experience learning."""

import pytest

from cruzr_sim.recovery.experience_graph import (
    ContextualRecoveryExperienceGraph,
    RecoveryContext,
)


def context(stage="cooperative_transport", visual=True):
    return RecoveryContext(
        "bimanual_slip", stage, False, False, visual, "high"
    )


def test_repeated_failure_changes_strategy_selection():
    graph = ContextualRecoveryExperienceGraph()
    candidates = (("fast_regrasp", 1.0), ("safe_regrasp", 2.2))
    initial, _ = graph.select(context(), candidates)
    assert initial.strategy_id == "fast_regrasp"
    for _ in range(8):
        graph.record(context(), "fast_regrasp", False, 4.0)
    learned, estimates = graph.select(context(), candidates)
    assert learned.strategy_id == "safe_regrasp"
    assert len(estimates) == 2


def test_similar_context_transfers_discounted_experience():
    graph = ContextualRecoveryExperienceGraph()
    graph.record(context(), "safe_regrasp", True, 1.5)
    estimate = graph.estimate(
        context(stage="cooperative_lift"), "safe_regrasp", 2.0
    )
    assert 0.0 < estimate.effective_samples < 1.0
    assert estimate.success_probability > graph.prior_success


def test_cost_cvar_emphasizes_tail_outcome():
    graph = ContextualRecoveryExperienceGraph(cvar_alpha=0.8)
    for cost in (1.0, 1.0, 1.0, 1.0, 8.0):
        graph.record(context(), "strategy", True, cost)
    estimate = graph.estimate(context(), "strategy")
    assert estimate.cost_cvar == pytest.approx(8.0)
    assert estimate.cost_cvar > estimate.expected_cost


def test_experience_graph_round_trips_json(tmp_path):
    graph = ContextualRecoveryExperienceGraph()
    graph.record(context(), "safe_regrasp", True, 1.25)
    path = tmp_path / "experience.json"
    graph.save(path)
    restored = ContextualRecoveryExperienceGraph.load(path)
    estimate = restored.estimate(context(), "safe_regrasp")
    assert estimate.effective_samples == pytest.approx(1.0)
    assert estimate.expected_cost == pytest.approx(1.25)


def test_cvar_ablation_removes_tail_penalty_only():
    full = ContextualRecoveryExperienceGraph(selection_mode="full")
    no_cvar = ContextualRecoveryExperienceGraph(selection_mode="no_cvar")
    for graph in (full, no_cvar):
        for cost in (1.0, 1.0, 8.0):
            graph.record(context(), "strategy", True, cost)
    assert (
        full.estimate(context(), "strategy").total_score
        > no_cvar.estimate(context(), "strategy").total_score
    )
