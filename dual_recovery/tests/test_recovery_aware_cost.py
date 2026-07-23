"""Tests for expected-cost and CVaR-aware plan selection."""

import importlib.util
import unittest

PLANNING_DEPS_AVAILABLE = (
    importlib.util.find_spec("numpy") is not None
    and importlib.util.find_spec("mujoco") is not None
)

if PLANNING_DEPS_AVAILABLE:
    from cruzr_sim.planning.recovery_aware_cost import (
        PlanRiskProfile,
        RecoveryAwareEvaluator,
    )


def profile(plan_id, nominal, failure, recoveries):
    return PlanRiskProfile(
        plan_id=plan_id,
        nominal_cost=nominal,
        failure_probability=failure,
        recovery_costs=tuple(recoveries),
        minimum_clearance=0.05,
        visibility=0.9,
        grasp_stability=0.9,
    )


@unittest.skipUnless(PLANNING_DEPS_AVAILABLE, "requires planning dependencies")
class RecoveryAwareEvaluatorTest(unittest.TestCase):
    def test_recoverable_plan_beats_short_brittle_plan(self):
        evaluator = RecoveryAwareEvaluator(risk_weight=0.5)
        brittle = profile("brittle", 1.0, 0.35, (8.0, 10.0))
        recoverable = profile("recoverable", 1.5, 0.08, (2.0, 3.0))
        selected, _ = evaluator.select([brittle, recoverable])
        self.assertEqual(selected.profile.plan_id, "recoverable")

    def test_cvar_exposes_severe_recovery_tail(self):
        evaluator = RecoveryAwareEvaluator(cvar_alpha=0.8)
        mild = evaluator.evaluate(profile("mild", 1.0, 0.2, (2.0, 2.0)))
        severe = evaluator.evaluate(profile("severe", 1.0, 0.2, (2.0, 12.0)))
        self.assertGreater(severe.cvar, mild.cvar)


if __name__ == "__main__":
    unittest.main()
