"""Retrieval-augmented, constraint-checked recovery planning."""

from dataclasses import replace
import uuid

from cruzr_sim.diagnosis.types import DiagnosisReport, FaultType

from .case_library import RecoveryCaseLibrary
from .constraint_checker import RecoveryConstraintChecker
from .types import RecoveryAction, RecoveryActionType, RecoveryPlan


class RecoveryPlanningError(RuntimeError):
    pass


class RecoveryPlanner:
    """Generate an executable baseline plan from diagnosed fault and cases.

    An LLM adapter can later propose actions through ``candidate_actions``. The
    same deterministic constraint checker remains the final safety gate.
    """

    def __init__(self, case_library=None, constraint_checker=None):
        self.case_library = case_library or RecoveryCaseLibrary()
        self.constraint_checker = constraint_checker or RecoveryConstraintChecker()

    def plan(
        self,
        report: DiagnosisReport,
        stage: str,
        attempt: int = 1,
        candidate_actions: tuple[RecoveryAction, ...] | None = None,
    ) -> RecoveryPlan:
        if not report.anomalous or report.primary_fault == FaultType.NORMAL:
            raise RecoveryPlanningError("recovery requested without a diagnosed anomaly")
        if candidate_actions is None:
            cases = self.case_library.retrieve(report.primary_fault, stage)
            if cases:
                actions = cases[0].actions
                source = f"case:{cases[0].case_id}"
            else:
                actions = (
                    RecoveryAction(RecoveryActionType.STOP),
                    RecoveryAction(RecoveryActionType.REOBSERVE),
                    RecoveryAction(RecoveryActionType.SAFE_ABORT, {
                        "reason": "no validated recovery case",
                    }),
                )
                source = "safe-fallback"
        else:
            actions = tuple(candidate_actions)
            source = "external-candidate"

        plan = RecoveryPlan(
            plan_id=f"recovery-{uuid.uuid4().hex[:10]}",
            fault=report.primary_fault,
            source=source,
            confidence=report.confidence,
            actions=actions,
            max_attempts=max(1, min(3, 3 - attempt + 1)),
        )
        validation = self.constraint_checker.validate(plan, stage)
        if not validation.valid:
            if candidate_actions is not None:
                return self.plan(report, stage, attempt, candidate_actions=None)
            raise RecoveryPlanningError("; ".join(validation.errors))
        return plan

    def adapt_after_failure(self, plan: RecoveryPlan, reason: str) -> RecoveryPlan:
        """Return a conservative alternative after a recovery attempt fails."""
        fallback = (
            RecoveryAction(RecoveryActionType.STOP, rationale=reason),
            RecoveryAction(RecoveryActionType.MOVE_TO_CLEARANCE),
            RecoveryAction(RecoveryActionType.REOBSERVE, {"target": "workspace"}),
            RecoveryAction(RecoveryActionType.SAFE_ABORT, {"reason": reason}),
        )
        return replace(
            plan,
            plan_id=f"recovery-{uuid.uuid4().hex[:10]}",
            source=f"fallback-after:{plan.plan_id}",
            actions=fallback,
            max_attempts=1,
        )
