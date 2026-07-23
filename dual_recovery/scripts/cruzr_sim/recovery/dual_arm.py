"""Task-graph recovery policies for diagnosed dual-arm failures."""

from dataclasses import dataclass

from cruzr_sim.diagnosis.dual_arm import DualArmFaultType
from .experience_graph import ContextualRecoveryExperienceGraph


@dataclass(frozen=True)
class DualArmRecoveryPlan:
    fault: DualArmFaultType
    recovery_node: str
    actions: tuple[str, ...]
    retry_node: str
    rationale: str
    strategy_id: str = "fixed_case"
    estimated_success: float = 0.0
    estimated_cvar: float = 0.0
    selection_score: float = 0.0


class DualArmRecoveryPlanner:
    """Map causal diagnoses to bounded task-graph recovery transactions."""

    def __init__(self, experience_graph=None, candidate_generator=None):
        self.experience_graph = (
            experience_graph or ContextualRecoveryExperienceGraph()
        )
        self.candidate_generator = candidate_generator
        self.last_llm_audit = None

    def plan(self, report, failed_node, context=None):
        if not report.anomalous:
            raise ValueError("dual-arm recovery requires an anomalous report")
        fault = report.primary_fault
        candidates = self._candidates(fault, failed_node)
        if self.candidate_generator is not None:
            candidates = self.candidate_generator.propose(
                report, failed_node, context, candidates
            )
            self.last_llm_audit = self.candidate_generator.last_audit
        if context is None:
            return candidates[0][0]
        preferred = set(
            self.candidate_generator.last_selected_strategy_ids
            if self.candidate_generator is not None else ()
        )
        preference_bonus = (
            self.candidate_generator.preference_bonus
            if self.candidate_generator is not None else 0.0
        )
        selected, _ = self.experience_graph.select(
            context,
            tuple((
                plan.strategy_id,
                max(0.0, nominal_cost - (
                    preference_bonus if plan.strategy_id in preferred else 0.0
                )),
            ) for plan, nominal_cost in candidates),
        )
        by_id = {plan.strategy_id: plan for plan, _ in candidates}
        plan = by_id[selected.strategy_id]
        return DualArmRecoveryPlan(
            plan.fault, plan.recovery_node, plan.actions, plan.retry_node,
            plan.rationale, plan.strategy_id, selected.success_probability,
            selected.cost_cvar, selected.total_score,
        )

    def record_outcome(self, context, plan, successful, cost):
        return self.experience_graph.record(
            context, plan.strategy_id, successful, cost
        )

    @staticmethod
    def _candidates(fault, failed_node):
        if fault == DualArmFaultType.VISUAL_DEGRADATION:
            primary = DualArmRecoveryPlan(
                fault, "active_reobserve", ("stop", "reobserve"),
                failed_node, "restore pose belief before resuming",
                "active_reobserve",
            )
            alternative = DualArmRecoveryPlan(
                fault, "active_reobserve", ("stop", "change_view", "reobserve"),
                failed_node, "change viewpoint before restoring pose belief",
                "viewpoint_reobserve",
            )
            return ((primary, 0.8), (alternative, 1.1))
        if fault in {
            DualArmFaultType.LEFT_GRASP_LOSS,
            DualArmFaultType.RIGHT_GRASP_LOSS,
        }:
            side = "left" if fault == DualArmFaultType.LEFT_GRASP_LOSS else "right"
            primary = DualArmRecoveryPlan(
                fault,
                "stabilize_object",
                ("stop", "support_hold", f"{side}_regrasp", "resynchronize"),
                failed_node,
                f"retain support-arm contact while restoring the {side} grasp",
                "support_hold_regrasp",
            )
            alternative = DualArmRecoveryPlan(
                fault, "stabilize_object",
                ("stop", "place_object", "dual_regrasp", "resynchronize"),
                failed_node, "set down before a conservative dual regrasp",
                "setdown_dual_regrasp",
            )
            return ((primary, 1.4), (alternative, 2.2))
        if fault in {
            DualArmFaultType.BIMANUAL_SLIP,
            DualArmFaultType.SYNCHRONIZATION_ERROR,
            DualArmFaultType.DYNAMIC_OBSTACLE,
        }:
            definitions = {
                DualArmFaultType.BIMANUAL_SLIP: (
                    ("visual_search_regrasp", ("stop", "place_object", "dual_regrasp", "resynchronize"), 2.0),
                    ("safe_setdown_regrasp", ("stop", "support_setdown", "reobserve", "dual_regrasp"), 2.6),
                ),
                DualArmFaultType.SYNCHRONIZATION_ERROR: (
                    ("pause_resynchronize", ("pause_leading_arm", "resynchronize"), 0.9),
                    ("rollback_resynchronize", ("stop", "rollback", "resynchronize"), 1.5),
                ),
                DualArmFaultType.DYNAMIC_OBSTACLE: (
                    ("retreat_replan", ("stop", "retreat", "replan_coupled_path"), 1.8),
                    ("role_reassign_replan", ("stop", "reassign_roles", "replan_coupled_path"), 2.1),
                ),
            }[fault]
            return tuple((DualArmRecoveryPlan(
                fault, "stabilize_object", actions, failed_node,
                "restore cooperative constraints before task continuation",
                strategy_id,
            ), cost) for strategy_id, actions, cost in definitions)
        primary = DualArmRecoveryPlan(
            fault,
            "reassign_roles",
            ("stop", "recompute_capability", "reassign_roles", "replan"),
            failed_node,
            "change the kinematic assignment after planning infeasibility",
            "capability_role_reassignment",
        )
        alternative = DualArmRecoveryPlan(
            fault, "reassign_roles", ("stop", "retreat", "replan"),
            failed_node, "retreat to a known feasible configuration",
            "clearance_retreat_replan",
        )
        return ((primary, 1.6), (alternative, 2.0))
