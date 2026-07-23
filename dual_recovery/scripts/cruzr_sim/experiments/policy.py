"""Experiment policies used for controlled ablation studies."""

from enum import Enum


class ExperimentPolicy(str, Enum):
    NO_RECOVERY = "no_recovery"
    FIXED_RULE = "fixed_rule"
    DIAGNOSIS_ONLY = "diagnosis_only"
    ACTIVE_CASE = "active_case"
    FULL = "full"
    B0_FIXED_FSM = "b0_fixed_fsm"
    B1_TASK_GRAPH = "b1_task_graph"
    B2_BELIEF_GRAPH = "b2_belief_graph"
    OURS = "ours"
    OURS_LLM = "ours_llm"

    @property
    def diagnosis_enabled(self):
        return self in {
            ExperimentPolicy.DIAGNOSIS_ONLY,
            ExperimentPolicy.ACTIVE_CASE,
            ExperimentPolicy.FULL,
            ExperimentPolicy.B2_BELIEF_GRAPH,
            ExperimentPolicy.OURS,
            ExperimentPolicy.OURS_LLM,
        }

    @property
    def learned_recovery_enabled(self):
        return self in {
            ExperimentPolicy.ACTIVE_CASE,
            ExperimentPolicy.FULL,
            ExperimentPolicy.OURS,
            ExperimentPolicy.OURS_LLM,
        }

    @property
    def task_graph_enabled(self):
        return self not in {
            ExperimentPolicy.NO_RECOVERY,
            ExperimentPolicy.FIXED_RULE,
            ExperimentPolicy.B0_FIXED_FSM,
        }

    @property
    def belief_enabled(self):
        return self in {
            ExperimentPolicy.B2_BELIEF_GRAPH,
            ExperimentPolicy.OURS,
            ExperimentPolicy.OURS_LLM,
            ExperimentPolicy.FULL,
        }

    @property
    def active_diagnosis_enabled(self):
        return self in {
            ExperimentPolicy.B2_BELIEF_GRAPH,
            ExperimentPolicy.OURS,
            ExperimentPolicy.OURS_LLM,
            ExperimentPolicy.ACTIVE_CASE,
            ExperimentPolicy.FULL,
        }

    @property
    def recovery_aware_cost_enabled(self):
        return self in {
            ExperimentPolicy.OURS, ExperimentPolicy.OURS_LLM,
            ExperimentPolicy.FULL,
        }

    @property
    def role_switch_enabled(self):
        return self in {
            ExperimentPolicy.OURS, ExperimentPolicy.OURS_LLM,
            ExperimentPolicy.FULL,
        }
