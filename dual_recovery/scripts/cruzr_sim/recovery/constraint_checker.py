"""Safety and executability checks for generated recovery plans."""

from .types import PlanValidation, RecoveryActionType, RecoveryPlan


class RecoveryConstraintChecker:
    def __init__(self, maximum_actions: int = 12, maximum_attempts: int = 3):
        self.maximum_actions = int(maximum_actions)
        self.maximum_attempts = int(maximum_attempts)

    def validate(self, plan: RecoveryPlan, stage: str) -> PlanValidation:
        errors = []
        if not plan.actions:
            errors.append("recovery plan is empty")
        if len(plan.actions) > self.maximum_actions:
            errors.append("recovery plan exceeds the action limit")
        if not 1 <= plan.max_attempts <= self.maximum_attempts:
            errors.append("max_attempts is outside the allowed range")

        actions = [item.action for item in plan.actions]
        if RecoveryActionType.REGRASP in actions:
            regrasp = actions.index(RecoveryActionType.REGRASP)
            required_before = {
                RecoveryActionType.OPEN_GRIPPER,
                RecoveryActionType.RESELECT_GRASP,
                RecoveryActionType.REPLAN_ARM,
            }
            if not required_before.intersection(actions[:regrasp]):
                errors.append("regrasp lacks preparation or replanning")
        if RecoveryActionType.RESUME_TASK in actions:
            resume = actions.index(RecoveryActionType.RESUME_TASK)
            if resume != len(actions) - 1:
                errors.append("resume_task must be the final action")
        if stage in {"lift", "hold", "transport"} and RecoveryActionType.OPEN_GRIPPER in actions:
            open_index = actions.index(RecoveryActionType.OPEN_GRIPPER)
            if not {
                RecoveryActionType.PLACE_BACK,
                RecoveryActionType.MOVE_TO_CLEARANCE,
            }.intersection(actions[:open_index]):
                errors.append("opening the gripper while elevated requires a safe placement")

        for item in plan.actions:
            if item.action == RecoveryActionType.RETREAT:
                distance = float(item.parameters.get("distance", 0.0))
                if not 0.02 <= distance <= 0.30:
                    errors.append("retreat distance must be between 0.02 and 0.30 m")
            if item.action == RecoveryActionType.ADJUST_GRIP_FORCE:
                scale = float(item.parameters.get("scale", 1.0))
                if not 0.5 <= scale <= 1.5:
                    errors.append("grip force scale must be between 0.5 and 1.5")
        return PlanValidation(not errors, tuple(errors))
