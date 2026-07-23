"""Non-blocking execution policy for pickup-task recovery plans."""

from dataclasses import dataclass

from cruzr_sim.diagnosis.types import FaultType
from cruzr_sim.recovery.types import RecoveryPlan


@dataclass(frozen=True)
class GraspRecoveryDirective:
    wait_for_grounded_object: bool
    reset_stage: str = "clearance"
    reopen_gripper: bool = True
    enable_object_contacts: bool = True


def compile_grasp_recovery(plan: RecoveryPlan) -> GraspRecoveryDirective:
    """Map a validated high-level plan to the pickup state machine."""
    elevated_faults = {
        FaultType.GRASP_SLIP,
        FaultType.UNSTABLE_GRASP,
    }
    return GraspRecoveryDirective(
        wait_for_grounded_object=plan.fault in elevated_faults,
        reset_stage="clearance",
    )
