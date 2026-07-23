"""Outcome checks for recovery execution."""

from dataclasses import dataclass

from cruzr_sim.diagnosis.types import FaultType, ManipulationObservation


@dataclass(frozen=True)
class RecoveryVerification:
    successful: bool
    reason: str


class RecoveryVerifier:
    def verify(
        self,
        fault: FaultType,
        observation: ManipulationObservation,
        initial_object_height: float,
    ) -> RecoveryVerification:
        if observation.collision or not observation.sensor_valid:
            return RecoveryVerification(False, "unsafe or invalid state remains")
        if fault in {FaultType.MISSED_GRASP, FaultType.GRASP_SLIP, FaultType.UNSTABLE_GRASP}:
            lifted = observation.object_position[2] - initial_object_height >= 0.05
            stable = observation.both_contacts and observation.object_vertical_velocity >= -0.01
            return RecoveryVerification(
                lifted and stable,
                "stable bilateral grasp restored" if lifted and stable
                else "stable grasp has not been restored",
            )
        if fault == FaultType.TARGET_DISPLACEMENT:
            lifted = observation.object_position[2] - initial_object_height >= 0.05
            stable = observation.both_contacts and observation.object_vertical_velocity >= -0.01
            return RecoveryVerification(
                lifted and stable,
                "target refreshed and stable grasp completed" if lifted and stable
                else "task did not complete after target refresh",
            )
        if fault in {FaultType.IK_FAILURE, FaultType.PLANNING_FAILURE}:
            success = observation.ik_success and observation.planning_success
            return RecoveryVerification(success, "feasible motion restored" if success else "motion remains infeasible")
        if fault == FaultType.COLLISION:
            return RecoveryVerification(not observation.collision, "collision cleared")
        if fault == FaultType.SENSOR_FAULT:
            return RecoveryVerification(
                observation.sensor_valid,
                "sensor stream restored" if observation.sensor_valid
                else "sensor stream remains invalid",
            )
        return RecoveryVerification(False, "no verifier is defined for this fault")
