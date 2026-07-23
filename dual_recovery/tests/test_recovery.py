"""Unit tests for retrieval and safety validation of recovery plans."""

import json
import unittest

from cruzr_sim.diagnosis.types import (
    DiagnosisReport,
    DiagnosticEvidence,
    FaultType,
    ManipulationObservation,
)
from cruzr_sim.recovery import (
    RecoveryAction,
    RecoveryActionType,
    RecoveryConstraintChecker,
    RecoveryPlan,
    RecoveryPlanner,
    RecoveryProposalError,
    RecoveryProposalParser,
    RecoveryVerifier,
)


def report(fault, confidence=0.9, stage="lift"):
    evidence = DiagnosticEvidence(fault, confidence, ("test evidence",))
    return DiagnosisReport(1.0, stage, True, fault, confidence, (evidence,))


class RecoveryPlanningTest(unittest.TestCase):
    def test_slip_plan_is_retrieved_and_validated(self):
        plan = RecoveryPlanner().plan(report(FaultType.GRASP_SLIP), "lift")
        self.assertEqual(plan.fault, FaultType.GRASP_SLIP)
        self.assertIn(RecoveryActionType.PLACE_BACK, [item.action for item in plan.actions])
        self.assertTrue(RecoveryConstraintChecker().validate(plan, "lift").valid)

    def test_unsafe_high_open_is_rejected(self):
        plan = RecoveryPlan(
            "unsafe", FaultType.GRASP_SLIP, "test", 0.9,
            (RecoveryAction(RecoveryActionType.OPEN_GRIPPER),),
        )
        validation = RecoveryConstraintChecker().validate(plan, "lift")
        self.assertFalse(validation.valid)
        self.assertTrue(any("safe placement" in error for error in validation.errors))

    def test_external_invalid_candidate_falls_back_to_case(self):
        candidate = (RecoveryAction(RecoveryActionType.OPEN_GRIPPER),)
        plan = RecoveryPlanner().plan(
            report(FaultType.GRASP_SLIP), "lift", candidate_actions=candidate
        )
        self.assertTrue(plan.source.startswith("case:"))
        self.assertTrue(RecoveryConstraintChecker().validate(plan, "lift").valid)

    def test_llm_proposal_parser_accepts_only_structured_actions(self):
        payload = {"actions": [{
            "action": "stop", "parameters": {}, "rationale": "freeze motion",
        }]}
        actions = RecoveryProposalParser().parse(json.dumps(payload))
        self.assertEqual(actions[0].action, RecoveryActionType.STOP)
        with self.assertRaises(RecoveryProposalError):
            RecoveryProposalParser().parse("First, stop the robot")

    def test_sensor_recovery_requires_valid_stream(self):
        sample = ManipulationObservation(
            timestamp=1.0,
            stage="clearance",
            object_position=(0.0, 0.0, 0.8),
            tool_position=(0.0, 0.0, 0.9),
            sensor_valid=True,
        )
        result = RecoveryVerifier().verify(FaultType.SENSOR_FAULT, sample, 0.8)
        self.assertTrue(result.successful)

    def test_target_recovery_accepts_consumed_precontact_reference(self):
        sample = ManipulationObservation(
            timestamp=4.0,
            stage="hold",
            object_position=(0.05, 0.0, 0.91),
            tool_position=(0.05, 0.0, 0.94),
            left_contact=True,
            right_contact=True,
            object_vertical_velocity=0.0,
            target_position=None,
            sensor_valid=True,
        )
        result = RecoveryVerifier().verify(
            FaultType.TARGET_DISPLACEMENT, sample, 0.8
        )
        self.assertTrue(result.successful)


if __name__ == "__main__":
    unittest.main()
