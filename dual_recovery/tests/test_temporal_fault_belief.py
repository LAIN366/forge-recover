"""Tests for stage-conditioned temporal diagnosis and active probing."""

import pytest

from cruzr_sim.diagnosis.dual_arm import (
    DualArmDiagnosticEvidence,
    DualArmDiagnosisReport,
    DualArmFaultType,
)
from cruzr_sim.diagnosis.temporal_belief import StageConditionedFaultFilter
from cruzr_sim.tasks.diagnostic_probes import evaluate_probe_outcome
from cruzr_sim.diagnosis.dual_arm import DualArmObservation


def report(stage, fault, confidence=0.8, probes=()):
    evidence = DualArmDiagnosticEvidence(fault, confidence, ("test",), probes)
    return DualArmDiagnosisReport(
        1.0, stage, True, fault, confidence, (evidence,),
        probes[0] if probes else None,
    )


def test_stage_prior_changes_posterior_for_identical_evidence():
    transport = StageConditionedFaultFilter()
    grasp = StageConditionedFaultFilter()
    transport_estimate = transport.update(report(
        "cooperative_transport", DualArmFaultType.SYNCHRONIZATION_ERROR,
    ))
    grasp_estimate = grasp.update(report(
        "dual_grasp", DualArmFaultType.SYNCHRONIZATION_ERROR,
    ))
    assert (
        transport_estimate.posterior[DualArmFaultType.SYNCHRONIZATION_ERROR]
        > grasp_estimate.posterior[DualArmFaultType.SYNCHRONIZATION_ERROR]
    )


def test_persistent_evidence_accumulates_temporal_fault_probability():
    fault_filter = StageConditionedFaultFilter()
    item = report(
        "cooperative_transport",
        DualArmFaultType.SYNCHRONIZATION_ERROR,
        probes=("pause_and_resample_sync",),
    )
    first = fault_filter.update(item)
    second = fault_filter.update(item)
    assert second.posterior[DualArmFaultType.SYNCHRONIZATION_ERROR] > first.posterior[
        DualArmFaultType.SYNCHRONIZATION_ERROR
    ]
    assert sum(second.posterior.values()) == pytest.approx(1.0)


def test_information_gain_selects_a_discriminating_probe():
    fault_filter = StageConditionedFaultFilter()
    fault_filter.posterior = {
        fault: 0.0 for fault in DualArmFaultType
    }
    fault_filter.posterior[DualArmFaultType.BIMANUAL_SLIP] = 0.45
    fault_filter.posterior[DualArmFaultType.SYNCHRONIZATION_ERROR] = 0.40
    fault_filter.posterior[DualArmFaultType.NORMAL] = 0.15
    probe, gain = fault_filter.select_probe({
        "pause_and_hold", "pause_and_resample_sync",
    })
    assert probe in {"pause_and_hold", "pause_and_resample_sync"}
    assert gain > 0.0


def test_normal_evidence_reduces_a_transient_fault_belief():
    fault_filter = StageConditionedFaultFilter()
    anomalous = report(
        "cooperative_transport", DualArmFaultType.SYNCHRONIZATION_ERROR,
    )
    first = fault_filter.update(anomalous)
    normal = DualArmDiagnosisReport(
        1.1, "cooperative_transport", False, DualArmFaultType.NORMAL,
        1.0, (), None,
    )
    second = fault_filter.update(normal)
    assert second.posterior[DualArmFaultType.SYNCHRONIZATION_ERROR] < first.posterior[
        DualArmFaultType.SYNCHRONIZATION_ERROR
    ]


def test_executed_probe_updates_posterior_and_is_not_repeated():
    fault_filter = StageConditionedFaultFilter()
    fault_filter.posterior = {fault: 0.0 for fault in DualArmFaultType}
    fault_filter.posterior[DualArmFaultType.BIMANUAL_SLIP] = 0.50
    fault_filter.posterior[DualArmFaultType.SYNCHRONIZATION_ERROR] = 0.40
    fault_filter.posterior[DualArmFaultType.NORMAL] = 0.10
    before = fault_filter.posterior[DualArmFaultType.BIMANUAL_SLIP]
    estimate = fault_filter.incorporate_probe_result("pause_and_hold", True)
    assert estimate.posterior[DualArmFaultType.BIMANUAL_SLIP] > before
    probe, _ = fault_filter.select_probe({"pause_and_hold"})
    assert probe is None


def test_contact_probe_uses_persistent_cross_arm_evidence():
    values = dict(
        timestamp=1.0, stage="cooperative_transport",
        object_position=(0.0, 0.0, 0.8), object_rpy=(0.0, 0.0, 0.0),
        left_tool_position=(-0.1, 0.0, 0.8),
        right_tool_position=(0.1, 0.0, 0.8),
        left_contacts=(False, False), right_contacts=(True, True),
        left_force=0.0, right_force=2.0,
    )
    before = DualArmObservation(**values)
    after = DualArmObservation(**{**values, "timestamp": 1.12})
    assert evaluate_probe_outcome("pause_and_contact_test", before, after)
