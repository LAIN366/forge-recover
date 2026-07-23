"""Paper-oriented metrics for cooperative dual-arm recovery experiments."""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import statistics


@dataclass(frozen=True)
class DualArmEpisodeMetrics:
    run_id: str
    policy: str
    fault: str
    severity: float | None
    seed: int
    exit_code: int
    success: bool
    terminal_stage: str
    simulation_duration: float
    recovery_count: int
    diagnosis_count: int
    diagnosis_latency: float | None
    recovery_duration: float | None
    maximum_synchronization_error: float
    maximum_object_tilt: float
    final_position_error: float
    final_object_tilt: float
    fault_classification_correct: bool | None
    diagnosis_confidence: float | None
    diagnosis_entropy: float | None
    diagnosis_brier_score: float | None
    diagnosis_negative_log_likelihood: float | None
    expected_information_gain: float | None
    recovery_success: bool | None
    replanning_count: int
    active_probe_count: int
    diagnostic_probe_started_count: int
    active_probe_completed_count: int
    active_probe_positive_count: int
    probe_entropy_reduction_sum: float
    graph_rollback_depth: int
    graph_rollback_cost: float
    preserved_task_nodes: int
    expected_recovery_cost: float | None
    cvar: float | None
    llm_request_count: int
    llm_accept_count: int
    llm_fallback_count: int
    llm_mean_latency: float | None
    llm_accepted_candidates: int
    llm_rejected_candidates: int
    recovery_strategy_id: str | None

    def to_dict(self):
        return asdict(self)


def read_jsonl(path):
    records = []
    path = Path(path)
    if not path.is_file():
        return records
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSONL at line {line_number}: {error}"
                ) from error
    return records


def _event_times(records, name):
    return [
        item["payload"].get("simulation_time")
        for item in records
        if item.get("record_type") == "event"
        and item.get("payload", {}).get("name") == name
        and item["payload"].get("simulation_time") is not None
    ]


def extract_dual_arm_metrics(
    path, run_id, policy, fault, seed, exit_code, severity=None,
):
    records = read_jsonl(path)
    summaries = [
        item["payload"] for item in records
        if item.get("record_type") == "summary"
    ]
    summary = summaries[-1] if summaries else {}
    diagnoses = [
        item["payload"] for item in records
        if item.get("record_type") == "diagnosis"
    ]
    recovery_plans = [
        item["payload"] for item in records
        if item.get("record_type") == "recovery_plan"
    ]
    injection_times = _event_times(records, "fault_injected")
    recovery_starts = _event_times(records, "recovery_started")
    recovery_ends = _event_times(records, "recovery_completed")
    event_payloads = [
        item.get("payload", {}) for item in records
        if item.get("record_type") == "event"
    ]
    llm_audits = [
        item for item in event_payloads
        if item.get("name") == "llm_candidate_audit"
    ]
    first_diagnosis_index = next((
        index for index, item in enumerate(records)
        if item.get("record_type") == "diagnosis"
    ), None)
    belief_search_records = (
        records[:first_diagnosis_index]
        if first_diagnosis_index is not None else records
    )
    belief_updates = [
        item.get("payload", {}) for item in belief_search_records
        if item.get("record_type") == "event"
        and item.get("payload", {}).get("name")
        == "temporal_fault_belief_updated"
    ]
    diagnosed_fault = diagnoses[0].get("primary_fault") if diagnoses else None
    expected_faults = {
        "left_contact_loss": "left_grasp_loss",
        "right_contact_loss": "right_grasp_loss",
        "transport_slip": "bimanual_slip",
        "synchronization_delay": "synchronization_error",
        "vision_dropout": "visual_degradation",
        "vision_occlusion": "visual_degradation",
        "dynamic_obstacle": "dynamic_obstacle",
    }
    classification_correct = None
    if fault in expected_faults:
        classification_correct = diagnosed_fault == expected_faults[fault]
    belief_update = belief_updates[-1] if belief_updates else {}
    posterior = belief_update.get("posterior", {})
    diagnosis_confidence = belief_update.get("leading_probability")
    expected_fault = expected_faults.get(fault)
    diagnosis_brier = None
    diagnosis_nll = None
    if expected_fault is not None and posterior:
        probabilities = {
            str(label): max(0.0, min(1.0, float(probability)))
            for label, probability in posterior.items()
        }
        diagnosis_brier = sum(
            (probability - float(label == expected_fault)) ** 2
            for label, probability in probabilities.items()
        )
        true_probability = max(1e-12, probabilities.get(expected_fault, 0.0))
        diagnosis_nll = -math.log(true_probability)
    recovery_records = [
        item for item in event_payloads
        if item.get("name") == "recovery_completed"
    ]
    recovery_success = None
    if recovery_records:
        recovery_success = bool(recovery_records[-1].get("successful", False))
    replanning_names = {
        "regrasp_planned", "fallback_dual_regrasp_planned",
        "dynamic_obstacle_coupled_replan_started",
        "dropped_object_relift_started",
    }
    active_probe_names = {
        "active_probe_started", "dropped_object_search_started",
        "active_reobserve_started",
    }
    probe_starts = [
        item for item in event_payloads
        if item.get("name") == "active_probe_started"
    ]
    probe_completions = [
        item for item in event_payloads
        if item.get("name") == "active_probe_completed"
    ]
    probe_entropy_reductions = [
        max(0.0, float(start["prior_entropy"]) - float(completed["entropy"]))
        for start, completed in zip(probe_starts, probe_completions)
        if start.get("prior_entropy") is not None
        and completed.get("entropy") is not None
    ]
    diagnosis_latency = None
    if diagnoses and injection_times:
        diagnosis_latency = max(
            0.0, float(diagnoses[0]["timestamp"]) - float(injection_times[0])
        )
    recovery_duration = None
    if recovery_starts and recovery_ends:
        recovery_duration = max(
            0.0, float(recovery_ends[0]) - float(recovery_starts[0])
        )
    return DualArmEpisodeMetrics(
        run_id=str(run_id),
        policy=str(policy),
        fault=str(fault),
        severity=(float(severity) if severity is not None else None),
        seed=int(seed),
        exit_code=int(exit_code),
        success=bool(summary.get("success", False)),
        terminal_stage=str(summary.get("terminal_stage", "unknown")),
        simulation_duration=float(summary.get("simulation_duration", 0.0)),
        recovery_count=int(summary.get("recovery_count", 0)),
        diagnosis_count=int(summary.get("diagnosis_count", len(diagnoses))),
        diagnosis_latency=diagnosis_latency,
        recovery_duration=recovery_duration,
        maximum_synchronization_error=float(
            summary.get("maximum_synchronization_error", 0.0)
        ),
        maximum_object_tilt=float(summary.get("maximum_object_tilt", 0.0)),
        final_position_error=float(summary.get("final_position_error", math.inf)),
        final_object_tilt=float(summary.get("final_object_tilt", math.inf)),
        fault_classification_correct=classification_correct,
        diagnosis_confidence=(
            float(diagnosis_confidence)
            if diagnosis_confidence is not None else None
        ),
        diagnosis_entropy=(
            float(belief_update["entropy"])
            if belief_update.get("entropy") is not None else None
        ),
        diagnosis_brier_score=diagnosis_brier,
        diagnosis_negative_log_likelihood=diagnosis_nll,
        expected_information_gain=(
            float(belief_update["expected_information_gain"])
            if belief_update.get("expected_information_gain") is not None else None
        ),
        recovery_success=recovery_success,
        replanning_count=sum(
            item.get("name") in replanning_names for item in event_payloads
        ),
        active_probe_count=sum(
            item.get("name") in active_probe_names for item in event_payloads
        ),
        diagnostic_probe_started_count=len(probe_starts),
        active_probe_completed_count=len(probe_completions),
        active_probe_positive_count=sum(
            bool(item.get("positive")) for item in probe_completions
        ),
        probe_entropy_reduction_sum=sum(probe_entropy_reductions),
        graph_rollback_depth=int(summary.get("graph_rollback_depth", 0)),
        graph_rollback_cost=float(summary.get("graph_rollback_cost", 0.0)),
        preserved_task_nodes=int(summary.get("preserved_task_nodes", 0)),
        expected_recovery_cost=(
            float(summary["expected_recovery_cost"])
            if summary.get("expected_recovery_cost") is not None else None
        ),
        cvar=(
            float(summary["cvar"])
            if summary.get("cvar") is not None else None
        ),
        llm_request_count=len(llm_audits),
        llm_accept_count=sum(item.get("status") == "accepted" for item in llm_audits),
        llm_fallback_count=sum(item.get("status") == "fallback" for item in llm_audits),
        llm_mean_latency=(
            statistics.fmean(float(item.get("latency_seconds", 0.0)) for item in llm_audits)
            if llm_audits else None
        ),
        llm_accepted_candidates=sum(int(item.get("accepted_count", 0)) for item in llm_audits),
        llm_rejected_candidates=sum(int(item.get("rejected_count", 0)) for item in llm_audits),
        recovery_strategy_id=(
            str(recovery_plans[-1].get("strategy_id"))
            if recovery_plans and recovery_plans[-1].get("strategy_id") else None
        ),
    )


def _wilson_interval(successes, total, z=1.959963984540054):
    if total <= 0:
        return 0.0, 0.0
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / total
        + z * z / (4.0 * total * total)
    ) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def _mean_std(values):
    if not values:
        return None, None
    return statistics.fmean(values), (
        statistics.stdev(values) if len(values) > 1 else 0.0
    )


def _upper_cvar(values, alpha=0.90):
    if not values:
        return None
    ordered = sorted(values)
    tail_count = max(1, math.ceil((1.0 - alpha) * len(ordered)))
    return statistics.fmean(ordered[-tail_count:])


def _expected_calibration_error(items, bins=10):
    """Compute confidence calibration error for diagnosed fault labels."""
    calibrated = [
        item for item in items
        if item.diagnosis_confidence is not None
        and item.fault_classification_correct is not None
    ]
    if not calibrated:
        return None
    error = 0.0
    for index in range(int(bins)):
        lower, upper = index / bins, (index + 1) / bins
        bucket = [
            item for item in calibrated
            if lower <= item.diagnosis_confidence
            and (item.diagnosis_confidence < upper or index == bins - 1)
        ]
        if not bucket:
            continue
        confidence = statistics.fmean(item.diagnosis_confidence for item in bucket)
        accuracy = statistics.fmean(
            float(item.fault_classification_correct) for item in bucket
        )
        error += len(bucket) / len(calibrated) * abs(accuracy - confidence)
    return error


def aggregate_dual_arm_metrics(episodes):
    groups = {}
    for episode in episodes:
        groups.setdefault(
            (episode.policy, episode.fault, episode.severity), []
        ).append(episode)
    rows = []
    for (policy, fault, severity), items in sorted(
        groups.items(), key=lambda item: (
            item[0][0], item[0][1],
            float("-inf") if item[0][2] is None else item[0][2],
        )
    ):
        successes = sum(item.success for item in items)
        ci_low, ci_high = _wilson_interval(successes, len(items))
        durations = [item.simulation_duration for item in items]
        sync_errors = [item.maximum_synchronization_error for item in items]
        pose_errors = [
            item.final_position_error for item in items
            if math.isfinite(item.final_position_error)
        ]
        final_tilts = [
            item.final_object_tilt for item in items
            if math.isfinite(item.final_object_tilt)
        ]
        diagnosis_latencies = [
            item.diagnosis_latency for item in items
            if item.diagnosis_latency is not None
        ]
        diagnosis_entropies = [
            item.diagnosis_entropy for item in items
            if item.diagnosis_entropy is not None
        ]
        diagnosis_brier_scores = [
            item.diagnosis_brier_score for item in items
            if item.diagnosis_brier_score is not None
        ]
        diagnosis_nlls = [
            item.diagnosis_negative_log_likelihood for item in items
            if item.diagnosis_negative_log_likelihood is not None
        ]
        information_gains = [
            item.expected_information_gain for item in items
            if item.expected_information_gain is not None
        ]
        duration_mean, duration_std = _mean_std(durations)
        sync_mean, sync_std = _mean_std(sync_errors)
        pose_mean, pose_std = _mean_std(pose_errors)
        diagnosis_mean, diagnosis_std = _mean_std(diagnosis_latencies)
        llm_latencies = [
            item.llm_mean_latency for item in items
            if item.llm_mean_latency is not None
        ]
        tilt_mean, tilt_std = _mean_std(final_tilts)
        rows.append({
            "policy": policy,
            "fault": fault,
            "severity": severity,
            "episodes": len(items),
            "success_rate": successes / len(items),
            "success_ci95_low": ci_low,
            "success_ci95_high": ci_high,
            "mean_duration": duration_mean,
            "std_duration": duration_std,
            "duration_cvar90": _upper_cvar(durations),
            "mean_diagnosis_latency": diagnosis_mean,
            "std_diagnosis_latency": diagnosis_std,
            "diagnosis_ece10": _expected_calibration_error(items),
            "mean_diagnosis_entropy": (
                statistics.fmean(diagnosis_entropies)
                if diagnosis_entropies else None
            ),
            "mean_diagnosis_brier_score": (
                statistics.fmean(diagnosis_brier_scores)
                if diagnosis_brier_scores else None
            ),
            "mean_diagnosis_negative_log_likelihood": (
                statistics.fmean(diagnosis_nlls) if diagnosis_nlls else None
            ),
            "mean_expected_information_gain": (
                statistics.fmean(information_gains) if information_gains else None
            ),
            "mean_maximum_sync_error": sync_mean,
            "std_maximum_sync_error": sync_std,
            "sync_error_cvar90": _upper_cvar(sync_errors),
            "mean_final_position_error": pose_mean,
            "std_final_position_error": pose_std,
            "mean_final_object_tilt": tilt_mean,
            "std_final_object_tilt": tilt_std,
            "mean_recovery_count": statistics.fmean(
                item.recovery_count for item in items
            ),
            "fault_classification_accuracy": (
                statistics.fmean(
                    float(item.fault_classification_correct)
                    for item in items
                    if item.fault_classification_correct is not None
                )
                if any(
                    item.fault_classification_correct is not None
                    for item in items
                ) else None
            ),
            "recovery_success_rate": (
                statistics.fmean(
                    float(item.recovery_success)
                    for item in items
                    if item.recovery_success is not None
                )
                if any(item.recovery_success is not None for item in items)
                else None
            ),
            "mean_replanning_count": statistics.fmean(
                item.replanning_count for item in items
            ),
            "mean_active_probe_count": statistics.fmean(
                item.active_probe_count for item in items
            ),
            "active_probe_completion_rate": (
                sum(item.active_probe_completed_count for item in items)
                / max(1, sum(item.diagnostic_probe_started_count for item in items))
            ),
            "active_probe_positive_rate": (
                sum(item.active_probe_positive_count for item in items)
                / max(1, sum(item.active_probe_completed_count for item in items))
            ),
            "mean_probe_entropy_reduction": (
                sum(item.probe_entropy_reduction_sum for item in items)
                / max(1, sum(item.active_probe_completed_count for item in items))
            ),
            "mean_graph_rollback_depth": statistics.fmean(
                item.graph_rollback_depth for item in items
            ),
            "mean_graph_rollback_cost": statistics.fmean(
                item.graph_rollback_cost for item in items
            ),
            "mean_preserved_task_nodes": statistics.fmean(
                item.preserved_task_nodes for item in items
            ),
            "llm_request_count": sum(item.llm_request_count for item in items),
            "llm_accept_rate": (
                sum(item.llm_accept_count for item in items)
                / max(1, sum(item.llm_request_count for item in items))
            ),
            "llm_fallback_rate": (
                sum(item.llm_fallback_count for item in items)
                / max(1, sum(item.llm_request_count for item in items))
            ),
            "mean_llm_latency": (
                statistics.fmean(llm_latencies) if llm_latencies else None
            ),
            "mean_llm_accepted_candidates": statistics.fmean(
                item.llm_accepted_candidates for item in items
            ),
            "mean_llm_rejected_candidates": statistics.fmean(
                item.llm_rejected_candidates for item in items
            ),
            "recovery_strategy_counts": {
                strategy: sum(item.recovery_strategy_id == strategy for item in items)
                for strategy in sorted({
                    item.recovery_strategy_id for item in items
                    if item.recovery_strategy_id is not None
                })
            },
        })
    return rows
