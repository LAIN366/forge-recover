"""Metrics extraction from append-only manipulation experiment logs."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import statistics


@dataclass(frozen=True)
class EpisodeMetrics:
    run_id: str
    policy: str
    fault: str
    seed: int
    exit_code: int
    success: bool
    terminal_stage: str
    simulation_duration: float
    cube_height_gain: float
    recovery_count: int
    diagnosis_count: int
    recovery_plan_count: int
    active_probe_count: int
    positive_probe_count: int
    first_diagnosis: str
    confirmed_fault: str
    diagnosis_latency: float | None

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
                raise ValueError(f"invalid JSONL at line {line_number}: {error}") from error
    return records


def extract_episode_metrics(path, run_id, policy, fault, seed, exit_code):
    records = read_jsonl(path)
    summaries = [item["payload"] for item in records if item["record_type"] == "summary"]
    summary = summaries[-1] if summaries else {}
    diagnoses = [item["payload"] for item in records if item["record_type"] == "diagnosis"]
    plans = [item["payload"] for item in records if item["record_type"] == "recovery_plan"]
    probe_events = [
        item["payload"] for item in records
        if item["record_type"] == "event"
        and item["payload"].get("name") == "active_probe_completed"
    ]
    injection_times = [
        item["payload"].get("simulation_time")
        for item in records
        if item["record_type"] == "event"
        and item["payload"].get("name") == "fault_injected"
    ]
    first_diagnosis_time = diagnoses[0].get("timestamp") if diagnoses else None
    injection_time = injection_times[0] if injection_times else None
    latency = (
        max(0.0, float(first_diagnosis_time) - float(injection_time))
        if first_diagnosis_time is not None and injection_time is not None else None
    )
    return EpisodeMetrics(
        run_id=run_id,
        policy=policy,
        fault=fault,
        seed=int(seed),
        exit_code=int(exit_code),
        success=bool(summary.get("success", False)),
        terminal_stage=str(summary.get("terminal_stage", "unknown")),
        simulation_duration=float(summary.get("simulation_duration", 0.0)),
        cube_height_gain=float(summary.get("cube_height_gain", 0.0)),
        recovery_count=int(summary.get("recovery_count", len(plans))),
        diagnosis_count=len(diagnoses),
        recovery_plan_count=len(plans),
        active_probe_count=len(probe_events),
        positive_probe_count=sum(bool(item.get("positive")) for item in probe_events),
        first_diagnosis=(
            str(diagnoses[0].get("primary_fault", "unknown")) if diagnoses else "none"
        ),
        confirmed_fault=(
            str(plans[0].get("fault", "unknown")) if plans else "none"
        ),
        diagnosis_latency=latency,
    )


def aggregate_episode_metrics(episodes):
    """Aggregate paired runs by policy and fault for paper-ready tables."""
    groups = {}
    for episode in episodes:
        groups.setdefault((episode.policy, episode.fault), []).append(episode)
    rows = []
    for (policy, fault), items in sorted(groups.items()):
        latencies = [
            item.diagnosis_latency for item in items
            if item.diagnosis_latency is not None
        ]
        rows.append({
            "policy": policy,
            "fault": fault,
            "episodes": len(items),
            "success_rate": statistics.fmean(item.success for item in items),
            "detection_rate": statistics.fmean(
                item.first_diagnosis != "none" for item in items
            ),
            "confirmed_recovery_rate": statistics.fmean(
                item.confirmed_fault != "none" for item in items
            ),
            "mean_simulation_duration": statistics.fmean(
                item.simulation_duration for item in items
            ),
            "mean_recovery_count": statistics.fmean(
                item.recovery_count for item in items
            ),
            "mean_active_probe_count": statistics.fmean(
                item.active_probe_count for item in items
            ),
            "mean_diagnosis_latency": (
                statistics.fmean(latencies) if latencies else None
            ),
        })
    return rows
