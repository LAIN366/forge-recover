"""Paired factorial experiment design and effect-size utilities."""

from dataclasses import dataclass
import math
import statistics

from cruzr_sim.experiments.policy import ExperimentPolicy


PAPER_POLICIES = (
    ExperimentPolicy.B0_FIXED_FSM,
    ExperimentPolicy.B1_TASK_GRAPH,
    ExperimentPolicy.B2_BELIEF_GRAPH,
    ExperimentPolicy.OURS,
)


@dataclass(frozen=True)
class ExperimentCell:
    policy: str
    fault: str
    severity: float
    seed: int


def build_paired_factorial_cells(
    faults,
    severities=(0.5, 1.0, 1.5),
    seeds=range(30),
    policies=PAPER_POLICIES,
):
    """Generate matched seeds across every method-factor-severity cell."""
    return tuple(
        ExperimentCell(
            policy=(policy.value if isinstance(policy, ExperimentPolicy) else str(policy)),
            fault=str(fault),
            severity=float(severity),
            seed=int(seed),
        )
        for fault in faults
        for severity in severities
        for seed in seeds
        for policy in policies
    )


def paired_cohens_d(baseline, proposed):
    """Cohen's dz for paired per-seed measurements."""
    if len(baseline) != len(proposed) or not baseline:
        raise ValueError("paired samples must be non-empty and equal length")
    differences = [float(new) - float(old) for old, new in zip(baseline, proposed)]
    if len(differences) == 1:
        return math.inf if differences[0] else 0.0
    deviation = statistics.stdev(differences)
    if deviation == 0.0:
        return math.inf if statistics.fmean(differences) else 0.0
    return statistics.fmean(differences) / deviation


def bootstrap_mean_interval(values, confidence=0.95, samples=2000, seed=7):
    """Deterministic percentile bootstrap interval without heavy dependencies."""
    import random

    values = tuple(float(value) for value in values)
    if not values:
        raise ValueError("values must be non-empty")
    rng = random.Random(seed)
    estimates = sorted(
        statistics.fmean(rng.choice(values) for _ in values)
        for _ in range(int(samples))
    )
    tail = (1.0 - float(confidence)) / 2.0
    low = estimates[max(0, int(tail * len(estimates)))]
    high = estimates[min(len(estimates) - 1, int((1.0 - tail) * len(estimates)))]
    return low, high


def exact_mcnemar_pvalue(baseline_only_success, proposed_only_success):
    """Two-sided exact McNemar test for paired binary outcomes."""
    b = int(baseline_only_success)
    c = int(proposed_only_success)
    discordant = b + c
    if discordant == 0:
        return 1.0
    lower = min(b, c)
    cumulative = sum(
        math.comb(discordant, k) for k in range(lower + 1)
    ) / (2 ** discordant)
    return min(1.0, 2.0 * cumulative)


def paired_policy_comparisons(episodes, baseline_policy, proposed_policy):
    """Compare policies on identical fault, severity, and seed trials.

    The returned rows are directly serializable as a paper result table and
    deliberately omit unmatched trials instead of treating them as independent.
    """
    by_cell = {}
    for episode in episodes:
        if episode.policy not in {baseline_policy, proposed_policy}:
            continue
        key = (episode.fault, episode.severity, episode.seed)
        by_cell.setdefault(key, {})[episode.policy] = episode

    groups = {}
    for (fault, severity, _seed), policies in by_cell.items():
        if baseline_policy in policies and proposed_policy in policies:
            groups.setdefault((fault, severity), []).append((
                policies[baseline_policy], policies[proposed_policy],
            ))

    rows = []
    for (fault, severity), pairs in sorted(
        groups.items(), key=lambda item: (
            item[0][0],
            float("-inf") if item[0][1] is None else item[0][1],
        )
    ):
        baseline_success = [float(old.success) for old, _ in pairs]
        proposed_success = [float(new.success) for _, new in pairs]
        success_differences = [
            new - old for old, new in zip(baseline_success, proposed_success)
        ]
        ci_low, ci_high = bootstrap_mean_interval(success_differences)
        baseline_only = sum(old.success and not new.success for old, new in pairs)
        proposed_only = sum(new.success and not old.success for old, new in pairs)
        baseline_duration = [old.simulation_duration for old, _ in pairs]
        proposed_duration = [new.simulation_duration for _, new in pairs]
        duration_differences = [
            new - old for old, new in zip(baseline_duration, proposed_duration)
        ]
        duration_ci_low, duration_ci_high = bootstrap_mean_interval(
            duration_differences
        )
        rows.append({
            "baseline_policy": baseline_policy,
            "proposed_policy": proposed_policy,
            "fault": fault,
            "severity": severity,
            "paired_episodes": len(pairs),
            "success_rate_difference": statistics.fmean(success_differences),
            "success_difference_ci95_low": ci_low,
            "success_difference_ci95_high": ci_high,
            "baseline_only_successes": baseline_only,
            "proposed_only_successes": proposed_only,
            "mcnemar_exact_pvalue": exact_mcnemar_pvalue(
                baseline_only, proposed_only
            ),
            "mean_duration_difference": statistics.fmean(duration_differences),
            "duration_difference_ci95_low": duration_ci_low,
            "duration_difference_ci95_high": duration_ci_high,
            "duration_paired_cohens_d": paired_cohens_d(
                baseline_duration, proposed_duration
            ),
        })
    return rows
