"""Batch experiment execution and evaluation utilities."""

from .metrics import (
    EpisodeMetrics,
    aggregate_episode_metrics,
    extract_episode_metrics,
)
from .policy import ExperimentPolicy
from .evaluation import evaluate_diagnosis
from .dataset import export_dataset
from .report import generate_markdown_report
from .research_design import (
    PAPER_POLICIES,
    ExperimentCell,
    bootstrap_mean_interval,
    build_paired_factorial_cells,
    paired_cohens_d,
)

__all__ = [
    "EpisodeMetrics",
    "ExperimentPolicy",
    "aggregate_episode_metrics",
    "evaluate_diagnosis",
    "export_dataset",
    "extract_episode_metrics",
    "generate_markdown_report",
    "PAPER_POLICIES",
    "ExperimentCell",
    "bootstrap_mean_interval",
    "build_paired_factorial_cells",
    "paired_cohens_d",
]
