"""Generate a compact Markdown report from experiment metrics."""

from pathlib import Path

from .evaluation import evaluate_diagnosis
from .metrics import aggregate_episode_metrics


def _markdown_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def generate_markdown_report(episodes, output_path, title="Cruzr S2 Experiment Report"):
    grouped = aggregate_episode_metrics(episodes)
    diagnosis, _ = evaluate_diagnosis(episodes)
    success_rows = [
        (
            row["policy"], row["fault"], row["episodes"],
            f"{row['success_rate']:.3f}",
            f"{row['mean_recovery_count']:.2f}",
            f"{row['mean_simulation_duration']:.3f}",
            "" if row["mean_diagnosis_latency"] is None
            else f"{row['mean_diagnosis_latency']:.3f}",
        )
        for row in grouped
    ]
    diagnosis_rows = [
        (
            row["policy"], row["episodes"],
            f"{row['classification_accuracy']:.3f}",
            f"{row['anomaly_precision']:.3f}",
            f"{row['anomaly_recall']:.3f}",
            f"{row['anomaly_f1']:.3f}",
        )
        for row in diagnosis
    ]
    content = "\n\n".join([
        f"# {title}",
        "## Task And Recovery Metrics\n\n" + _markdown_table(
            ["Policy", "Fault", "N", "Success", "Recoveries", "Duration", "Latency"],
            success_rows,
        ),
        "## Diagnosis Metrics\n\n" + _markdown_table(
            ["Policy", "N", "Accuracy", "Precision", "Recall", "F1"],
            diagnosis_rows,
        ),
    ]) + "\n"
    Path(output_path).write_text(content, encoding="utf-8")
    return content
