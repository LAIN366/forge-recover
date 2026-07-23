#!/usr/bin/env python3
"""Run reproducible headless fault/recovery experiments and summarize results."""

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

from cruzr_sim.experiments import (
    ExperimentPolicy,
    aggregate_episode_metrics,
    evaluate_diagnosis,
    extract_episode_metrics,
    generate_markdown_report,
)
from cruzr_sim.faults import FaultScenario


DEFAULT_OUTPUT_ROOT = Path("/home/lain/robosuite_ws/datasets/experiments")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--faults", nargs="+",
        choices=tuple(item.value for item in FaultScenario),
        default=[item.value for item in FaultScenario],
    )
    parser.add_argument(
        "--policies", nargs="+",
        choices=tuple(item.value for item in ExperimentPolicy),
        default=[ExperimentPolicy.ACTIVE_CASE.value],
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=130.0)
    parser.add_argument("--fault-severity", type=float, default=1.0)
    parser.add_argument("--scene-jitter", type=float, default=0.04)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def write_csv(path, metrics):
    rows = [item.to_dict() if hasattr(item, "to_dict") else item for item in metrics]
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if args.repeats < 1:
        raise SystemExit("--repeats must be positive")
    batch_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = args.output_root.expanduser() / batch_id
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "batch_id": batch_id,
        "faults": args.faults,
        "policies": args.policies,
        "repeats": args.repeats,
        "base_seed": args.seed,
        "timeout": args.timeout,
        "fault_severity": args.fault_severity,
        "scene_jitter": args.scene_jitter,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    environment = os.environ.copy()
    environment.setdefault("MUJOCO_GL", "egl")
    launcher = Path(__file__).with_name("physical_grasp_demo.py")
    results = []
    for policy in args.policies:
        for fault_index, fault in enumerate(args.faults):
            for repeat in range(args.repeats):
                seed = args.seed + 1000 * fault_index + repeat
                run_id = f"{policy}-{fault}-seed-{seed}"
                jsonl_path = output_dir / f"{run_id}.jsonl"
                stdout_path = output_dir / f"{run_id}.stdout.log"
                command = [
                    sys.executable, str(launcher), "--headless",
                    "--policy", policy,
                    "--fault", fault,
                    "--fault-severity", str(args.fault_severity),
                    "--seed", str(seed),
                    "--timeout", str(args.timeout),
                    "--scene-jitter", str(args.scene_jitter),
                    "--log", str(jsonl_path),
                ]
                print(f"[{len(results) + 1}] running {run_id}", flush=True)
                try:
                    completed = subprocess.run(
                        command,
                        cwd=launcher.parent,
                        env=environment,
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        timeout=max(60.0, args.timeout * 2.0),
                        check=False,
                    )
                    exit_code = completed.returncode
                    output = completed.stdout
                except subprocess.TimeoutExpired as error:
                    exit_code = 124
                    captured = error.stdout or ""
                    if isinstance(captured, bytes):
                        captured = captured.decode("utf-8", "replace")
                    output = captured + "\nBATCH RUNNER: episode timed out\n"
                stdout_path.write_text(output, encoding="utf-8")
                metrics = extract_episode_metrics(
                    jsonl_path, run_id, policy, fault, seed,
                    exit_code,
                )
                results.append(metrics)
                write_csv(output_dir / "metrics.csv", results)
                write_csv(
                    output_dir / "group_metrics.csv",
                    aggregate_episode_metrics(results),
                )
                print(
                    f"    success={metrics.success} "
                    f"diagnosis={metrics.first_diagnosis} "
                    f"recoveries={metrics.recovery_count}",
                    flush=True,
                )

    successes = sum(item.success for item in results)
    diagnosis_summary, confusion_rows = evaluate_diagnosis(results)
    write_csv(output_dir / "diagnosis_metrics.csv", diagnosis_summary)
    write_csv(output_dir / "diagnosis_confusion.csv", confusion_rows)
    generate_markdown_report(results, output_dir / "report.md")
    summary = {
        **manifest,
        "episodes": len(results),
        "successes": successes,
        "success_rate": successes / len(results) if results else 0.0,
        "diagnosis": diagnosis_summary,
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
