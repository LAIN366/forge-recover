#!/usr/bin/env python3
"""Run paired-seed dual-arm recovery experiments and write paper tables."""

import argparse
import csv
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys

from cruzr_sim.experiments.dual_arm_metrics import (
    aggregate_dual_arm_metrics,
    extract_dual_arm_metrics,
)
from cruzr_sim.experiments.research_design import paired_policy_comparisons
from cruzr_sim.experiments.experience_protocol import (
    validate_disjoint_seeds,
    verify_frozen_experience,
)
from cruzr_sim.faults import DualArmFaultScenario
from cruzr_sim.experiments.research_design import PAPER_POLICIES
from cruzr_sim.experiments.policy import ExperimentPolicy


SCRIPT_DIR = Path(__file__).resolve().parent
LAUNCHER = SCRIPT_DIR / "dual_arm_transport_demo.py"


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_experiments(args):
    frozen_manifest = None
    if args.paper_protocol:
        if len(set(args.seeds)) < 30:
            raise ValueError("paper protocol requires at least 30 unique test seeds")
        if args.experience_graph is None:
            raise ValueError("paper protocol requires a frozen experience graph")
        frozen_manifest = verify_frozen_experience(args.experience_graph)
        validate_disjoint_seeds(frozen_manifest["training_seeds"], args.seeds)
    args.output.mkdir(parents=True, exist_ok=True)
    episodes = []
    environment = os.environ.copy()
    environment.setdefault("MUJOCO_GL", "egl")
    for fault in args.faults:
        for severity in args.severities:
            for seed in args.seeds:
                for policy in args.policies:
                    run_id = (
                        f"{fault}__severity_{severity:.2f}__{policy}__seed_{seed}"
                    )
                    log_path = args.output / f"{run_id}.jsonl"
                    command = [
                    sys.executable,
                    str(LAUNCHER),
                    "--headless",
                    "--timeout", str(args.timeout),
                    "--fault", fault,
                    "--fault-severity", str(severity),
                    "--seed", str(seed),
                    "--policy", policy,
                    "--log", str(log_path),
                    "--scene-jitter", str(args.scene_jitter),
                    "--diagnosis-ablation", args.diagnosis_ablation,
                    ]
                    if args.experience_graph is not None:
                        command.extend([
                            "--experience-graph", str(args.experience_graph),
                            "--experience-mode", "frozen",
                            "--experience-ablation", args.experience_ablation,
                        ])
                    if args.llm_replay_response is not None and policy == "ours_llm":
                        command.extend([
                            "--llm-replay-response", str(args.llm_replay_response),
                        ])
                    if policy == "ours_llm":
                        command.extend([
                            "--qwen-model", args.qwen_model,
                            "--qwen-base-url", args.qwen_base_url,
                            "--qwen-timeout", str(args.qwen_timeout),
                        ])
                    if args.resume and log_path.is_file():
                        exit_code = 0
                    else:
                        result = subprocess.run(
                        command,
                        cwd=SCRIPT_DIR.parent,
                        env=environment,
                        timeout=args.wall_timeout,
                        check=False,
                        )
                        exit_code = result.returncode
                    episode = extract_dual_arm_metrics(
                        log_path, run_id, policy, fault, seed, exit_code,
                        severity=severity,
                    )
                    episodes.append(episode)
                    print(
                    f"{run_id}: success={episode.success} "
                    f"duration={episode.simulation_duration:.3f} "
                    f"recoveries={episode.recovery_count}"
                    )

    episode_rows = [asdict(item) for item in episodes]
    aggregate_rows = aggregate_dual_arm_metrics(episodes)
    paired_rows = []
    if "ours" in args.policies:
        for baseline in ("b0_fixed_fsm", "b1_task_graph", "b2_belief_graph"):
            if baseline in args.policies:
                paired_rows.extend(
                    paired_policy_comparisons(episodes, baseline, "ours")
                )
    _write_csv(args.output / "episodes.csv", episode_rows)
    _write_csv(args.output / "aggregate.csv", aggregate_rows)
    _write_csv(args.output / "paired_comparisons.csv", paired_rows)
    (args.output / "aggregate.json").write_text(
        json.dumps(aggregate_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (args.output / "paired_comparisons.json").write_text(
        json.dumps(paired_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if frozen_manifest is not None:
        verified = verify_frozen_experience(args.experience_graph)
        if verified["sha256"] != frozen_manifest["sha256"]:
            raise RuntimeError("frozen experience graph changed during evaluation")
    return 0 if all(item.exit_code in {0, 1} for item in episodes) else 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--faults", nargs="+",
        choices=tuple(item.value for item in DualArmFaultScenario),
        default=["synchronization_delay"],
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=(
            "no_recovery", "full",
            *(policy.value for policy in PAPER_POLICIES),
            ExperimentPolicy.OURS_LLM.value,
        ),
        default=[policy.value for policy in PAPER_POLICIES],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 27])
    parser.add_argument("--severities", nargs="+", type=float, default=[0.5, 1.0, 1.5])
    parser.add_argument("--scene-jitter", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--wall-timeout", type=float, default=180.0)
    parser.add_argument("--output", type=Path, default=Path("results/dual_arm_batch"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--experience-graph", type=Path)
    parser.add_argument(
        "--experience-ablation",
        choices=("success_only", "no_cvar", "full"),
        default="full",
    )
    parser.add_argument(
        "--diagnosis-ablation",
        choices=("full", "no_active_probe", "no_temporal"),
        default="full",
    )
    parser.add_argument(
        "--paper-protocol",
        action="store_true",
        help="enforce 30 held-out seeds and immutable recovery experience",
    )
    parser.add_argument("--llm-replay-response", type=Path)
    parser.add_argument("--qwen-model", default="qwen-plus")
    parser.add_argument(
        "--qwen-base-url",
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    parser.add_argument("--qwen-timeout", type=float, default=30.0)
    raise SystemExit(run_experiments(parser.parse_args()))


if __name__ == "__main__":
    main()
