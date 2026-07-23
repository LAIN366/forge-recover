#!/usr/bin/env python3
"""Train and freeze recovery experience using training-only seeds."""

import argparse
from pathlib import Path
import subprocess
import sys

from cruzr_sim.experiments.experience_protocol import freeze_experience_graph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[1000, 1001, 1002])
    parser.add_argument(
        "--faults", nargs="+",
        default=["transport_slip", "synchronization_delay"],
    )
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()
    args.work.parent.mkdir(parents=True, exist_ok=True)
    launcher = Path(__file__).with_name("dual_arm_transport_demo.py")
    for fault in args.faults:
        for seed in args.seeds:
            command = [
                sys.executable, str(launcher), "--headless",
                "--fault", fault, "--policy", "ours",
                "--seed", str(seed), "--timeout", str(args.timeout),
                "--experience-mode", "train",
                "--experience-graph", str(args.work),
            ]
            result = subprocess.run(command, check=False)
            if result.returncode not in {0, 1}:
                raise SystemExit(result.returncode)
    manifest_path, manifest = freeze_experience_graph(
        args.work, args.frozen, args.seeds
    )
    print(f"Frozen {manifest['nodes']} nodes / {manifest['outcomes']} outcomes")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
