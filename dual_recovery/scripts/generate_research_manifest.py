#!/usr/bin/env python3
"""Write a reproducible JSON manifest for paper-scale experiment batches."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from cruzr_sim.experiments.research_design import build_paired_factorial_cells
from cruzr_sim.faults import DualArmFaultScenario


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/research_manifest.json"))
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument(
        "--faults", nargs="+",
        choices=tuple(item.value for item in DualArmFaultScenario),
        default=[
            "left_contact_loss", "right_contact_loss", "transport_slip",
            "synchronization_delay", "vision_occlusion", "sensor_dropout",
            "target_pose_shift", "dynamic_obstacle",
        ],
    )
    args = parser.parse_args()
    seeds = range(args.seed_start, args.seed_start + args.trials)
    cells = build_paired_factorial_cells(args.faults, seeds=seeds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps([asdict(cell) for cell in cells], indent=2),
        encoding="utf-8",
    )
    print(f"wrote {len(cells)} paired experiment cells to {args.output}")


if __name__ == "__main__":
    main()
