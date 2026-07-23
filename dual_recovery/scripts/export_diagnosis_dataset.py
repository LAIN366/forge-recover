#!/usr/bin/env python3
"""Export experiment JSONL traces to a flat diagnosis CSV dataset."""

import argparse
from pathlib import Path

from cruzr_sim.experiments.dataset import export_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = []
    for item in args.inputs:
        if item.is_dir():
            paths.extend(sorted(item.rglob("*.jsonl")))
        elif item.is_file():
            paths.append(item)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = export_dataset(paths, args.output)
    print(f"Exported {count} observations to {args.output}")


if __name__ == "__main__":
    main()
