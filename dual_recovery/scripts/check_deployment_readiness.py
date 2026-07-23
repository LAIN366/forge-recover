#!/usr/bin/env python3
"""Check deployment artifacts without connecting to or commanding a robot."""

import argparse

from cruzr_sim.adapters.deployment_readiness import check_deployment_readiness


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/deployment/cruzr_s2.json")
    parser.add_argument("--acceptance", default="configs/deployment/acceptance.json")
    args = parser.parse_args()
    report = check_deployment_readiness(args.config, args.acceptance)
    if report.ready_for_first_motion:
        print("READY_FOR_OPERATOR_REVIEW: pre-motion checks passed; real motion remains disabled")
        return 0
    print("NOT_READY")
    for error in report.errors:
        print(f"- {error}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
