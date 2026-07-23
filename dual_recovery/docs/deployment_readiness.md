# Cruzr S2 Deployment Readiness Boundary

This repository is prepared up to, but not including, real-robot motion.

## Completed in simulation

- The task, diagnosis, recovery, and planning layers use an SDK-neutral backend.
- Bilateral commands are validated as a batch before either arm is transmitted.
- The hardware port rejects missing or stale telemetry heartbeats.
- Emergency stop input is latched and requires port recreation after inspection.
- SDK motion calls require an explicit `True` acknowledgement.
- `DryRunRobotDriver` records validated commands without forwarding motion.
- Joint limits, finite values, maximum step, duration, arm side, and vector size
  are checked at the final command boundary.
- Paired experiment mode verifies frozen experience checksums before and after a
  batch and rejects training/test seed overlap.

## Required operator-owned inputs

Copy the two example files in `configs/deployment/` and replace every placeholder:

1. Measure `base_to_left_arm`, `base_to_right_arm`, and `base_to_head_camera`.
2. Map ordered SDK joint names and vendor joint limits for both arms.
3. Map joint telemetry, joint command, RGB-D, and an independent hardware E-stop.
4. Keep `real_motion_enabled` set to `false` during all readiness checks.
5. Record evidence and operator sign-off for dry-run logging, simulator replay,
   and motors-off telemetry.

Run the non-communicating checker:

```bash
PYTHONPATH=scripts python3 scripts/check_deployment_readiness.py \
  --config configs/deployment/cruzr_s2.json \
  --acceptance configs/deployment/acceptance.json
```

`READY_FOR_OPERATOR_REVIEW` means only that the pre-motion artifacts are
consistent. It never enables motion and is not authorization to operate the robot.

## Staged real-robot procedure

The operator must stop and inspect evidence after every stage:

1. Dry-run command logging with `DryRunRobotDriver`.
2. Replay the exact plans in MuJoCo.
3. Motors-off telemetry and frame validation.
4. Single-arm, low-speed, free-space motion.
5. Dual-arm, low-speed, free-space synchronized motion.
6. Soft-object grasp and release.
7. Fault recovery only after all earlier records are signed.

Actual deployment requires the vendor SDK return-code mapping, measured
calibration, independent physical E-stop, workspace supervision, and an operator
at the robot. None of those can be inferred from simulation.

## Known research boundary

The unseen workpiece domain is not equivalent to arbitrary-object handling.
Seed 31 currently yields no feasible fixed-base contact pose; the planner rejects
it instead of relaxing IK tolerances. Supporting such samples requires mobile-base
repositioning or a revised, calibrated workspace definition. Seed 7 passes normal
unseen transport, unseen transport-slip recovery, and dynamic-obstacle recovery.

The CPU segmentation pilot validates the data-to-weight pipeline only. It used a
randomly initialized YOLO11n-seg model for one epoch and produced zero validation
mAP. A pretrained YOLO11s-seg run on a GPU with an independent test set remains
required before making perception claims.
