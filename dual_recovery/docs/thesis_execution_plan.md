# Thesis Execution Plan and Reproducibility Contract

## Research pipeline

The implemented pipeline is:

1. Load task assets and fault metadata only from `scripts/cruzr_sim/scenes/`.
2. Estimate RGB-D 6D pose and fuse vision, depth, bilateral contact, force, and
   joint tracking into calibrated manipulation beliefs.
3. Update task-node readiness and arm roles online in the cooperative task graph.
4. Generate task, role, grasp, and joint-motion candidates hierarchically.
5. Rank candidates by nominal cost, expected recovery cost, clearance, visual
   confidence, grasp stability, and CVaR tail risk.
6. Confirm faults from multiple frames and execute bounded local graph rollback.
7. For transport slip, stop, switch to the downward waist RGB-D view, search the
   table, estimate a new visual 6D pose, regrasp, relift, and resume the failed
   transport node.
8. Compare four paired policies: B0 fixed FSM, B1 deterministic task graph, B2
   belief graph with active diagnosis, and the full recovery-aware method.
9. Run at least 30 disjoint test seeds per method, fault, and severity level.
10. Export episode logs, aggregate statistics, confidence intervals, tail risk,
    paired effect size, failure cases, and a reproducible experiment manifest.

## Required experiment command

Generate a paper-scale manifest before launching expensive simulation:

```bash
cd ~/robosuite_ws/dual_recovery
PYTHONPATH=scripts python3 scripts/generate_research_manifest.py \
  --trials 30 --output results/paper/research_manifest.json
```

Run a bounded pilot before the full factorial batch:

```bash
MUJOCO_GL=egl PYTHONPATH=scripts python3 scripts/run_dual_arm_experiments.py \
  --faults transport_slip vision_occlusion synchronization_delay \
  --severities 0.5 1.0 1.5 --seeds 1000 1001 1002 \
  --output results/pilot
```

The final batch must use 30 or more held-out seeds. Pilot runs are not evidence
for a publication claim.

## Latest mechanism-level pilot

The transport-slip pilot for seed 7 and severity 1.0 completed the full logging
and aggregation pipeline. B0 and B1 failed the final pose check with no recovery.
B2 and Ours both recovered with one active observation and one replan. Ours
reported correct fault classification, successful recovery, graph rollback
depth 3, and final position error of approximately 0.0033 m. Slip recovery also
passed direct physical regressions for seeds 7, 17, and 27; these cover direct
recovery, support-arm setdown after unilateral relift loss, and visual retry
after bilateral relift loss. These are regression milestones, not paper-level
evidence.

## Real-robot integration boundary

Real hardware is connected only through `RobotCommandPort`. The task graph,
diagnosis, recovery, and planning layers do not import the UBTECH SDK directly.
The deployment driver must implement joint telemetry, bounded joint commands,
gripper commands, synchronized RGB-D input, and emergency stop.

Every real command passes `SafetyGatedCommandPort`, which checks operator enable,
finite targets, vector size, joint limits, command duration, and maximum joint
step. Collision checks, watchdog heartbeat, SDK return-code validation, and an
independent physical emergency stop remain mandatory before hardware motion.

The staged hardware procedure is dry-run logging, simulator replay, motors-off
telemetry, single-arm low-speed free-space motion, dual-arm free-space motion,
soft-object grasp, and only then fault recovery. Automatic recovery is disabled
until each earlier stage has a signed acceptance record.

## Claim boundary

Implemented mechanisms are research infrastructure, not evidence by themselves.
Claims require the paired multi-seed experiments, confidence intervals, effect
sizes, ablations, robustness curves, and disclosed timeouts described above.

## Constrained LLM extension

`ours_llm` adds Qwen-Plus only as a bounded preference over task-graph-authorized
recovery strategies. The complete protocol, fallback contract, metrics, and claim
limits are defined in `docs/llm_recovery_protocol.md`.
