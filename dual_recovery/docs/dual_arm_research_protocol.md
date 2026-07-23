# Dual-Arm Research Protocol

## Research scope

The main task is cooperative transport of a long object under uncertain RGB-D
observations, contact disturbances, arm synchronization errors, and scene
changes. The engineering baseline is not treated as a contribution. Research
claims must be supported by controlled comparisons and repeated trials.

## Core hypotheses

1. A belief task graph that propagates perception and contact uncertainty into
   node readiness improves task success over an FSM and a deterministic DAG.
2. Recovery-aware hierarchical planning reduces unrecoverable states and total
   recovery cost compared with shortest-path dual-arm planning.
3. Active diagnosis and dynamic arm-role switching improve fault isolation and
   recovery success compared with fixed retry policies.

## Proposed method

### Belief task graph

Each node maintains completion belief, observation confidence, failure risk,
arm roles, and recovery edges. Beliefs are updated from RGB-D pose confidence,
contact consistency, tracking stability, and joint-following error. Low-belief
nodes trigger active observation instead of unconditional execution.

### Recovery-aware hierarchical planner

The task layer selects graph nodes and arm roles. The pose layer selects grasp
and support poses. The joint layer uses IK and RRT-Connect with synchronized
dual-arm execution. Candidate plans are ranked by motion cost, clearance,
visibility, grasp stability, estimated failure probability, and expected
recovery cost. A risk-sensitive variant will evaluate CVaR in addition to the
expected cost.

### Active diagnosis and role switching

When execution deviates from the task graph, the robot selects a low-risk probe
that maximizes expected information gain. Probe outcomes update fault beliefs
before choosing re-observation, regrasp, support-arm stabilization, role switch,
local replanning, or safe termination.

The implemented diagnosis core is a stage-conditioned temporal Bayesian fault
filter. Task stage changes the transition prior; multimodal detector hypotheses
form the observation likelihood; posterior persistence suppresses transient
evidence. When the posterior is uncertain, candidate probes are ranked by
expected entropy reduction minus probe cost. The trace logs the complete fault
distribution, entropy, selected probe, and expected information gain, enabling
calibration, diagnosis-delay, and active-sensing ablations rather than only
end-to-end success comparisons.

Executable probes now form a closed loop. Visual ambiguity triggers a fresh
camera observation; contact and synchronization ambiguity trigger a bounded
motion hold followed by resampling. The observed binary outcome is applied as
an additional Bayesian likelihood, logged with the post-probe posterior and
entropy, and the same probe is not repeated within one recovery cycle.

The runtime exposes three controlled diagnosis variants: `full` retains the
temporal posterior and executable probes, `no_active_probe` retains temporal
filtering but suppresses probes, and `no_temporal` uses the instantaneous
diagnosis path and suppresses probes.

### Contextual recovery experience graph

Each recovery outcome updates a context node containing fault type, failed task
stage, bilateral contact pattern, visual reliability, and severity. Strategy
success is represented by a Beta posterior, while observed execution costs form
an empirical tail distribution. Similar contexts transfer discounted evidence.
Candidate strategies are ranked jointly by expected cost, posterior failure
probability, epistemic uncertainty, and empirical cost CVaR. This preserves a
deterministic safety envelope while allowing sample-efficient online adaptation.

Experience graphs must be trained only on training seeds and frozen before the
reported validation/test trials. B2 disables experience updates; Ours enables
them, isolating the contribution from temporal belief inference.

## Comparison groups

| Group | Task model | Uncertainty | Recovery-aware cost | Active diagnosis |
|---|---|---:|---:|---:|
| B0 | Fixed FSM | No | No | No |
| B1 | Deterministic task graph | No | No | No |
| B2 | Belief task graph | Yes | No | Yes |
| Ours | Belief task graph | Yes | Yes | Yes |

An additional oracle group may use MuJoCo ground-truth object poses only as an
upper bound. Ground truth must never drive the reported vision-based policy.

## Fault and uncertainty factors

- RGB-D pose noise and temporary occlusion
- one-arm missed grasp or contact loss
- object slip during cooperative lift or transport
- left-right trajectory delay and synchronization error
- obstacle displacement after planning
- intermittent contact or pose sensor dropout

## Generalization protocol

The training workpiece domain randomizes dimensions, mass, friction, position,
and yaw. The unseen domain extends beyond every training interval and must never
update detector weights, normalization statistics, or recovery experience before
evaluation. Learned perception supplies instance masks only; depth geometry
recovers the 6D pose and deterministic planning retains execution authority.

Report in-domain and unseen-domain success separately, together with relative
performance drop, safe-rejection rate, negative-transfer rate, pose error, and
recovery CVaR. Claims are restricted to rigid, graspable workpieces within the
robot workspace and payload envelope.

Each factor must have at least three severity levels and seeded random initial
poses. Combined faults are evaluated only after single-factor results are
stable.

## Metrics

- end-to-end task success and recovery success
- fault classification accuracy and calibration error
- planning time and execution time
- path length, minimum clearance, and synchronization error
- active probe count and information gain
- graph rollback depth and number of replanned nodes
- expected recovery cost and unrecoverable-state rate
- final object pose error and peak contact force
- diagnosis Brier score, negative log likelihood, and expected calibration error
- weighted rollback cost and number of preserved successful task nodes

## Experimental rigor

- Use at least 30 seeded trials per method-factor-severity cell.
- Report mean, standard deviation, median, and 95% confidence intervals.
- Use paired trials with identical seeds across methods.
- Report effect size in addition to significance tests.
- Keep training, validation, and test scene seeds disjoint.
- Publish failure cases and timeout rules, not only successful demonstrations.
- Never pool different fault severity levels in the same aggregate result row.
- Use exact McNemar tests for paired success and Cohen's dz for paired continuous
  outcomes; always report effect estimates and confidence intervals.

The formal claim-to-code-to-ablation mapping is maintained in
`docs/research_contributions.md`.

## Implementation status

Implemented and physically regression-tested in MuJoCo:

- RGB-D pose belief, risk-aware grasp selection, dynamic arm roles, and
  synchronized dual-arm trajectories.
- Seeded synchronization delay represented as a stale servo reference.
- 10 Hz multimodal observations containing bilateral contact, contact force,
  RGB-D confidence, object pose and velocity, joint tracking residuals, and
  normalized left-right trajectory phase residuals.
- Three-frame fault confirmation and task-graph recovery transactions that
  pause the leading arm, resynchronize, and resume the interrupted node.
- Active RGB-D reobservation guarded by a newly valid frame, plus unilateral
  contact recovery using 6D point-cloud orientation, support-arm contact
  anchoring, local regrasp planning, sustained-contact verification, and
  bimanual object leveling.
- A physical mocap dynamic obstacle, inserted only when the current and terminal
  states remain collision-free but at least one future target-arm waypoint is
  invalidated. Path invalidation immediately holds execution before diagnosis.
- Dynamic-obstacle recovery as a bounded task-graph transaction: coupled hold,
  radial or tangential coupled retreat, obstacle-preserving coupled replan, and
  resumption of the interrupted transport goal. Retreat edges are densely
  checked at the controller joint-step limit; only conservative bounds overlap
  is tolerated while escaping the initial state, never tool or physical contact.
- JSONL traces and paired-seed aggregate metrics with Wilson confidence
  intervals and upper-tail CVaR.

Synchronization delay, left/right contact loss, and visual dropout have passed
single-seed physical regression. They still require paired multi-seed and
multi-severity evaluation before a paper claim. Dynamic obstacle insertion and
bounded recovery are now physically connected to the dual-arm MuJoCo scene,
but the current evidence is a debug regression milestone rather than a
generalizable success-rate result.

Latest debug regression snapshot:

- A post-release `retreat_arms` node removed residual finger contact; the
  no-fault baseline passed 3/3 seeds with 0.015 m initial XY jitter.
- Contact recovery now uses a bounded hierarchy: local support-arm-anchored
  regrasp first, then support-arm setdown, stable RGB-D reobservation, dual
  regrasp, cooperative relift, and task-space residual replanning.
- Setdown reobservation fuses RGB-D with the task-graph context that the object
  is resting on the known table plane. This constrains only center height; XY
  position and orientation remain visually estimated, and collision checks are
  unchanged.
- With 0.015 m initial XY jitter, left-contact loss passed 3/3 debug seeds and
  right-contact loss passed 3/3. Right-side trials used one recovery each;
  left-side seeds 7 and 27 required the bounded fallback after a local regrasp.
  These six debug trials establish a regression milestone, not a paper-level
  success-rate claim.
- The open limitation is recovery efficiency and commitment stability: some
  local regrasps satisfy instantaneous bilateral contact but do not remain
  stable under resumed motion. Multi-severity 30-seed evaluation is still
  required before claiming generalization.
- Visual-dropout outcomes matched the corresponding no-fault scene outcome,
  confirming that guarded reobservation itself did not introduce the earlier
  post-release failures.
- The final dense-safety dynamic-obstacle regression completed for seeds 7, 17,
  and 27. All three debug trials reached the goal with one diagnosis, one
  recovery, negligible final tilt, and no obstacle-safety violation. Final
  position errors were 0.00853, 0.01220, and 0.01226 m, respectively. The
  obstacle invalidated 12 future left-arm waypoints for seed 7 and 11 future
  right-arm waypoints for each of seeds 17 and 27. The recoveries used bounded
  horizontal retreats of 0.140--0.180 m and replans of 7, 31, and 23 waypoints.
- With recovery disabled, seed 7 held position after path invalidation and
  terminated by the 15 s timeout with zero recoveries and no obstacle-safety
  violation. This is a mechanism-level ablation, not yet a paired performance
  comparison. Three debug seeds are insufficient for a paper-level claim;
  paired baselines, multiple severities, at least 30 test seeds per cell,
  confidence intervals, effect sizes, and disclosed failure cases remain
  required.
- The present obstacle is a thin plate entering from a constrained side/above
  candidate set. Coupled recovery preserves the common task-space translation
  at recovery endpoints, but closed-chain residuals are not yet quantified at
  every trajectory sample. Maximum synchronization errors were approximately
  0.203, 0.780, and 0.681 for seeds 7, 17, and 27, so synchronization quality
  remains an explicit limitation despite safe task completion.

Debug ablation command:

```bash
MUJOCO_GL=egl python3 scripts/run_dual_arm_experiments.py \
  --faults synchronization_delay \
  --policies no_recovery full \
  --seeds 7 17 27 \
  --output results/dual_arm_batch_debug
```

Diagnosis ablation commands (use identical faults, severities, and seeds):

```bash
for variant in full no_active_probe no_temporal; do
  MUJOCO_GL=egl python3 scripts/run_dual_arm_experiments.py \
    --faults synchronization_delay vision_dropout \
    --policies ours --seeds 7 17 27 \
    --diagnosis-ablation "$variant" \
    --output "results/diagnosis_${variant}"
done
```

## Resource boundary

The local RTX 3050 Laptop GPU has 4 GB VRAM. Perception training therefore uses
nano/small detectors, mixed precision, small batches, and no full-dataset GPU
cache. Task-graph inference, risk evaluation, diagnosis, and planning remain
lightweight and deterministic unless a learned component demonstrably improves
the controlled baseline.

## LLM boundary

No LLM API is required for the baseline or proposed core method. If evaluated,
an LLM may only propose high-level recovery primitives. Deterministic constraint
checks retain authority over arm roles, trajectories, collision safety, and
execution. The LLM variant is an optional comparison, not a dependency.
