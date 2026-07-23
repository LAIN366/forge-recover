# Research Contributions and Evidence Map

This document defines the thesis claims before experiments are run. Engineering
features are not counted as contributions unless a controlled ablation isolates
their causal effect.

## C1: Reliability-aware temporal active diagnosis

The latent fault state is inferred by a stage-conditioned Bayesian filter:

\[
b_t(z) \propto p(o_t\mid z)\sum_{z'}p(z\mid z',s_t)b_{t-1}(z').
\]

Candidate probes maximize expected entropy reduction minus probe cost. The
implementation records the complete posterior, entropy and information gain.
Reported evidence must include accuracy, latency, Brier score, negative log
likelihood and ECE, not accuracy alone.

Ablations: single modality, reliability adaptation disabled, temporal update
disabled, active probe disabled. Required stressors: visual occlusion, depth
dropout, force drift and conflicting visual/contact observations.

The executable core ablations are `--diagnosis-ablation no_active_probe` and
`--diagnosis-ablation no_temporal`, compared against
`--diagnosis-ablation full` under paired seeds. The former isolates active
probing while preserving the temporal filter; the latter disables both the
temporal filter and probes so that probes never depend on an unavailable
posterior.

## C2: Dependency-closed local task-graph repair

For failed node \(v_f\), the invalidation set is the least fixed point

\[
R^* = \mu R.\;\{v_f\}\cup\{v\mid pred(v)\cap R\neq\emptyset\}.
\]

This is the minimal rollback set that restores dependency consistency. The
repair transaction additionally exposes occupied resources and the coupled-arm
synchronization boundary. Its cost is

\[
J_R=\sum_{v\in R^*}c_v,
\]

while successful nodes outside \(R^*\) remain committed. The runtime logs
rollback depth, weighted cost and number of preserved task nodes.

Ablations: full task restart, fixed-depth rollback and dependency-closed local
repair. Evidence: success, recovery time, re-executed nodes, preserved nodes and
rollback cost. Minimality is an algorithmic property; performance improvement
still requires paired experiments.

## C3: Recovery-aware hierarchical dual-arm planning

Task, coordination and joint-motion candidates are ranked by nominal motion
cost, clearance, synchronization, expected recovery cost and tail risk. The
physical planner remains subject to IK, collision, joint-step and coupled-arm
constraints.

Ablations: shortest path, expected-risk planning without CVaR, full risk-aware
planning and role switching disabled. Evidence: planning time, success,
clearance, synchronization residual, path length, unrecoverable-state rate and
cost CVaR.

## C4: Contextual Bayesian recovery experience

Each context-strategy edge maintains a Beta posterior over success and an
empirical cost distribution. Similar contexts contribute discounted evidence.
Selection combines expected cost, posterior failure probability, epistemic
uncertainty and upper-tail CVaR.

Ablations: no experience, success probability only, no CVaR and full method.
Training seeds must be disjoint from evaluation seeds and the experience graph
must be frozen during reported tests. Evidence includes sample efficiency,
negative transfer on unseen workpieces and recovery cost CVaR.

## Statistical decision rules

Every method-fault-severity cell uses at least 30 identical paired seeds. Fault
severity is a grouping key and must never be pooled implicitly. Success is
reported with Wilson intervals; paired success differences use bootstrap
intervals and exact McNemar tests. Continuous paired outcomes report bootstrap
intervals and Cohen's dz. Effect size and uncertainty take precedence over a
binary p-value.

The experiment runner writes `aggregate.csv`, `paired_comparisons.csv` and their
JSON equivalents. A claim is accepted only when its named ablation improves the
primary metric without increasing collision or unsafe-termination rates.

## LLM boundary

The LLM variant is an optional high-level candidate-ranking study, not one of
the four core contributions. Deterministic candidates, constraint checking,
collision safety and execution authority remain outside the LLM. A single live
API trial proves connectivity only and cannot support a performance claim.
