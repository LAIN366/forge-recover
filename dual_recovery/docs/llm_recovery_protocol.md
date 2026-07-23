# Constrained Qwen-Plus Recovery Protocol

## Research question

Evaluate whether an LLM preference can improve high-level recovery candidate
prioritization under uncertain or unfamiliar failure context without transferring
motion authority away from the deterministic task graph and risk planner.

Qwen-Plus never emits joint positions, poses, trajectories, SDK calls, or free-form
actions. It receives a structured fault summary and a fault-specific whitelist of
strategy IDs. All deterministic candidates remain in the candidate set even when
the model omits them.

## Decision pipeline

```text
temporal fault posterior + task node + contact/vision context
  -> Qwen-Plus ordered strategy IDs
  -> strict JSON and whitelist parser
  -> bounded preference bonus (maximum 0.25)
  -> Beta success posterior + contextual transfer + CVaR ranking
  -> task-graph rollback and strategy-specific recovery primitives
  -> IK, path planning, collision checks, execution backend
```

An invalid response, missing key, timeout, network error, oversized response, or
non-whitelisted strategy produces an auditable fallback to the complete
deterministic candidate set. Raw model responses are not written to episode logs;
only their SHA-256 digest and aggregate audit fields are retained.

## Reproducible modes

Live Qwen-Plus mode reads the key only from `DASHSCOPE_API_KEY`:

```bash
MUJOCO_GL=egl PYTHONPATH=scripts python3 scripts/dual_arm_transport_demo.py \
  --headless --fault transport_slip --policy ours_llm --seed 7
```

Deterministic response replay exercises the same parser and planner without an API
call:

```bash
MUJOCO_GL=egl PYTHONPATH=scripts python3 scripts/dual_arm_transport_demo.py \
  --headless --fault transport_slip --policy ours_llm --seed 7 \
  --llm-replay-response configs/llm/qwen_transport_slip.example.json
```

The replay file is simulation-only experimental input. It is still size-bounded
and whitelist-validated, and it has no access to a robot backend.

## Required comparisons

Use matched seeds and identical simulator state for:

- `ours`: task graph, active diagnosis, experience graph, and CVaR without LLM.
- `ours_llm`: the same system plus bounded Qwen preference.
- `ours_llm` with forced timeout: verifies deterministic degradation.
- `ours_llm` with malformed and non-whitelisted responses: verifies rejection.

Formal evidence requires at least 30 held-out paired seeds for every
fault/severity/domain cell. Report task success, recovery success, recovery time,
tail cost, final pose error, LLM acceptance and fallback rates, model latency,
accepted/rejected candidate counts, selected strategy distribution, and safety
violations. Training experience and test seeds must remain disjoint.

## Claim boundary

The current integration demonstrates constrained candidate prioritization, not
autonomous LLM planning and not arbitrary novel-action generation. The model
cannot delete deterministic candidates, bypass CVaR, or command motion. A replay
pilot and a missing-key fallback pilot are engineering validation, not publication
evidence. Live Qwen experiments require the operator to configure the API key in
the virtual-machine process environment; the key must never enter the repository.

One live `qwen-plus` transport-slip pilot was completed through the Beijing MaaS
OpenAI-compatible endpoint. The response passed the whitelist with no fallback;
API latency was approximately 2.285 seconds, and both `ours` and `ours_llm`
completed recovery for paired seed 7. The deterministic risk layer selected
`visual_search_regrasp` for both policies. This single-seed connectivity result is
not evidence that the LLM improves task success or recovery quality.
