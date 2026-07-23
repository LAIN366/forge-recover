import io
import json
from types import SimpleNamespace
from urllib import error

from cruzr_sim.recovery.dual_arm import DualArmRecoveryPlanner
from cruzr_sim.recovery.experience_graph import RecoveryContext
from cruzr_sim.recovery.llm_adapter import (
    ConstrainedDualArmCandidateGenerator,
    DualArmStrategyProposalParser,
    QwenClientError,
    QwenOpenAICompatibleClient,
    ReplayCandidateClient,
)
from cruzr_sim.diagnosis.dual_arm import DualArmFaultType


def report(fault=DualArmFaultType.BIMANUAL_SLIP):
    return SimpleNamespace(anomalous=True, primary_fault=fault, confidence=0.91)


def context():
    return RecoveryContext("bimanual_slip", "cooperative_transport", False, False, True, "high")


class StubClient:
    model = "qwen-plus"

    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error

    def complete_json(self, *_):
        if self.error:
            raise self.error
        return self.response


def test_strategy_parser_rejects_non_whitelisted_actions():
    parser = DualArmStrategyProposalParser()
    try:
        parser.parse('{"strategy_ids":["move_joint_7"]}', ("safe_regrasp",))
    except ValueError as exc:
        assert "non-whitelisted" in str(exc)
    else:
        raise AssertionError("unsafe strategy was accepted")


def test_qwen_cannot_remove_lower_risk_deterministic_candidate():
    client = StubClient(json.dumps({"strategy_ids": ["safe_setdown_regrasp"]}))
    generator = ConstrainedDualArmCandidateGenerator(client)
    planner = DualArmRecoveryPlanner(candidate_generator=generator)
    plan = planner.plan(report(), "cooperative_transport", context())
    assert plan.strategy_id == "visual_search_regrasp"
    assert planner.last_llm_audit.status == "accepted"
    assert len(planner.last_llm_audit.response_sha256) == 64


def test_qwen_preference_is_bounded():
    client = StubClient(json.dumps({"strategy_ids": ["safe_setdown_regrasp"]}))
    generator = ConstrainedDualArmCandidateGenerator(client, preference_bonus=99.0)
    assert generator.preference_bonus == 0.25


def test_timeout_falls_back_to_full_deterministic_candidate_set():
    client = StubClient(error=QwenClientError("timeout"))
    generator = ConstrainedDualArmCandidateGenerator(client)
    planner = DualArmRecoveryPlanner(candidate_generator=generator)
    plan = planner.plan(report(), "cooperative_transport", context())
    assert plan.strategy_id == "visual_search_regrasp"
    assert planner.last_llm_audit.status == "fallback"


def test_qwen_client_requires_environment_key():
    client = QwenOpenAICompatibleClient(api_key=None)
    client.api_key = None
    try:
        client.complete_json("system", {})
    except QwenClientError as exc:
        assert "DASHSCOPE_API_KEY" in str(exc)
    else:
        raise AssertionError("request without API key was accepted")


def test_qwen_client_uses_openai_compatible_response_shape():
    body = json.dumps({"choices": [{"message": {"content": "{}"}}]}).encode()

    class Response(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *_): return False

    captured = {}
    def opener(req, timeout):
        captured["authorization"] = req.headers["Authorization"]
        captured["timeout"] = timeout
        return Response(body)

    client = QwenOpenAICompatibleClient(api_key="secret", opener=opener, timeout=3.0)
    assert client.complete_json("system", {"fault": "slip"}) == "{}"
    assert captured == {"authorization": "Bearer secret", "timeout": 3.0}


def test_replay_client_is_bounded_and_deterministic(tmp_path):
    path = tmp_path / "response.json"
    path.write_text('{"strategy_ids":["visual_search_regrasp"]}', encoding="utf-8")
    client = ReplayCandidateClient(path)
    assert client.complete_json("ignored", {}) == path.read_text(encoding="utf-8")
    path.write_text("x" * 11, encoding="utf-8")
    client = ReplayCandidateClient(path, maximum_bytes=10)
    try:
        client.complete_json("ignored", {})
    except QwenClientError as exc:
        assert "size limit" in str(exc)
    else:
        raise AssertionError("oversized replay was accepted")
