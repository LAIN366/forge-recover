from argparse import Namespace
import json

import pytest

import run_dual_arm_experiments as experiments
from cruzr_sim.experiments.experience_protocol import freeze_experience_graph


def args(tmp_path, seeds, graph=None):
    return Namespace(
        paper_protocol=True,
        seeds=list(seeds),
        experience_graph=graph,
        output=tmp_path / "results",
        faults=[],
        severities=[1.0],
        policies=["ours"],
        timeout=1.0,
        wall_timeout=1.0,
        scene_jitter=0.0,
        resume=False,
        experience_ablation="full",
        diagnosis_ablation="full",
        llm_replay_response=None,
        qwen_model="qwen-plus",
        qwen_base_url="https://example.invalid/v1",
        qwen_timeout=1.0,
    )


def frozen_graph(tmp_path, training_seeds=(1, 2, 3)):
    online = tmp_path / "online.json"
    online.write_text(json.dumps([{
        "context": {}, "strategy_id": "safe", "successes": 1,
        "failures": 0, "costs": [1.0],
    }]), encoding="utf-8")
    frozen = tmp_path / "frozen.json"
    freeze_experience_graph(online, frozen, training_seeds)
    return frozen


def test_paper_protocol_rejects_small_pilot(tmp_path):
    with pytest.raises(ValueError, match="at least 30"):
        experiments.run_experiments(args(tmp_path, range(10)))


def test_paper_protocol_rejects_seed_leakage(tmp_path):
    graph = frozen_graph(tmp_path, training_seeds=(1, 2, 3))
    with pytest.raises(ValueError, match="leakage"):
        experiments.run_experiments(args(tmp_path, range(1, 31), graph))


def test_empty_formal_design_validates_frozen_graph_without_mutation(tmp_path):
    graph = frozen_graph(tmp_path)
    before = graph.read_bytes()
    result = experiments.run_experiments(args(tmp_path, range(100, 130), graph))
    assert result == 0
    assert graph.read_bytes() == before


def test_runner_forwards_diagnosis_ablation(tmp_path, monkeypatch):
    invocation = args(tmp_path, [7])
    invocation.paper_protocol = False
    invocation.faults = ["synchronization_delay"]
    invocation.diagnosis_ablation = "no_active_probe"
    commands = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        return Namespace(returncode=1)

    monkeypatch.setattr(experiments.subprocess, "run", fake_run)
    assert experiments.run_experiments(invocation) == 0
    option = commands[0].index("--diagnosis-ablation")
    assert commands[0][option + 1] == "no_active_probe"
