"""Tests for experience graph freeze and leakage prevention."""

import json

import pytest

from cruzr_sim.experiments.experience_protocol import (
    freeze_experience_graph,
    validate_disjoint_seeds,
    verify_frozen_experience,
)


def test_seed_overlap_is_rejected():
    with pytest.raises(ValueError, match="leakage"):
        validate_disjoint_seeds((1, 2), (2, 3))


def test_frozen_graph_has_verifiable_checksum(tmp_path):
    source = tmp_path / "online.json"
    source.write_text(json.dumps([{
        "context": {}, "strategy_id": "x", "successes": 2,
        "failures": 1, "costs": [1.0, 2.0, 3.0],
    }]), encoding="utf-8")
    frozen = tmp_path / "frozen.json"
    _, manifest = freeze_experience_graph(source, frozen, (1000, 1001))
    assert manifest["outcomes"] == 3
    assert verify_frozen_experience(frozen)["training_seeds"] == [1000, 1001]


def test_modified_frozen_graph_is_rejected(tmp_path):
    source = tmp_path / "online.json"
    source.write_text(json.dumps([{
        "context": {}, "strategy_id": "x", "successes": 1,
        "failures": 0, "costs": [1.0],
    }]), encoding="utf-8")
    frozen = tmp_path / "frozen.json"
    freeze_experience_graph(source, frozen, (1000,))
    frozen.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        verify_frozen_experience(frozen)
