"""Tests for reproducible workpiece-domain generalization splits."""

from cruzr_sim.scenes.domain_randomization import sample_workpiece


def test_workpiece_sampling_is_seed_reproducible():
    assert sample_workpiece("train", 41) == sample_workpiece("train", 41)


def test_unseen_domain_extends_beyond_training_support():
    unseen = [sample_workpiece("unseen", seed) for seed in range(100)]
    assert any(item.half_length < 0.15 for item in unseen)
    assert any(item.half_length > 0.20 for item in unseen)
    assert any(item.friction < 0.80 for item in unseen)
    assert any(item.friction > 1.60 for item in unseen)


def test_sampled_workpiece_remains_physically_valid():
    item = sample_workpiece("unseen", 7)
    assert all(value > 0.0 for value in item.half_size)
    assert item.mass > 0.0
    assert item.friction > 0.0
    assert item.position[2] > item.half_size[2]
