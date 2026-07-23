"""Tests for the optional learned-mask RGB-D pose adapter."""

import numpy as np
import pytest

from cruzr_sim.perception.instance_segmentation import (
    InstanceSegmentationRgbdDetector,
)


class Tensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class FakeModel:
    def predict(self, **kwargs):
        mask = np.zeros((1, 120, 160), dtype=float)
        mask[:, 35:85, 40:120] = 1.0
        masks = type("Masks", (), {"data": Tensor(mask)})()
        boxes = type("Boxes", (), {"conf": Tensor([0.81])})()
        return [type("Result", (), {"masks": masks, "boxes": boxes})()]


def test_learned_mask_reuses_depth_pose_geometry():
    detector = InstanceSegmentationRgbdDetector(None, model=FakeModel())
    rgb = np.zeros((120, 160, 3), dtype=np.uint8)
    depth = np.full((120, 160), 0.5, dtype=float)
    detection = detector.detect(
        rgb, depth,
        camera_position=(0.0, 0.0, 0.0),
        camera_forward=(1.0, 0.0, 0.0),
        camera_up=(0.0, 0.0, 1.0),
        fovy=62.0,
    )
    assert detection is not None
    assert detection.label == "WORKPIECE"
    assert detection.confidence == pytest.approx((0.81 * 0.99) ** 0.5)
    assert detection.mask_area == 4000


def test_missing_trained_weights_fail_closed(tmp_path):
    with pytest.raises(FileNotFoundError, match="trained workpiece"):
        InstanceSegmentationRgbdDetector(tmp_path / "missing.pt")
