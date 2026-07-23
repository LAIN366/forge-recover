"""Learned instance segmentation with RGB-D geometric 6D pose recovery."""

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from .rgbd_pose_detector import BlueCubeRgbdDetector


class InstanceSegmentationRgbdDetector:
    """Use learned masks for recognition and deterministic depth geometry for pose."""

    def __init__(
        self, weights, *, confidence=0.45, device="auto", object_half_extent=0.04,
        label="WORKPIECE", model=None,
    ):
        self.weights = Path(weights) if weights is not None else None
        self.confidence = float(confidence)
        self.device = str(device)
        self.label = str(label)
        if model is None:
            if self.weights is None or not self.weights.is_file():
                raise FileNotFoundError(
                    "trained workpiece segmentation weights are required"
                )
            try:
                from ultralytics import YOLO
            except ImportError as error:
                raise RuntimeError(
                    "install ultralytics in the perception environment"
                ) from error
            model = YOLO(str(self.weights))
        self.model = model
        self.geometry = BlueCubeRgbdDetector(
            minimum_area=100,
            object_half_extent=object_half_extent,
            aspect_range=(0.08, 12.0),
            yaw_symmetry=2,
            label=self.label,
            maximum_depth=1.2,
        )

    def _best_mask(self, rgb):
        kwargs = {"source": rgb, "conf": self.confidence, "verbose": False}
        if self.device != "auto":
            kwargs["device"] = self.device
        results = self.model.predict(**kwargs)
        candidates = []
        for result in results:
            if result.masks is None or result.boxes is None:
                continue
            masks = result.masks.data.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            for mask, confidence in zip(masks, confidences):
                candidates.append((float(confidence), mask))
        if not candidates:
            return None, 0.0
        confidence, mask = max(candidates, key=lambda item: item[0])
        mask = cv2.resize(
            mask.astype(np.float32), (rgb.shape[1], rgb.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ) >= 0.5
        return mask, confidence

    def detect(self, rgb, depth, **camera):
        rgb = np.asarray(rgb, dtype=np.uint8)
        mask, model_confidence = self._best_mask(rgb)
        if mask is None:
            return None
        # Reuse the verified RGB-D/PCA geometry path with the learned mask.
        synthetic_rgb = np.zeros_like(rgb)
        synthetic_rgb[mask] = (0, 0, 255)
        detection = self.geometry.detect(synthetic_rgb, depth, **camera)
        if detection is None:
            return None
        combined_confidence = float(np.sqrt(
            max(0.0, model_confidence) * max(0.0, detection.confidence)
        ))
        return replace(
            detection, label=self.label, confidence=combined_confidence
        )
