"""Temporal features used by stage-aware manipulation diagnosis."""

from collections import deque
from dataclasses import dataclass
import math
import statistics

from .types import ManipulationObservation


@dataclass(frozen=True)
class TemporalFeatures:
    sample_count: int
    height_change: float
    horizontal_displacement: float
    mean_vertical_velocity: float
    contact_loss_ratio: float
    asymmetric_contact_ratio: float
    mean_normal_force: float
    mean_tangent_ratio: float
    max_tracking_error: float


class TemporalFeatureExtractor:
    def __init__(self, window_seconds: float = 0.8, max_samples: int = 200):
        self.window_seconds = float(window_seconds)
        self.samples: deque[ManipulationObservation] = deque(maxlen=max_samples)

    def reset(self) -> None:
        self.samples.clear()

    def update(self, observation: ManipulationObservation) -> TemporalFeatures:
        self.samples.append(observation)
        cutoff = observation.timestamp - self.window_seconds
        while len(self.samples) > 1 and self.samples[0].timestamp < cutoff:
            self.samples.popleft()
        return self.features()

    def features(self) -> TemporalFeatures:
        if not self.samples:
            return TemporalFeatures(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        samples = list(self.samples)
        first_position = samples[0].object_position
        last_position = samples[-1].object_position
        normal = [sample.normal_force for sample in samples]
        tangent_ratio = [
            sample.tangent_force / max(sample.normal_force, 1e-3)
            for sample in samples
        ]
        return TemporalFeatures(
            sample_count=len(samples),
            height_change=float(last_position[2] - first_position[2]),
            horizontal_displacement=float(math.dist(last_position[:2], first_position[:2])),
            mean_vertical_velocity=float(statistics.fmean(sample.object_vertical_velocity for sample in samples)),
            contact_loss_ratio=float(statistics.fmean(not sample.any_contact for sample in samples)),
            asymmetric_contact_ratio=float(statistics.fmean(sample.left_contact != sample.right_contact for sample in samples)),
            mean_normal_force=float(statistics.fmean(normal)),
            mean_tangent_ratio=float(statistics.fmean(tangent_ratio)),
            max_tracking_error=float(max(sample.tracking_error for sample in samples)),
        )
