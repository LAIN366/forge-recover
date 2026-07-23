"""Reliability-aware fusion of vision, force, contact, and proprioception."""

from dataclasses import dataclass
import math

from cruzr_sim.tasks.belief_update import (
    BayesianBeliefUpdater,
    binary_evidence,
    confidence_evidence,
)


@dataclass(frozen=True)
class MultimodalMeasurement:
    timestamp: float
    pose_6d: tuple[float, float, float, float, float, float] | None
    vision_confidence: float
    depth_valid: bool
    left_contacts: tuple[bool, bool]
    right_contacts: tuple[bool, bool]
    left_force: float
    right_force: float
    joint_tracking_error: float
    sensor_valid: bool = True


@dataclass(frozen=True)
class ManipulationBeliefState:
    timestamp: float
    pose_6d: tuple[float, float, float, float, float, float] | None
    pose_belief: float
    grasp_belief: float
    execution_belief: float
    source_reliability: dict[str, float]
    observable: bool


class ReliabilityAwareFusion:
    """Maintain source reliability and expose auditable task beliefs."""

    def __init__(self, prior=0.55, reliability_decay=0.92):
        self.prior = float(prior)
        self.reliability_decay = float(reliability_decay)
        self.reliability = {
            "vision": 0.80,
            "depth": 0.90,
            "contact": 0.90,
            "force": 0.80,
            "proprioception": 0.95,
        }

    def _update_reliability(self, name, consistent):
        old = self.reliability[name]
        target = 1.0 if consistent else 0.05
        self.reliability[name] = min(0.99, max(
            0.05,
            self.reliability_decay * old
            + (1.0 - self.reliability_decay) * target,
        ))

    def update(self, measurement):
        vision_ok = (
            measurement.sensor_valid
            and measurement.pose_6d is not None
            and measurement.vision_confidence >= 0.25
        )
        contact_ok = all(measurement.left_contacts + measurement.right_contacts)
        force_ok = min(measurement.left_force, measurement.right_force) > 0.05
        tracking_ok = math.isfinite(measurement.joint_tracking_error) and (
            measurement.joint_tracking_error < 0.045
        )
        for source, consistent in (
            ("vision", vision_ok),
            ("depth", measurement.depth_valid),
            ("contact", contact_ok),
            ("force", force_ok),
            ("proprioception", tracking_ok),
        ):
            self._update_reliability(source, consistent)

        pose_updater = BayesianBeliefUpdater(self.prior)
        pose_belief = pose_updater.fuse((
            confidence_evidence(
                measurement.vision_confidence * self.reliability["vision"],
                "rgbd_6d_pose",
            ),
            binary_evidence(
                measurement.depth_valid,
                "valid_depth",
                true_positive=self.reliability["depth"],
            ),
        ))
        grasp_updater = BayesianBeliefUpdater(self.prior)
        grasp_belief = grasp_updater.fuse((
            binary_evidence(
                contact_ok, "bilateral_contact",
                true_positive=self.reliability["contact"],
            ),
            binary_evidence(
                force_ok, "force_support",
                true_positive=self.reliability["force"],
            ),
        ))
        execution_updater = BayesianBeliefUpdater(self.prior)
        execution_belief = execution_updater.fuse((
            binary_evidence(
                tracking_ok, "joint_tracking",
                true_positive=self.reliability["proprioception"],
            ),
            binary_evidence(measurement.sensor_valid, "sensor_health"),
        ))
        return ManipulationBeliefState(
            timestamp=measurement.timestamp,
            pose_6d=measurement.pose_6d,
            pose_belief=pose_belief,
            grasp_belief=grasp_belief,
            execution_belief=execution_belief,
            source_reliability=dict(self.reliability),
            observable=vision_ok or contact_ok,
        )
