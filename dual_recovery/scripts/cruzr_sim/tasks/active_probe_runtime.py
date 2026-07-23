"""Non-blocking active diagnostic probes for the physical grasp task."""

from dataclasses import dataclass
import math

from cruzr_sim.diagnosis.types import DiagnosisReport, ManipulationObservation


PROBE_DURATION = {
    "reobserve": 0.25,
    "small_gripper_close": 0.35,
    "micro_lift": 0.35,
    "pause_and_hold": 0.40,
}


@dataclass(frozen=True)
class ProbeOutcome:
    name: str
    positive: bool
    rationale: str
    report: DiagnosisReport


class ActiveProbeRuntime:
    def __init__(self):
        self.reset()

    def reset(self):
        """Cancel any in-flight probe and discard its baseline."""
        self.name = None
        self.started_at = 0.0
        self.baseline_height = 0.0
        self.baseline_target_position = None
        self.report = None

    @property
    def active(self):
        return self.name is not None

    def start(self, name, observation, report):
        if name not in PROBE_DURATION or self.active:
            return False
        self.name = name
        self.started_at = observation.timestamp
        self.baseline_height = observation.object_position[2]
        self.baseline_target_position = observation.target_position
        self.report = report
        return True

    def update(self, observation, physical_contact_flags=None):
        if not self.active:
            return None
        if observation.timestamp - self.started_at < PROBE_DURATION[self.name]:
            return None
        contacts = physical_contact_flags or (
            observation.left_contact, observation.right_contact
        )
        both_contacts = bool(contacts[0] and contacts[1])
        any_contact = bool(contacts[0] or contacts[1])
        height_delta = observation.object_position[2] - self.baseline_height
        name = self.name
        if name == "pause_and_hold":
            positive = (
                not both_contacts
                and (observation.object_vertical_velocity < -0.01 or height_delta < -0.008)
            )
            rationale = "contact loss with downward object motion"
        elif name == "reobserve":
            target_position = (
                observation.target_position
                if observation.target_position is not None
                else self.baseline_target_position
            )
            displacement = (
                math.dist(target_position, observation.object_position)
                if target_position is not None else 0.0
            )
            positive = displacement >= 0.045 or not observation.sensor_valid
            rationale = f"refreshed target displacement={displacement:.3f} m"
        elif name == "small_gripper_close":
            positive = not any_contact
            rationale = "test closure still produced no contact"
        elif name == "micro_lift":
            force_ratio = observation.tangent_force / max(observation.normal_force, 1e-3)
            positive = (
                not both_contacts
                or observation.object_vertical_velocity < -0.01
                or force_ratio > 1.0
            )
            rationale = f"micro-lift contact={both_contacts}, force_ratio={force_ratio:.3f}"
        else:
            positive = False
            rationale = "unsupported probe"
        self.name = None
        report = self.report
        self.report = None
        return ProbeOutcome(name, positive, rationale, report)
