"""Bounded active-search policy after an object is dropped in transport."""

from dataclasses import dataclass
from enum import Enum


class SlipRecoveryPhase(str, Enum):
    STOP = "stop"
    LOOK_DOWN = "look_down"
    SEARCH = "search"
    APPROACH = "approach"
    REGRASP = "regrasp"
    VERIFY = "verify"
    RESUME = "resume"
    SAFE_STOP = "safe_stop"


@dataclass(frozen=True)
class SearchObservation:
    timestamp: float
    detection_valid: bool
    pose_confidence: float
    pose_6d: tuple[float, float, float, float, float, float] | None
    grounded: bool
    bilateral_contact: bool = False


@dataclass(frozen=True)
class SlipRecoveryCommand:
    phase: SlipRecoveryPhase
    primitive: str
    target_pose_6d: tuple[float, float, float, float, float, float] | None = None


class SlipRecoveryPolicy:
    """Drive camera search and regrasp without using simulator ground truth."""

    def __init__(self, minimum_pose_confidence=0.65, search_timeout=8.0):
        self.minimum_pose_confidence = float(minimum_pose_confidence)
        self.search_timeout = float(search_timeout)
        self.reset()

    def reset(self):
        self.phase = SlipRecoveryPhase.STOP
        self.started_at = None
        self.last_pose = None

    def update(self, observation):
        if self.started_at is None:
            self.started_at = observation.timestamp
            self.phase = SlipRecoveryPhase.LOOK_DOWN
            return SlipRecoveryCommand(self.phase, "waist_camera_look_down")
        if observation.timestamp - self.started_at > self.search_timeout:
            self.phase = SlipRecoveryPhase.SAFE_STOP
            return SlipRecoveryCommand(self.phase, "safe_stop")
        if self.phase == SlipRecoveryPhase.LOOK_DOWN:
            self.phase = SlipRecoveryPhase.SEARCH
            return SlipRecoveryCommand(self.phase, "scan_table_workspace")
        if self.phase == SlipRecoveryPhase.SEARCH:
            if not observation.detection_valid or (
                observation.pose_confidence < self.minimum_pose_confidence
            ):
                return SlipRecoveryCommand(self.phase, "scan_next_view")
            self.last_pose = observation.pose_6d
            self.phase = SlipRecoveryPhase.APPROACH
            return SlipRecoveryCommand(
                self.phase, "plan_bimanual_pregrasp", self.last_pose
            )
        if self.phase == SlipRecoveryPhase.APPROACH:
            if not observation.grounded:
                self.phase = SlipRecoveryPhase.SEARCH
                return SlipRecoveryCommand(self.phase, "refresh_rgbd_6d_pose")
            self.phase = SlipRecoveryPhase.REGRASP
            return SlipRecoveryCommand(
                self.phase, "execute_bimanual_regrasp", self.last_pose
            )
        if self.phase == SlipRecoveryPhase.REGRASP:
            self.phase = SlipRecoveryPhase.VERIFY
            return SlipRecoveryCommand(self.phase, "verify_force_and_contact")
        if self.phase == SlipRecoveryPhase.VERIFY:
            if not observation.bilateral_contact:
                self.phase = SlipRecoveryPhase.SEARCH
                return SlipRecoveryCommand(self.phase, "reobserve_after_miss")
            self.phase = SlipRecoveryPhase.RESUME
            return SlipRecoveryCommand(self.phase, "resume_interrupted_task_node")
        return SlipRecoveryCommand(self.phase, "hold")
