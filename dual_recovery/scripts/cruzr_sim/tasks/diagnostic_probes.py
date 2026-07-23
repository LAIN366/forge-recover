"""Bounded active probes evaluated from portable dual-arm observations."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ProbeSpecification:
    name: str
    hold_seconds: float
    camera_view: str | None = None


PROBE_SPECIFICATIONS = {
    "active_reobserve": ProbeSpecification("active_reobserve", 0.12, "head"),
    "pause_and_contact_test": ProbeSpecification(
        "pause_and_contact_test", 0.12
    ),
    "pause_and_hold": ProbeSpecification("pause_and_hold", 0.15),
    "pause_and_resample_sync": ProbeSpecification(
        "pause_and_resample_sync", 0.12
    ),
}


def probe_specification(name):
    try:
        return PROBE_SPECIFICATIONS[str(name)]
    except KeyError as error:
        raise ValueError(f"probe is not executable: {name}") from error


def evaluate_probe_outcome(name, before, after):
    """Return True when the probe observation supports its target fault."""
    if name == "active_reobserve":
        return not after.visual_valid or after.visual_confidence < 0.42
    if name == "pause_and_contact_test":
        unilateral_loss = before.left_grasp != before.right_grasp
        persistent_loss = (
            before.left_grasp == after.left_grasp
            and before.right_grasp == after.right_grasp
        )
        force_imbalance = abs(after.left_force - after.right_force) > 1.0
        return unilateral_loss and (persistent_loss or force_imbalance)
    if name == "pause_and_hold":
        return (
            not after.left_grasp
            and not after.right_grasp
            and after.object_vertical_velocity < -0.01
        )
    if name == "pause_and_resample_sync":
        return (
            after.synchronization_error > 0.035
            or max(after.left_tracking_error, after.right_tracking_error) > 0.045
        )
    raise ValueError(f"probe is not executable: {name}")
