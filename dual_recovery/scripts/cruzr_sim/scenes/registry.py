"""Validated access to task scene assets and research scenario metadata."""

from dataclasses import dataclass
import json
from pathlib import Path
import xml.etree.ElementTree as ET


SCENES_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ScenarioDefinition:
    name: str
    trigger_stage: str | None
    action: str


def scene_path(filename):
    """Resolve an asset while preventing scene files from escaping scenes/."""
    path = (SCENES_DIR / filename).resolve()
    if SCENES_DIR.resolve() not in path.parents:
        raise ValueError("scene path must remain inside cruzr_sim/scenes")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_mjcf_worldbody(filename):
    root = ET.parse(scene_path(filename)).getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"scene has no worldbody: {filename}")
    return tuple(worldbody)


def load_research_scenarios(filename="research_scenarios.json"):
    payload = json.loads(scene_path(filename).read_text(encoding="utf-8"))
    return {
        name: ScenarioDefinition(name, item["trigger_stage"], item["action"])
        for name, item in payload["scenarios"].items()
    }
