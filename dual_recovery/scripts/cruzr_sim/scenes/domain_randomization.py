"""Seeded workpiece distributions for in-domain and unseen-scene tests."""

from dataclasses import dataclass
import json
import math
import random

from .registry import scene_path


@dataclass(frozen=True)
class WorkpieceSpec:
    half_size: tuple[float, float, float]
    mass: float
    friction: float
    position: tuple[float, float, float]
    quaternion: tuple[float, float, float, float]
    domain: str
    seed: int

    @property
    def half_length(self):
        return self.half_size[0]


def sample_workpiece(domain, seed, filename="workpiece_domains.json"):
    payload = json.loads(scene_path(filename).read_text(encoding="utf-8"))
    if domain not in payload["domains"]:
        raise ValueError(f"unknown workpiece domain: {domain}")
    ranges = payload["domains"][domain]
    rng = random.Random(int(seed))

    def uniform(name):
        lower, upper = ranges[name]
        return rng.uniform(float(lower), float(upper))

    half_size = (
        uniform("half_length"), uniform("half_width"),
        uniform("half_height"),
    )
    yaw = math.radians(uniform("yaw_degrees"))
    table_top = 0.77
    return WorkpieceSpec(
        half_size=half_size,
        mass=uniform("mass"),
        friction=uniform("friction"),
        position=(
            uniform("position_x"), uniform("position_y"),
            table_top + half_size[2] + 0.002,
        ),
        quaternion=(math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)),
        domain=str(domain),
        seed=int(seed),
    )
