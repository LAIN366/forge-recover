"""Train/test isolation and immutable manifests for recovery experience."""

import hashlib
import json
from pathlib import Path
import shutil


def validate_disjoint_seeds(training_seeds, test_seeds):
    overlap = set(map(int, training_seeds)).intersection(map(int, test_seeds))
    if overlap:
        raise ValueError(f"training/test seed leakage: {sorted(overlap)}")


def freeze_experience_graph(source, destination, training_seeds):
    source = Path(source)
    destination = Path(destination)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not payload:
        raise ValueError("cannot freeze an empty experience graph")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "experience_graph": destination.name,
        "sha256": digest,
        "training_seeds": sorted(set(map(int, training_seeds))),
        "nodes": len(payload),
        "outcomes": sum(
            int(node["successes"]) + int(node["failures"]) for node in payload
        ),
        "frozen": True,
    }
    manifest_path = destination.with_suffix(destination.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest_path, manifest


def verify_frozen_experience(graph_path):
    graph_path = Path(graph_path)
    manifest_path = graph_path.with_suffix(graph_path.suffix + ".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(graph_path.read_bytes()).hexdigest()
    if digest != manifest["sha256"]:
        raise ValueError("frozen experience graph checksum mismatch")
    if not manifest.get("frozen"):
        raise ValueError("experience manifest is not frozen")
    return manifest
