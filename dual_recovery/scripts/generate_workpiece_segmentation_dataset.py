#!/usr/bin/env python3
"""Render seeded workpiece masks into a YOLO segmentation dataset."""

import argparse
from pathlib import Path
import random

import cv2
import mujoco
import numpy as np

from cruzr_sim.perception import HeadCameraWindow
from cruzr_sim.scenes.domain_randomization import sample_workpiece
from cruzr_sim.simulation import build_dual_arm_scene


def _polygon_from_mask(mask):
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 80:
        return None
    epsilon = 0.003 * cv2.arcLength(contour, True)
    contour = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    if len(contour) < 3:
        return None
    height, width = mask.shape
    values = []
    for x, y in contour:
        values.extend((x / width, y / height))
    return "0 " + " ".join(f"{value:.6f}" for value in values)


def _randomize_appearance(model, object_geom, seed):
    rng = random.Random(int(seed))
    hue = rng.random()
    saturation = rng.uniform(0.35, 0.95)
    value = rng.uniform(0.35, 0.95)
    hsv = np.uint8([[[round(179 * hue), round(255 * saturation), round(255 * value)]]])
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0] / 255.0
    model.geom_rgba[object_geom] = (*rgb, 1.0)
    model.vis.headlight.ambient[:] = rng.uniform(0.15, 0.45)
    model.vis.headlight.diffuse[:] = rng.uniform(0.45, 0.95)


class DatasetRenderer:
    def __init__(self, seed, width=480, height=360):
        initial = sample_workpiece("train", seed)
        self.model, self.data, self.object_geom, *_ = build_dual_arm_scene(initial)
        self.object_body = self.model.geom_bodyid[self.object_geom]
        joint_id = self.model.body_jntadr[self.object_body]
        self.object_qadr = self.model.jnt_qposadr[joint_id]
        self.camera = HeadCameraWindow(
            self.model, width=width, height=height, detector=None, display=False
        )

    def render_sample(self, output, split, domain, seed):
        workpiece = sample_workpiece(domain, seed)
        self.model.geom_size[self.object_geom] = workpiece.half_size
        self.data.qpos[:] = self.model.qpos0
        self.data.qvel[:] = 0.0
        self.data.qpos[self.object_qadr:self.object_qadr + 3] = workpiece.position
        self.data.qpos[self.object_qadr + 3:self.object_qadr + 7] = workpiece.quaternion
        _randomize_appearance(self.model, self.object_geom, seed)
        mujoco.mj_forward(self.model, self.data)
        self.camera.render(self.data)
        rgb = self.camera.latest_rgb.copy()
        self.camera.renderer.enable_segmentation_rendering()
        self.camera.renderer.update_scene(self.data, camera=self.camera.camera)
        segmentation = self.camera.renderer.render()
        self.camera.renderer.disable_segmentation_rendering()
        mask = (
            (segmentation[:, :, 0] == self.object_geom)
            & (segmentation[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
        )
        label = _polygon_from_mask(mask)
        if label is None:
            return False
        stem = f"{domain}_{seed:06d}"
        image_path = output / "images" / split / f"{stem}.jpg"
        label_path = output / "labels" / split / f"{stem}.txt"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(image_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        label_path.write_text(label + "\n", encoding="ascii")
        return True

    def close(self):
        self.camera.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=400)
    parser.add_argument("--val-count", type=int, default=80)
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()
    generated = {"train": 0, "val": 0}
    renderer = DatasetRenderer(args.seed)
    for split, count, offset in (
        ("train", args.train_count, 0), ("val", args.val_count, 100000)
    ):
        for index in range(count):
            seed = args.seed + offset + index
            generated[split] += int(renderer.render_sample(
                args.output, split, "train", seed
            ))
    renderer.close()
    dataset_yaml = (
        f"path: {args.output.resolve().as_posix()}\n"
        "train: images/train\nval: images/val\n"
        "names:\n  0: workpiece\n"
    )
    (args.output / "dataset.yaml").write_text(dataset_yaml, encoding="utf-8")
    print(f"Generated {generated}")


if __name__ == "__main__":
    main()
