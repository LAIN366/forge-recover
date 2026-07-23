"""RGB-D object detection and camera-to-world 6D pose estimation."""

from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class VisualPoseDetection:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]
    position: tuple[float, float, float]
    rpy: tuple[float, float, float]
    mask_area: int

    @property
    def pose6d(self):
        return self.position + self.rpy


class BlueCubeRgbdDetector:
    """Detect the blue workpiece in RGB and recover its pose from depth.

    This deterministic detector is the simulation perception baseline. Its
    output is computed from rendered pixels and depth, not MuJoCo object state.
    The interface is intentionally compatible with a future learned detector.
    """

    def __init__(
        self,
        minimum_area=80,
        object_half_extent=0.03,
        aspect_range=(0.35, 2.8),
        yaw_symmetry=4,
        label="TARGET",
        maximum_depth=None,
    ):
        self.minimum_area = int(minimum_area)
        self.object_half_extent = float(object_half_extent)
        self.minimum_aspect = float(aspect_range[0])
        self.maximum_aspect = float(aspect_range[1])
        if not 0.0 < self.minimum_aspect <= self.maximum_aspect:
            raise ValueError("aspect_range must contain positive ordered values")
        self.yaw_symmetry = int(yaw_symmetry)
        if self.yaw_symmetry < 1:
            raise ValueError("yaw_symmetry must be positive")
        self.label = str(label)
        self.maximum_depth = (
            None if maximum_depth is None else float(maximum_depth)
        )
        if self.maximum_depth is not None and self.maximum_depth <= 0.0:
            raise ValueError("maximum_depth must be positive")
        self.lower_hsv = np.array([90, 90, 35], dtype=np.uint8)
        self.upper_hsv = np.array([135, 255, 255], dtype=np.uint8)

    def detect(
        self,
        rgb,
        depth,
        *,
        camera_position,
        camera_forward,
        camera_up,
        fovy,
    ) -> VisualPoseDetection | None:
        rgb = np.asarray(rgb, dtype=np.uint8)
        depth = np.asarray(depth, dtype=float)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or depth.shape != rgb.shape[:2]:
            raise ValueError("RGB and depth frames have incompatible shapes")

        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)
        if self.maximum_depth is not None:
            foreground = (
                np.isfinite(depth)
                & (depth > 0.02)
                & (depth <= self.maximum_depth)
            )
            mask = cv2.bitwise_and(
                mask, np.where(foreground, 255, 0).astype(np.uint8)
            )
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        height, width = mask.shape
        candidates = []
        for index in range(1, count):
            x, y, box_width, box_height, area = (
                int(value) for value in stats[index]
            )
            touches_border = (
                x <= 0 or y <= 0
                or x + box_width >= width
                or y + box_height >= height
            )
            aspect = box_width / max(box_height, 1)
            if (
                area >= self.minimum_area
                and not touches_border
                and self.minimum_aspect <= aspect <= self.maximum_aspect
            ):
                candidates.append((area, index))
        if not candidates:
            return None

        _, component = max(candidates)
        component_mask = labels == component
        valid_depth = component_mask & np.isfinite(depth) & (depth > 0.02)
        if np.count_nonzero(valid_depth) < self.minimum_area // 2:
            return None

        x, y, box_width, box_height, area = (
            int(value) for value in stats[component]
        )
        contours, _ = cv2.findContours(
            component_mask.astype(np.uint8), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        contour = max(contours, key=cv2.contourArea)
        moments = cv2.moments(contour)
        if abs(moments["m00"]) < 1e-6:
            return None
        center_u = float(moments["m10"] / moments["m00"])
        center_v = float(moments["m01"] / moments["m00"])

        depths = depth[valid_depth]
        surface_depth = float(np.median(depths))
        depth_mad = float(np.median(np.abs(depths - surface_depth)))
        fill_ratio = area / max(box_width * box_height, 1)
        depth_compactness = math.exp(-depth_mad / 0.025)
        confidence = float(np.clip(
            0.45 + 0.35 * fill_ratio + 0.20 * depth_compactness,
            0.0, 0.99,
        ))

        camera_position = self._unit_or_value(camera_position, normalize=False)
        forward = self._unit_or_value(camera_forward)
        up = self._unit_or_value(camera_up)
        right = self._unit_or_value(np.cross(forward, up))
        up = self._unit_or_value(np.cross(right, forward))
        focal = (height * 0.5) / math.tan(math.radians(float(fovy)) * 0.5)

        offset_right = (center_u - width * 0.5) * surface_depth / focal
        offset_up = -(center_v - height * 0.5) * surface_depth / focal
        surface_position = (
            camera_position
            + forward * surface_depth
            + right * offset_right
            + up * offset_up
        )
        ys, xs = np.where(valid_depth)
        stride = max(1, len(xs) // 800)
        xs = xs[::stride].astype(float)
        ys = ys[::stride].astype(float)
        zs = depth[valid_depth][::stride]
        points = (
            camera_position[None, :]
            + zs[:, None] * forward[None, :]
            + (((xs - width * 0.5) * zs / focal)[:, None] * right[None, :])
            - (((ys - height * 0.5) * zs / focal)[:, None] * up[None, :])
        )
        rpy, surface_normal = self._estimate_world_orientation(
            points, forward, self.yaw_symmetry
        )
        center_normal = (
            surface_normal if abs(float(surface_normal[2])) >= 0.5
            else -forward
        )
        center_position = surface_position - center_normal * self.object_half_extent
        return VisualPoseDetection(
            label=self.label,
            confidence=confidence,
            bbox=(x, y, x + box_width - 1, y + box_height - 1),
            position=tuple(float(value) for value in center_position),
            rpy=rpy,
            mask_area=area,
        )

    @staticmethod
    def _unit_or_value(value, normalize=True):
        vector = np.asarray(value, dtype=float)
        if not normalize:
            return vector
        norm = float(np.linalg.norm(vector))
        if norm < 1e-9:
            raise ValueError("camera basis vector has zero length")
        return vector / norm

    @staticmethod
    def _estimate_world_yaw(points, symmetry=4):
        if len(points) < 3:
            return 0.0
        xy = points[:, :2] - np.mean(points[:, :2], axis=0)
        covariance = np.cov(xy, rowvar=False)
        values, vectors = np.linalg.eigh(covariance)
        axis = vectors[:, int(np.argmax(values))]
        yaw = math.atan2(float(axis[1]), float(axis[0]))
        period = 2.0 * math.pi / int(symmetry)
        return (yaw + period / 2.0) % period - period / 2.0

    @classmethod
    def _estimate_world_orientation(cls, points, camera_forward, symmetry=4):
        """Estimate object RPY from its principal axis and visible plane."""
        if len(points) < 3:
            return (0.0, 0.0, 0.0), -cls._unit_or_value(camera_forward)
        centered = points - np.mean(points, axis=0)
        covariance = np.cov(centered, rowvar=False)
        values, vectors = np.linalg.eigh(covariance)
        normal = cls._unit_or_value(vectors[:, int(np.argmin(values))])
        camera_forward = cls._unit_or_value(camera_forward)
        if float(np.dot(normal, -camera_forward)) < 0.0:
            normal = -normal

        yaw = cls._estimate_world_yaw(points, symmetry)
        horizontal_axis = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        object_x = cls._unit_or_value(vectors[:, int(np.argmax(values))])
        if float(np.dot(object_x[:2], horizontal_axis[:2])) < 0.0:
            object_x = -object_x
        pitch = math.atan2(
            float(-object_x[2]),
            float(np.linalg.norm(object_x[:2])),
        )
        if abs(float(normal[2])) < 0.5:
            return (0.0, pitch, yaw), normal

        object_x = object_x - normal * float(np.dot(object_x, normal))
        object_x = cls._unit_or_value(object_x)
        object_y = cls._unit_or_value(np.cross(normal, object_x))
        object_x = cls._unit_or_value(np.cross(object_y, normal))
        rotation = np.column_stack((object_x, object_y, normal))

        pitch = math.asin(float(np.clip(-rotation[2, 0], -1.0, 1.0)))
        if abs(math.cos(pitch)) > 1e-6:
            roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
            recovered_yaw = math.atan2(
                float(rotation[1, 0]), float(rotation[0, 0])
            )
        else:
            roll = math.atan2(
                float(-rotation[1, 2]), float(rotation[1, 1])
            )
            recovered_yaw = yaw
        return (roll, pitch, recovered_yaw), normal
