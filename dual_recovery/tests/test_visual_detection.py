"""Tests for RGB-D target detection and world-pose recovery."""

import importlib.util
import unittest


VISUAL_DEPS_AVAILABLE = all(
    importlib.util.find_spec(name) is not None
    for name in ("cv2", "numpy")
)

if VISUAL_DEPS_AVAILABLE:
    import numpy as np

    from cruzr_sim.perception.rgbd_pose_detector import BlueCubeRgbdDetector


@unittest.skipUnless(VISUAL_DEPS_AVAILABLE, "requires OpenCV and NumPy")
class RgbdPoseDetectorTest(unittest.TestCase):
    def test_detector_accepts_task_specific_shape_constraints(self):
        detector = BlueCubeRgbdDetector(
            aspect_range=(0.1, 8.0), yaw_symmetry=2,
            label="LONG OBJECT", maximum_depth=0.8,
        )
        self.assertEqual(detector.minimum_aspect, 0.1)
        self.assertEqual(detector.maximum_aspect, 8.0)
        self.assertEqual(detector.yaw_symmetry, 2)
        self.assertEqual(detector.label, "LONG OBJECT")
        self.assertEqual(detector.maximum_depth, 0.8)

    def test_detects_pixels_and_recovers_metric_position(self):
        rgb = np.zeros((120, 160, 3), dtype=np.uint8)
        rgb[40:80, 60:100] = (0, 0, 255)
        depth = np.full((120, 160), 2.0, dtype=float)
        detector = BlueCubeRgbdDetector(
            minimum_area=40, object_half_extent=0.03
        )
        detection = detector.detect(
            rgb,
            depth,
            camera_position=(0.0, 0.0, 0.0),
            camera_forward=(1.0, 0.0, 0.0),
            camera_up=(0.0, 0.0, 1.0),
            fovy=60.0,
        )
        self.assertIsNotNone(detection)
        self.assertEqual(detection.bbox, (60, 40, 99, 79))
        self.assertGreater(detection.confidence, 0.9)
        self.assertAlmostEqual(detection.position[0], 2.03, places=2)
        self.assertAlmostEqual(detection.position[1], 0.0, places=1)
        self.assertAlmostEqual(detection.position[2], 0.0, places=1)

    def test_rejects_border_background_and_empty_frame(self):
        detector = BlueCubeRgbdDetector(minimum_area=40)
        depth = np.ones((120, 160), dtype=float)
        empty = np.zeros((120, 160, 3), dtype=np.uint8)
        self.assertIsNone(detector.detect(
            empty,
            depth,
            camera_position=(0.0, 0.0, 0.0),
            camera_forward=(1.0, 0.0, 0.0),
            camera_up=(0.0, 0.0, 1.0),
            fovy=60.0,
        ))

        background = empty.copy()
        background[:50, :] = (0, 0, 255)
        self.assertIsNone(detector.detect(
            background,
            depth,
            camera_position=(0.0, 0.0, 0.0),
            camera_forward=(1.0, 0.0, 0.0),
            camera_up=(0.0, 0.0, 1.0),
            fovy=60.0,
        ))

    def test_depth_plane_recovers_roll_and_pitch(self):
        detector = BlueCubeRgbdDetector(yaw_symmetry=2)
        roll = 0.18
        pitch = -0.12
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        rotation = np.array([
            [cp, sp * sr, sp * cr],
            [0.0, cr, -sr],
            [-sp, cp * sr, cp * cr],
        ])
        xs, ys = np.meshgrid(
            np.linspace(-0.2, 0.2, 20), np.linspace(-0.05, 0.05, 8)
        )
        local = np.column_stack((xs.ravel(), ys.ravel(), np.zeros(xs.size)))
        points = local @ rotation.T + np.array([1.0, 0.0, 0.8])
        rpy, normal = detector._estimate_world_orientation(
            points, camera_forward=(0.0, 0.0, -1.0), symmetry=2
        )
        self.assertAlmostEqual(rpy[0], roll, places=2)
        self.assertAlmostEqual(rpy[1], pitch, places=2)
        self.assertGreater(normal[2], 0.0)

if __name__ == "__main__":
    unittest.main()
