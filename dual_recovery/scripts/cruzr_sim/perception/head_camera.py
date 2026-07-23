#!/usr/bin/env python3
"""Switchable waist/head camera sharing the main MuJoCo model and data."""

import cv2
import glfw
import mujoco
import numpy as np
import shutil
import subprocess


HEAD_BODY_CANDIDATES = (
    "head_pitch_link",
    "head_yaw_link",
    "head_link",
)

WAIST_BODY_CANDIDATES = (
    "waist_yaw_link",
    "waist_link",
    "lifter_pitch_3_link",
    "lifter_pitch_2_link",
)


class HeadCameraWindow:
    CAMERA_LOOKAHEAD = 0.45

    def __init__(self, model, main_window=None, width=480, height=360, fovy=62.0,
                 detector=None, display=True):
        self.model = model
        self.main_window = main_window
        self.width = width
        self.height = height
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.camera)
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.fixedcamid = -1
        self.camera.distance = 1.0
        self.camera.azimuth = 0.0
        self.camera.elevation = 0.0
        self.camera.lookat[:] = 0.0
        self.fovy = fovy
        self.window_name = "Cruzr S2 Robot Camera"
        self.closed = False
        self.display = bool(display)
        self.detector = detector
        self.latest_detection = None
        self.latest_frame = None
        self.latest_rgb = None
        # The imported floor is intentionally large, which makes MuJoCo's
        # automatic near plane clip the nearby waist-camera workspace.
        self.model.vis.map.znear = min(float(self.model.vis.map.znear), 0.001)
        head_id, head_name = self._find_body(
            HEAD_BODY_CANDIDATES, "head", "head"
        )
        waist_id, waist_name = self._find_body(
            WAIST_BODY_CANDIDATES, "waist", "waist"
        )
        self.views = {
            "waist": {
                "body_id": waist_id,
                "body_name": waist_name,
                "offset": np.array([0.18, 0.0, 0.08]),
                "pitch_down": 24.0,
                # Waist link axes are not guaranteed to match the camera
                # convention, so use the robot's global forward direction.
                "use_body_forward": False,
            },
            "head": {
                "body_id": head_id,
                "body_name": head_name,
                "offset": np.array([0.09, 0.0, 0.02]),
                "pitch_down": 0.0,
                "use_body_forward": True,
            },
        }
        self.active_view = "waist"
        if self.display:
            # Keep the client area matched to the rendered frame under desktop
            # display scaling and place it above the MuJoCo viewer.
            window_flags = cv2.WINDOW_AUTOSIZE
            window_flags |= getattr(cv2, "WINDOW_GUI_NORMAL", 0)
            cv2.namedWindow(self.window_name, window_flags)
            self._place_over_main_window()
            topmost_property = getattr(cv2, "WND_PROP_TOPMOST", None)
            if topmost_property is not None:
                try:
                    cv2.setWindowProperty(self.window_name, topmost_property, 1)
                except cv2.error:
                    pass
            self._force_topmost()
        print(f"Waist camera attached to: {waist_name}")
        print(f"Head camera attached to: {head_name}")
        print("Robot camera: press TAB in the camera window to switch views")

    def _draw_detection(self, frame, detection):
        color = (70, 230, 90) if detection is not None else (0, 210, 255)
        if detection is not None:
            x1, y1, x2, y2 = detection.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame, f"{detection.label}  {detection.confidence:.2f}",
                (x1, max(48, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX,
                0.52, color, 2, cv2.LINE_AA,
            )
            cv2.drawMarker(
                frame, ((x1 + x2) // 2, (y1 + y2) // 2), color,
                cv2.MARKER_CROSS, 12, 1,
            )
            x, y, z = detection.position
            roll, pitch, yaw = (np.degrees(value) for value in detection.rpy)
            cv2.putText(
                frame, f"XYZ {x:+.3f} {y:+.3f} {z:+.3f} m", (14, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA,
            )
            cv2.putText(
                frame, f"RPY {roll:+.1f} {pitch:+.1f} {yaw:+.1f} deg", (14, 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA,
            )
        status = "TARGET LOCKED" if detection is not None else "SEARCHING"
        cv2.putText(
            frame, status, (self.width - 170, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA,
        )

    def _place_over_main_window(self):
        if self.main_window is None:
            return
        main_x, main_y = glfw.get_window_pos(self.main_window)
        main_width, _ = glfw.get_window_size(self.main_window)
        camera_x = main_x + main_width - self.width + 38
        cv2.moveWindow(self.window_name, camera_x, main_y + 42)

    def _force_topmost(self):
        """Ask the Linux window manager to keep the camera above the viewer."""
        if shutil.which("wmctrl") is None:
            print("Head camera topmost needs wmctrl: sudo apt install wmctrl")
            return
        subprocess.run(
            ["wmctrl", "-r", self.window_name, "-b", "add,above"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def _find_body(self, candidates, fallback_keyword, label):
        for name in candidates:
            body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, name
            )
            if body_id >= 0:
                return body_id, name

        matches = []
        for body_id in range(1, self.model.nbody):
            name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_BODY, body_id
            )
            if name and fallback_keyword in name.lower():
                matches.append((body_id, name))
        if not matches:
            raise ValueError(f"No {label} body was found in the MuJoCo model.")
        return matches[-1]

    def render(self, data):
        if self.closed:
            return None
        if (
            self.display
            and cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1
        ):
            self.closed = True
            return None

        view = self.views[self.active_view]
        body_id = view["body_id"]
        rotation = data.xmat[body_id].reshape(3, 3)
        body_position = data.xpos[body_id]

        # Cruzr link frames use +X as the forward direction. Position the
        # virtual lens slightly ahead of the head shell and look forward.
        lens_position = body_position + rotation @ view["offset"]
        pitch = np.radians(view["pitch_down"])
        if view["use_body_forward"]:
            local_direction = np.array([np.cos(pitch), 0.0, -np.sin(pitch)])
            direction = rotation @ local_direction
        else:
            # The complete rendered robot is already rotated into its world
            # pose, so derive a horizontal forward direction from the torso.
            forward = rotation[:, 0].copy()
            forward[2] = 0.0
            norm = np.linalg.norm(forward)
            if norm < 1e-6:
                forward = np.array([1.0, 0.0, 0.0])
            else:
                forward /= norm
            direction = np.cos(pitch) * forward + np.array(
                [0.0, 0.0, -np.sin(pitch)]
            )
        # The free camera sits ``distance`` behind lookat along its
        # azimuth/elevation vector. Matching distance and look-ahead keeps the
        # camera at the virtual lens while it looks along ``direction``.
        self.camera.lookat[:] = lens_position + direction * self.CAMERA_LOOKAHEAD
        self.camera.distance = self.CAMERA_LOOKAHEAD

        camera_position_vector = direction
        horizontal = np.hypot(
            camera_position_vector[0], camera_position_vector[1]
        )
        self.camera.azimuth = np.degrees(np.arctan2(
            camera_position_vector[1], camera_position_vector[0]
        ))
        self.camera.elevation = np.degrees(np.arctan2(
            camera_position_vector[2], horizontal
        ))

        self.model.vis.global_.fovy = self.fovy
        self.renderer.update_scene(data, camera=self.camera)
        rgb = self.renderer.render()
        self.latest_rgb = rgb.copy()
        camera_position = np.zeros(3)
        camera_forward = np.zeros(3)
        camera_up = np.zeros(3)
        mujoco.mjv_cameraInModel(
            camera_position, camera_forward, camera_up, self.renderer.scene
        )
        detection = None
        if self.detector is not None:
            self.renderer.enable_depth_rendering()
            self.renderer.update_scene(data, camera=self.camera)
            depth = self.renderer.render()
            self.renderer.disable_depth_rendering()
            detection = self.detector.detect(
                rgb,
                depth,
                camera_position=camera_position,
                camera_forward=camera_forward,
                camera_up=camera_up,
                fovy=self.fovy,
            )
        self.latest_detection = detection
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        self._draw_detection(frame, detection)
        label = "WAIST CAMERA" if self.active_view == "waist" else "HEAD CAMERA"
        cv2.putText(
            frame, label, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
            (255, 255, 255), 2, cv2.LINE_AA,
        )
        cv2.putText(
            frame, "TAB: switch view", (14, self.height - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA,
        )
        self.latest_frame = frame
        if not self.display:
            return detection

        cv2.imshow(self.window_name, frame)
        self._place_over_main_window()
        key = cv2.waitKey(1) & 0xFF
        if key == 9:
            self.active_view = "head" if self.active_view == "waist" else "waist"
            print(f"Robot camera switched to: {self.active_view}")
        return detection

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.renderer.close()
        if self.display:
            try:
                cv2.destroyWindow(self.window_name)
            except cv2.error:
                pass
