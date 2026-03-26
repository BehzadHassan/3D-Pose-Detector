# -*- coding: utf-8 -*-
"""
model_3d.py
Real-time 3D stick-figure renderer using OpenCV.

Takes normalised (x, y, z) landmark coords from PoseDetector and renders
a rotatable 3D skeleton onto a BGR frame (no matplotlib required).
"""

import math
import cv2
import numpy as np

# ── Skeleton topology ─────────────────────────────────────────────────────────
# Simplified connections for a clean 3D look (same as pose_detector._CONNECTIONS)
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32),
    (27, 31), (28, 32),
]

# Major body connections for thicker rendering
_MAJOR_BONES = {
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # arms
    (11, 23), (12, 24), (23, 24),                        # torso
    (23, 25), (24, 26), (25, 27), (26, 28),              # legs
}

# Joint groups for colour coding
_HEAD_IDS  = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
_TORSO_IDS = {11, 12, 23, 24}
_ARM_IDS   = {13, 14, 15, 16, 17, 18, 19, 20, 21, 22}
_LEG_IDS   = {25, 26, 27, 28, 29, 30, 31, 32}

# Colours (BGR)
COL_HEAD  = (0, 255, 200)
COL_TORSO = (0, 200, 120)
COL_ARM   = (255, 180, 0)
COL_LEG   = (200, 100, 255)
COL_BONE_GLOW  = (0, 80, 40)
COL_BONE_BRIGHT = (0, 220, 120)
COL_FLOOR = (20, 30, 20)


def _joint_color(idx: int) -> tuple[int, int, int]:
    """Return a colour based on joint group."""
    if idx in _HEAD_IDS:
        return COL_HEAD
    if idx in _TORSO_IDS:
        return COL_TORSO
    if idx in _ARM_IDS:
        return COL_ARM
    return COL_LEG


# ── 3D maths ──────────────────────────────────────────────────────────────────

def _rot_y(angle_deg: float) -> np.ndarray:
    """3×3 rotation matrix around Y axis."""
    a = math.radians(angle_deg)
    return np.array([
        [math.cos(a), 0, math.sin(a)],
        [0,            1, 0           ],
        [-math.sin(a), 0, math.cos(a)],
    ])


def _rot_x(angle_deg: float) -> np.ndarray:
    """3×3 rotation matrix around X axis."""
    a = math.radians(angle_deg)
    return np.array([
        [1, 0,            0           ],
        [0, math.cos(a), -math.sin(a)],
        [0, math.sin(a),  math.cos(a)],
    ])


# ── Renderer ──────────────────────────────────────────────────────────────────

class Model3DRenderer:
    """
    Renders a 3D stick figure from normalised landmarks.

    Call `render(landmarks_3d)` each frame.
    Use `yaw` / `pitch` to control the viewpoint.
    """

    def __init__(self, width: int = 960, height: int = 540):
        self.W = width
        self.H = height
        self.yaw   = 0.0     # degrees, set by mouse drag
        self.pitch = 15.0    # degrees
        self.zoom  = 1.0     # zoom multiplier

        # Smoothing (exponential moving average for each landmark)
        self._smooth: np.ndarray | None = None
        self._alpha = 0.45   # blending factor (0 = heavy smooth, 1 = raw)

    def render(
        self,
        landmarks_3d: list[tuple[float, float, float]],
        anim_tick: int = 0,
    ) -> np.ndarray:
        """
        Render a 3D skeleton frame (BGR).

        Args:
            landmarks_3d: list of (x, y, z) normalised coords from MediaPipe
            anim_tick:    frame counter for subtle animations
        Returns:
            BGR frame (self.W × self.H)
        """
        W, H = self.W, self.H
        frame = np.zeros((H, W, 3), dtype=np.uint8)

        if not landmarks_3d:
            # Draw spinning loading arc
            cx, cy = W // 2, H // 2
            a = (anim_tick * 8) % 360
            cv2.ellipse(frame, (cx, cy), (40, 40), a, 0, 90, (0, 255, 140), 3, cv2.LINE_AA)
            cv2.ellipse(frame, (cx, cy), (40, 40), a+180, 0, 90, (0, 255, 140), 3, cv2.LINE_AA)
            
            cv2.putText(frame, "Searching for pose...",
                        (cx - 85, cy + 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 200, 100), 1, cv2.LINE_AA)
            return frame

        # ── Convert to numpy & centre ──────────────────────────────────
        raw = np.array(landmarks_3d, dtype=np.float64)  # (N, 3)
        # MediaPipe: x right, y down, z towards camera
        # Convert to: x right, y UP, z towards viewer
        raw[:, 1] = -raw[:, 1]
        raw[:, 2] = -raw[:, 2]

        # Centre on midpoint of hips (indices 23, 24)
        if len(raw) > 24:
            centre = (raw[23] + raw[24]) / 2.0
        else:
            centre = raw.mean(axis=0)
        pts = raw - centre

        # ── Smooth ─────────────────────────────────────────────────────
        if self._smooth is None or self._smooth.shape != pts.shape:
            self._smooth = pts.copy()
        else:
            self._smooth = self._alpha * pts + (1 - self._alpha) * self._smooth
        pts = self._smooth.copy()

        # ── Scale to fill frame ────────────────────────────────────────
        extent = max(np.abs(pts).max(), 0.001)
        scale  = min(W, H) * 0.38 / extent * self.zoom
        pts   *= scale

        # ── Apply rotation ─────────────────────────────────────────────
        R = _rot_y(self.yaw) @ _rot_x(self.pitch)
        pts = (R @ pts.T).T  # (N, 3)

        # ── Perspective projection ─────────────────────────────────────
        fov_d  = 800.0  # focal length for perspective
        cx, cy = W // 2, H // 2

        proj: dict[int, tuple[int, int]] = {}
        depths: dict[int, float] = {}
        for i, (x, y, z) in enumerate(pts):
            pz = z + fov_d
            if pz < 1:
                pz = 1
            px = int(cx + x * fov_d / pz)
            py = int(cy - y * fov_d / pz)  # -y because screen y is down
            proj[i] = (px, py)
            depths[i] = pz

        # ── Draw ground grid ──────────────────────────────────────────
        self._draw_ground(frame, R, scale, fov_d, cx, cy)

        # ── Sort bones by average depth (painter's algorithm) ─────────
        bone_order = sorted(
            CONNECTIONS,
            key=lambda ab: -(depths.get(ab[0], 0) + depths.get(ab[1], 0)) / 2,
        )

        # ── Draw bones ─────────────────────────────────────────────────
        for a, b in bone_order:
            if a not in proj or b not in proj:
                continue
            pa, pb = proj[a], proj[b]
            is_major = (a, b) in _MAJOR_BONES or (b, a) in _MAJOR_BONES
            thick_glow  = 8 if is_major else 5
            thick_bright = 3 if is_major else 2

            # Glow
            cv2.line(frame, pa, pb, COL_BONE_GLOW, thick_glow, cv2.LINE_AA)
            # Bright
            cv2.line(frame, pa, pb, COL_BONE_BRIGHT, thick_bright, cv2.LINE_AA)

        # ── Draw joints (sorted front-to-back) ────────────────────────
        joint_order = sorted(proj.keys(), key=lambda i: -depths.get(i, 0))
        for i in joint_order:
            px, py = proj[i]
            col = _joint_color(i)
            # Glow ring
            cv2.circle(frame, (px, py), 7, (col[0] // 3, col[1] // 3, col[2] // 3),
                        -1, cv2.LINE_AA)
            # Solid dot
            cv2.circle(frame, (px, py), 4, col, -1, cv2.LINE_AA)

        # ── Head circle (encapsulating facial landmarks) ───────────────
        if 0 in proj:
            max_dist = 0.0
            p0 = np.array(proj[0])
            # Check distance from nose to eyes, ears, mouth (indices 1 to 10)
            for i in range(1, 11):
                if i in proj:
                    d = np.linalg.norm(np.array(proj[i]) - p0)
                    if d > max_dist:
                        max_dist = float(d)
            
            # Make the radius large enough to cover all face points
            base_r = max_dist * 1.5 if max_dist > 5.0 else 16.0
            head_r = int(base_r + 3 * math.sin(anim_tick * 0.12))
            
            cv2.circle(frame, proj[0], head_r,
                        (0, 180 + int(60 * math.sin(anim_tick * 0.12)), 100),
                        1, cv2.LINE_AA)

        # ── Axis label ─────────────────────────────────────────────────
        cv2.putText(frame, f"Yaw {self.yaw:.0f}\u00b0  Pitch {self.pitch:.0f}\u00b0",
                    (12, H - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 100, 50), 1, cv2.LINE_AA)

        # Label
        cv2.putText(frame, "3D POSE MODEL",
                    (W // 2 - 75, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 180, 90), 1, cv2.LINE_AA)

        return frame

    # ── Ground grid ───────────────────────────────────────────────────────

    def _draw_ground(
        self, frame, R, scale, fov_d, cx, cy,
        grid_y: float = -0.35, grid_range: float = 0.6, grid_step: int = 6,
    ):
        """Draw a simple perspective ground grid."""
        lines: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = []
        n = grid_step
        for i in range(-n, n + 1):
            t = i / n * grid_range * scale
            r = grid_range * scale
            y = grid_y * scale
            lines.append(((t, y, -r), (t, y, r)))
            lines.append(((-r, y, t), (r, y, t)))

        for (x1, y1, z1), (x2, y2, z2) in lines:
            p1 = R @ np.array([x1, y1, z1])
            p2 = R @ np.array([x2, y2, z2])

            def _proj(p):
                pz = p[2] + fov_d
                if pz < 1:
                    pz = 1
                return int(cx + p[0] * fov_d / pz), int(cy - p[1] * fov_d / pz)

            cv2.line(frame, _proj(p1), _proj(p2), COL_FLOOR, 1, cv2.LINE_AA)

    # ── Idle frame (no detection running) ─────────────────────────────────

    def render_idle(self, tick: int = 0) -> np.ndarray:
        """Render a rotating wireframe T-pose when no detection is active."""
        # Synthetic T-pose landmarks (normalised)
        t_pose = [
            (0.50, 0.15, 0.0),   # 0  nose
            (0.48, 0.13, 0.01),  # 1  left eye inner
            (0.52, 0.13, 0.01),  # 2  right eye inner
            (0.45, 0.14, 0.02),  # 3  left ear
            (0.55, 0.14, 0.02),  # 4  right ear
            (0.45, 0.14, 0.02),  # 5  left eye outer (≈ear)
            (0.55, 0.14, 0.02),  # 6  right eye outer
            (0.44, 0.15, 0.03),  # 7  left ear (use)
            (0.56, 0.15, 0.03),  # 8  right ear (use)
            (0.49, 0.13, 0.0),   # 9  mouth left
            (0.51, 0.13, 0.0),   # 10 mouth right
            (0.35, 0.30, 0.0),   # 11 left shoulder
            (0.65, 0.30, 0.0),   # 12 right shoulder
            (0.20, 0.30, 0.0),   # 13 left elbow
            (0.80, 0.30, 0.0),   # 14 right elbow
            (0.08, 0.30, 0.0),   # 15 left wrist
            (0.92, 0.30, 0.0),   # 16 right wrist
            (0.05, 0.30, 0.0),   # 17 left pinky
            (0.95, 0.30, 0.0),   # 18 right pinky
            (0.06, 0.29, 0.0),   # 19 left index
            (0.94, 0.29, 0.0),   # 20 right index
            (0.07, 0.31, 0.0),   # 21 left thumb
            (0.93, 0.31, 0.0),   # 22 right thumb
            (0.42, 0.58, 0.0),   # 23 left hip
            (0.58, 0.58, 0.0),   # 24 right hip
            (0.42, 0.76, 0.0),   # 25 left knee
            (0.58, 0.76, 0.0),   # 26 right knee
            (0.42, 0.94, 0.0),   # 27 left ankle
            (0.58, 0.94, 0.0),   # 28 right ankle
            (0.40, 0.96, 0.0),   # 29 left heel
            (0.60, 0.96, 0.0),   # 30 right heel
            (0.43, 0.97, 0.0),   # 31 left foot index
            (0.57, 0.97, 0.0),   # 32 right foot index
        ]

        # Slowly auto-rotate
        saved_yaw = self.yaw
        self.yaw = (tick * 0.8) % 360
        self._smooth = None          # no smoothing for idle
        result = self.render(t_pose, tick)
        self.yaw = saved_yaw
        self._smooth = None

        # Overlay label
        cv2.putText(result, "IDLE  -  Drag to rotate",
                    (self.W // 2 - 130, self.H - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (0, 140, 70), 1, cv2.LINE_AA)

        return result

    def close(self):
        """No-op for OpenCV renderer, matches API in app.py."""
        pass
