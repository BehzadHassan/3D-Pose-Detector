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

# Body groups
_HEAD_IDS  = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
_TORSO_IDS = {11, 12, 23, 24}
_ARM_IDS   = {13, 14, 15, 16, 17, 18, 19, 20, 21, 22}
_LEG_IDS   = {25, 26, 27, 28, 29, 30, 31, 32}

# Hand groups (21 landmarks)
_THUMB_IDS  = {1, 2, 3, 4}
_INDEX_IDS  = {5, 6, 7, 8}
_MIDDLE_IDS = {9, 10, 11, 12}
_RING_IDS   = {13, 14, 15, 16}
_PINKY_IDS  = {17, 18, 19, 20}
_WRIST_ID   = 0

POSE_CONNECTIONS = [
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

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),      # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),      # index
    (0, 9), (9, 10), (10, 11), (11, 12), # middle
    (0, 13), (13, 14), (14, 15), (15, 16), # ring
    (0, 17), (17, 18), (18, 19), (19, 20), # pinky
    (0, 5), (5, 9), (9, 13), (13, 17), (0, 17) # palm
]

# Professional Palette (BGR)
COL_BG      = (13, 17, 23)
COL_BONE    = (180, 180, 180)
COL_BONE_HI = (240, 240, 240)
COL_GRID    = (45, 40, 35)
COL_GRID_HI = (100, 90, 80)

PALETTE_BODY = {
    "head":   (255, 255, 255),
    "torso":  (160, 160, 160),
    "arm_l":  (200, 100, 50),
    "arm_r":  (50, 120, 200),
    "leg_l":  (180, 80, 40),
    "leg_r":  (40, 100, 180),
    "accent": (0, 220, 120),
}

PALETTE_HAND = {
    "thumb" : (50, 120, 255),
    "index" : (255, 100, 50),
    "middle": (50, 200, 100),
    "ring"  : (50, 255, 255),
    "pinky" : (200, 50, 255),
    "wrist" : (255, 255, 255),
}


def _get_body_color(idx: int) -> tuple[int, int, int]:
    """Return a base BGR color for a body joint index."""
    if idx in _HEAD_IDS:  return PALETTE_BODY["head"]
    if idx in _TORSO_IDS: return PALETTE_BODY["torso"]
    if idx in _ARM_IDS:   return PALETTE_BODY["arm_l"] if idx % 2 != 0 else PALETTE_BODY["arm_r"]
    if idx in _LEG_IDS:   return PALETTE_BODY["leg_l"] if idx % 2 != 0 else PALETTE_BODY["leg_r"]
    return PALETTE_BODY["accent"]


def _get_hand_color(idx: int) -> tuple[int, int, int]:
    """Return a base BGR color for a hand joint index."""
    if idx == _WRIST_ID:    return PALETTE_HAND["wrist"]
    if idx in _THUMB_IDS:   return PALETTE_HAND["thumb"]
    if idx in _INDEX_IDS:   return PALETTE_HAND["index"]
    if idx in _MIDDLE_IDS:  return PALETTE_HAND["middle"]
    if idx in _RING_IDS:    return PALETTE_HAND["ring"]
    return PALETTE_HAND["pinky"]


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


# ── Internal Rendering Logic ──────────────────────────────────────────────────

def _draw_sphere(frame, pos, radius, color, light_dir=(0.5, 0.5, 1.0)):
    """Draw a shaded sphere-like circle using radial gradients."""
    cx, cy = int(pos[0]), int(pos[1])
    r = int(radius)
    if r < 1: return

    # Base circle (Ambient + Diffuse)
    # Highlight is shifted towards Top-Left-Front
    hx, hy = cx - r // 3, cy - r // 3
    
    # Layered circles for soft shading
    for i in range(1, 4):
        alpha = i / 3.0
        step_r = int(r * (1.1 - 0.2 * i))
        # Fade from darker to base color
        c = (int(color[0] * alpha), int(color[1] * alpha), int(color[2] * alpha))
        cv2.circle(frame, (cx, cy), step_r, c, -1, cv2.LINE_AA)
    
    # Specular highlight
    h_r = max(1, r // 4)
    cv2.circle(frame, (hx, hy), h_r, (255, 255, 255), -1, cv2.LINE_AA)


def _draw_capsule(frame, p1, p2, r1, r2, color):
    """Draw a volumetric shaded bone (tapered cylinder)."""
    # Calculate screen-space normal to the segment (for width)
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dist = math.sqrt(dx*dx + dy*dy)
    if dist < 1e-6: return

    nx = -dy / dist
    ny =  dx / dist

    # Trapezoid corners
    v1 = (int(p1[0] + nx * r1), int(p1[1] + ny * r1))
    v2 = (int(p1[0] - nx * r1), int(p1[1] - ny * r1))
    v3 = (int(p2[0] - nx * r2), int(p2[1] - ny * r2))
    v4 = (int(p2[0] + nx * r2), int(p2[1] + ny * r2))

    pts = np.array([v1, v2, v3, v4], dtype=np.int32)
    
    # Base fill
    cv2.fillPoly(frame, [pts], color, cv2.LINE_AA)
    
    # Highlight line (Specularity)
    h1 = (int(p1[0] + nx * r1 * 0.4), int(p1[1] + ny * r1 * 0.4))
    h2 = (int(p2[0] + nx * r2 * 0.4), int(p2[1] + ny * r2 * 0.4))
    cv2.line(frame, h1, h2, (255, 255, 255), 1, cv2.LINE_AA)


# ── Renderer ──────────────────────────────────────────────────────────────────

class Model3DRenderer:
    """
    Renders a 3D stick figure from normalised landmarks.

    Call `render(landmarks_3d)` each frame.
    Use `yaw` / `pitch` to control the viewpoint.
    """

    def __init__(self, mode: str = "body", width: int = 960, height: int = 540):
        self.W = width
        self.H = height
        self.mode  = mode    # "body" or "hand"
        self.yaw   = 0.0
        self.pitch = 15.0
        self.zoom  = 1.0

        # Smoothing (exponential moving average for each landmark)
        self._smooth: np.ndarray | None = None
        self._alpha = 0.45   # blending factor (0 = heavy smooth, 1 = raw)

    def render(
        self,
        landmarks: list, # List of (x,y,z) for body, or List of List of (x,y,z) for hands
        anim_tick: int = 0,
    ) -> np.ndarray:
        """Render a 3D frame (BGR)."""
        W, H = self.W, self.H
        frame = np.zeros((H, W, 3), dtype=np.uint8)

        if not landmarks:
            # Draw professional loading circles
            cx, cy = W // 2, H // 2
            for i in range(3):
                a = (anim_tick * (5 + i*2)) % 360
                r = 30 + i * 15
                cv2.ellipse(frame, (cx, cy), (r, r), a, 0, 90, (100, 85, 80), 2, cv2.LINE_AA)
            
            lbl = "SCANNING FOR POSE..." if self.mode == "body" else "SCANNING FOR HANDS..."
            cv2.putText(frame, lbl, (cx - 90, cy + 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (130, 110, 100), 1, cv2.LINE_AA)
            cv2.putText(frame, "Ensure target is clearly visible to the camera", (cx - 150, cy + 125),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 80, 70), 1, cv2.LINE_AA)
            return frame

        try:
            R_main = _rot_y(self.yaw) @ _rot_x(self.pitch)
        except:
            R_main = np.eye(3)

        def _render_entity(entity_lms, entity_idx=0):
            raw = np.array(entity_lms, dtype=np.float64)
            raw[:, 1] = -raw[:, 1]
            raw[:, 2] = -raw[:, 2]

            if self.mode == "body":
                centre = (raw[23] + raw[24]) / 2.0 if len(raw) > 24 else raw.mean(axis=0)
                connections = POSE_CONNECTIONS
                color_func = _get_body_color
                base_thick = 0.006
            else:
                centre = raw[0] # Wrist
                connections = HAND_CONNECTIONS
                color_func = _get_hand_color
                base_thick = 0.004

            pts = raw - centre
            extent = max(np.abs(pts).max(), 0.001)
            b_sc = 0.38 if self.mode == "body" else 0.5
            sc = min(W, H) * b_sc / extent * self.zoom
            pts *= sc

            pts_rot = (R_main @ pts.T).T
            fov_d = 800.0
            cx, cy = W // 2, H // 2
            
            proj = {}
            depths = {}
            for i, (x, y, z) in enumerate(pts_rot):
                pz = z + fov_d
                if pz < 1: pz = 1
                proj[i] = (int(cx + x * fov_d / pz), int(cy - y * fov_d / pz))
                depths[i] = pz

            if entity_idx == 0:
                self._draw_ground(frame, R_main, sc, fov_d, cx, cy)
                self._draw_shadow(frame, pts_rot, R_main, sc, fov_d, cx, cy)

            sizes = {i: max(1.5, 1800.0 * base_thick / d) for i, d in depths.items()}
            
            bone_order = sorted(connections, key=lambda ab: -(depths.get(ab[0],0)+depths.get(ab[1],0))/2)
            for a, b in bone_order:
                if a in proj and b in proj:
                    r1, r2 = sizes.get(a, 2.0)*0.8, sizes.get(b, 2.0)*0.8
                    col = PALETTE_BODY["torso"] if self.mode=="body" and a in _TORSO_IDS and b in _TORSO_IDS else COL_BONE
                    _draw_capsule(frame, proj[a], proj[b], r1, r2, col)

            joint_order = sorted(proj.keys(), key=lambda i: -depths.get(i,0))
            for i in joint_order:
                _draw_sphere(frame, proj[i], sizes.get(i, 3.0), color_func(i))

            if self.mode == "body" and 0 in proj:
                self._draw_head_circle(frame, proj, anim_tick)

        if self.mode == "hand" and landmarks and isinstance(landmarks[0], list):
            for idx, h_lms in enumerate(landmarks):
                if h_lms: _render_entity(h_lms, idx)
        elif landmarks:
            _render_entity(landmarks, 0)

        self._draw_gizmo(frame, R_main)
        cv2.putText(frame, f"MODE: {'Body' if self.mode=='body' else 'Hand Analysis'} | YAW {self.yaw:+.0f}",
                    (15, H - 20), cv2.FONT_HERSHEY_PLAIN, 0.8, (100, 90, 85), 1, cv2.LINE_AA)
        cv2.putText(frame, "3D SPATIAL ANALYZER v2.0", (20, 25), cv2.FONT_HERSHEY_DUPLEX, 0.5, (120, 110, 100), 1, cv2.LINE_AA)

        return frame

    # ── Ground grid ───────────────────────────────────────────────────────

    def _draw_ground(
        self, frame, R, scale, fov_d, cx, cy,
        grid_y: float = -0.35, grid_range: float = 0.8, grid_step: int = 10,
    ):
        """Draw a professional perspective ground grid."""
        lines = []
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
            
            # Simple fade based on Z
            avg_z = (p1[2] + p2[2]) / 2 + fov_d
            alpha = max(0.1, min(0.6, 1.0 - (avg_z / (fov_d * 1.5))))
            color = (int(COL_GRID[0] * alpha), int(COL_GRID[1] * alpha), int(COL_GRID[2] * alpha))

            def _proj(p):
                pz = p[2] + fov_d
                if pz < 1: pz = 1
                return int(cx + p[0] * fov_d / pz), int(cy - p[1] * fov_d / pz)

            cv2.line(frame, _proj(p1), _proj(p2), color, 1, cv2.LINE_AA)

    def _draw_shadow(self, frame, pts, R, scale, fov_d, cx, cy, grid_y=-0.35):
        """Draw a soft shadow on the floor below the skeleton center."""
        if len(pts) < 1: return
        
        # Center of hips or mean
        com = (pts[23] + pts[24]) / 2.0 if len(pts) > 24 else pts.mean(axis=0)
        sx, sz = com[0], com[2]
        y = grid_y * scale
        
        # Draw 2 squashed ellipses for a soft shadow effect
        for r_ext, alpha in [(30.0, 0.2), (15.0, 0.4)]:
            p_center = R @ np.array([sx, y, sz])
            pz = p_center[2] + fov_d
            if pz < 1: continue
            
            sc = fov_d / pz
            center_pix = (int(cx + p_center[0] * sc), int(cy - p_center[1] * sc))
            
            # Use overlay to blend shadow
            overlay = frame.copy()
            cv2.ellipse(overlay, center_pix, (int(r_ext * sc * 2.5), int(r_ext * sc)), 
                        0, 0, 360, (0,0,0), -1, cv2.LINE_AA)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    def _draw_gizmo(self, frame, R):
        """Small XYZ Axis Gizmo in the bottom right corner."""
        size = 35
        base_x, base_y = self.W - 60, self.H - 60
        
        axes = [
            ((size, 0, 0), (0, 0, 255), "X"), # Red
            ((0, size, 0), (0, 255, 0), "Y"), # Green
            ((0, 0, size), (255, 0, 0), "Z"), # Blue
        ]
        
        for (vec, col, label) in axes:
            p = R @ np.array(vec)
            # Simple orthographic for gizmo
            end_x = int(base_x + p[0])
            end_y = int(base_y - p[1])
            cv2.line(frame, (base_x, base_y), (end_x, end_y), col, 2, cv2.LINE_AA)
            cv2.putText(frame, label, (end_x + 2, end_y), 
                        cv2.FONT_HERSHEY_PLAIN, 0.7, col, 1, cv2.LINE_AA)

    def _draw_head_circle(self, frame, proj, anim_tick):
        """Draw an animated circle around the facial landmarks for Body Pose."""
        if 0 not in proj: return
        max_dist = 0.0
        p0 = np.array(proj[0])
        # Check distance from nose to eyes, ears, mouth (indices 1 to 10)
        for i in range(1, 11):
            if i in proj:
                d = np.linalg.norm(np.array(proj[i]) - p0)
                if d > max_dist: max_dist = float(d)
        
        # Make the radius large enough to cover all face points
        base_r = max_dist * 1.5 if max_dist > 5.0 else 16.0
        head_r = int(base_r + 3 * math.sin(anim_tick * 0.12))
        cv2.circle(frame, proj[0], head_r, (200, 200, 200), 1, cv2.LINE_AA)

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
        cv2.putText(result, "SYSTEM IDLE  •  INTERACTIVE VIEWPORT",
                    (30, self.H - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (100, 90, 80), 1, cv2.LINE_AA)

        return result

    def close(self):
        """No-op for OpenCV renderer, matches API in app.py."""
        pass
