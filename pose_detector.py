# -*- coding: utf-8 -*-
"""
pose_detector.py
Detects human body pose landmarks using the NEW MediaPipe Tasks API
(compatible with mediapipe >= 0.10.x).

On first run, the lite pose-landmarker model (~9 MB) is auto-downloaded.
"""

import os
import urllib.request
import cv2
import mediapipe as mp
import numpy as np

# ── Model setup ───────────────────────────────────────────────────────────────
_MODEL_DIR  = os.path.dirname(os.path.abspath(__file__))
_MODEL_FILE = os.path.join(_MODEL_DIR, "pose_landmarker_lite.task")
_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/"
    "pose_landmarker_lite.task"
)

# Body-part skeleton connections (landmark index pairs)
_CONNECTIONS = [
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

_DOT_COLOR  = (0, 255, 127)   # neon green
_LINE_COLOR = (255, 165, 0)   # orange


def _ensure_model(progress_cb=None) -> str:
    """Download the model if it is not already present. Returns the file path."""
    if not os.path.exists(_MODEL_FILE):
        if progress_cb:
            progress_cb("Downloading pose model (~9 MB)…")

        def _report(block, block_size, total):
            if total > 0 and progress_cb:
                pct = min(block * block_size * 100 // total, 100)
                progress_cb(f"Downloading pose model… {pct}%")

        urllib.request.urlretrieve(_MODEL_URL, _MODEL_FILE, _report)
        if progress_cb:
            progress_cb("Model ready.")
    return _MODEL_FILE


# ── PoseDetector ──────────────────────────────────────────────────────────────
class PoseDetector:
    """
    Wraps the MediaPipe Tasks PoseLandmarker.

    Usage:
        detector = PoseDetector()          # downloads model on first call
        annotated, n = detector.detect(bgr_frame)
        detector.close()
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float  = 0.5,
        progress_cb=None,
    ):
        self._det_conf = min_detection_confidence
        self._trk_conf = min_tracking_confidence

        model_path = _ensure_model(progress_cb)

        BaseOptions          = mp.tasks.BaseOptions
        PoseLandmarker       = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOpts   = mp.tasks.vision.PoseLandmarkerOptions
        RunningMode          = mp.tasks.vision.RunningMode

        opts = PoseLandmarkerOpts(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._lm = PoseLandmarker.create_from_options(opts)
        self._ts = 0          # monotonic timestamp counter (ms)

    # ── Public API ────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> tuple[np.ndarray, int, list]:
        """
        Run pose detection on a BGR frame.

        Returns:
            annotated_frame : BGR frame with skeleton drawn on it
            landmark_count  : number of visible landmarks (0 if none)
            landmarks_3d    : list of (x, y, z) normalised coords, or []
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        self._ts += 33        # ~30 fps step
        result = self._lm.detect_for_video(mp_img, self._ts)

        annotated       = frame.copy()
        landmark_count  = 0
        landmarks_3d: list[tuple[float, float, float]] = []

        if result.pose_landmarks:
            h, w = frame.shape[:2]
            
            # Use real-world 3D coordinates (in meters) for accurate 3D rendering
            # Fall back to normalized if world landmarks aren't available (rare)
            has_world = bool(result.pose_world_landmarks)
            
            for i, landmarks in enumerate(result.pose_landmarks):
                landmark_count = len(landmarks)

                # Collect 3D coordinates
                if has_world:
                    world_lms = result.pose_world_landmarks[i]
                    landmarks_3d = [(lm.x, lm.y, lm.z) for lm in world_lms]
                else:
                    landmarks_3d = [(lm.x, lm.y, lm.z) for lm in landmarks]

                # Map normalised coords → pixel coords for drawing the 2D overlay
                pts: dict[int, tuple[int, int]] = {
                    idx: (int(lm.x * w), int(lm.y * h))
                    for idx, lm in enumerate(landmarks)
                }

                # Draw skeleton lines
                for a, b in _CONNECTIONS:
                    if a in pts and b in pts:
                        cv2.line(annotated, pts[a], pts[b], _LINE_COLOR, 2, cv2.LINE_AA)

                # Draw landmark dots
                for (x, y) in pts.values():
                    cv2.circle(annotated, (x, y), 4, _DOT_COLOR, -1, cv2.LINE_AA)

        return annotated, landmark_count, landmarks_3d

    def update_confidence(self, detection: float, tracking: float) -> None:
        """Recreate the landmarker with updated confidence thresholds."""
        self.close()
        self._det_conf = detection
        self._trk_conf = tracking
        self.__init__(detection, tracking)

    def close(self) -> None:
        """Release MediaPipe resources."""
        try:
            self._lm.close()
        except Exception:
            pass
