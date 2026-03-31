# -*- coding: utf-8 -*-
"""
detectors.py
Unified detection module for Body Pose and Hand Pose using MediaPipe Tasks API.
"""

import os
import urllib.request
import cv2
import mediapipe as mp
import numpy as np

_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Configuration ─────────────────────────────────────────────────────────────
MODELS = {
    "pose": {
        "file": os.path.join(_MODEL_DIR, "pose_landmarker_lite.task"),
        "url":  (
            "https://storage.googleapis.com/mediapipe-models/"
            "pose_landmarker/pose_landmarker_lite/float16/latest/"
            "pose_landmarker_lite.task"
        ),
        "name": "Body Pose Model",
    },
    "hand": {
        "file": os.path.join(_MODEL_DIR, "hand_landmarker.task"),
        "url":  (
            "https://storage.googleapis.com/mediapipe-models/"
            "hand_landmarker/hand_landmarker/float16/latest/"
            "hand_landmarker.task"
        ),
        "name": "Hand Pose Model",
    }
}

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
    (5, 9), (9, 13), (13, 17), (0, 5), (0, 17) # palm
]

_DOT_COLOR  = (0, 255, 127)   # neon green
_LINE_COLOR = (255, 165, 0)   # orange

def _ensure_model(key: str, progress_cb=None) -> str:
    """Download model if not present."""
    config = MODELS[key]
    path = config["file"]
    if not os.path.exists(path):
        if progress_cb: progress_cb(f"Downloading {config['name']}…")
        def _report(block, block_size, total):
            if total > 0 and progress_cb:
                pct = min(block * block_size * 100 // total, 100)
                progress_cb(f"Downloading {config['name']}… {pct}%")
        urllib.request.urlretrieve(config["url"], path, _report)
    return path

# ── Detector Interface ────────────────────────────────────────────────────────

class BaseDetector:
    def __init__(self):
        self._lm = None
        self._ts = 0

    def detect(self, frame: np.ndarray) -> tuple[np.ndarray, int, list]:
        raise NotImplementedError

    def close(self):
        try:
            if self._lm: self._lm.close()
        except: pass

# ── Body Pose Detector ────────────────────────────────────────────────────────

class PoseDetector(BaseDetector):
    def __init__(self, det_conf=0.5, trk_conf=0.5, progress_cb=None):
        super().__init__()
        path = _ensure_model("pose", progress_cb)
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOpts = mp.tasks.vision.PoseLandmarkerOptions
        
        opts = PoseLandmarkerOpts(
            base_options=BaseOptions(model_asset_path=path),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            min_pose_detection_confidence=det_conf,
            min_pose_presence_confidence=det_conf,
            min_tracking_confidence=trk_conf,
        )
        self._lm = PoseLandmarker.create_from_options(opts)

    def detect(self, frame: np.ndarray) -> tuple[np.ndarray, int, list]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._ts += 33
        result = self._lm.detect_for_video(mp_img, self._ts)

        annotated = frame.copy()
        landmarks_3d = []
        count = 0

        if result.pose_landmarks:
            h, w = frame.shape[:2]
            for i, lms in enumerate(result.pose_landmarks):
                count = len(lms)
                if result.pose_world_landmarks:
                    landmarks_3d = [(lm.x, lm.y, lm.z) for lm in result.pose_world_landmarks[i]]
                else:
                    landmarks_3d = [(lm.x, lm.y, lm.z) for lm in lms]

                pts = {idx: (int(lm.x * w), int(lm.y * h)) for idx, lm in enumerate(lms)}
                for a, b in POSE_CONNECTIONS:
                    if a in pts and b in pts:
                        cv2.line(annotated, pts[a], pts[b], _LINE_COLOR, 2, cv2.LINE_AA)
                for (x, y) in pts.values():
                    cv2.circle(annotated, (x, y), 4, _DOT_COLOR, -1, cv2.LINE_AA)
        
        return annotated, count, landmarks_3d

# ── Hand Pose Detector ────────────────────────────────────────────────────────

class HandDetector(BaseDetector):
    def __init__(self, det_conf=0.5, trk_conf=0.5, progress_cb=None):
        super().__init__()
        path = _ensure_model("hand", progress_cb)
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOpts = mp.tasks.vision.HandLandmarkerOptions
        
        opts = HandLandmarkerOpts(
            base_options=BaseOptions(model_asset_path=path),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            min_hand_detection_confidence=det_conf,
            min_hand_presence_confidence=det_conf,
            min_tracking_confidence=trk_conf,
            num_hands=2,
        )
        self._lm = HandLandmarker.create_from_options(opts)

    def detect(self, frame: np.ndarray) -> tuple[np.ndarray, int, list]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._ts += 33
        result = self._lm.detect_for_video(mp_img, self._ts)

        annotated = frame.copy()
        landmarks_3d = [] # We'll store multiple hands as a flattened or grouped list
        count = 0

        if result.hand_landmarks:
            h, w = frame.shape[:2]
            for i, lms in enumerate(result.hand_landmarks):
                count += len(lms)
                
                # For 3D rendering, we'll focus on the first detected hand for simplicity in this version
                # or pass all hands if the renderer supports it. 
                # Let's pass all hands as a list of lists.
                if result.hand_world_landmarks and len(result.hand_world_landmarks) > i:
                    wlms = [(lm.x, lm.y, lm.z) for lm in result.hand_world_landmarks[i]]
                    landmarks_3d.append(wlms)
                else:
                    # Fallback to normalized landmarks if world landmarks are missing for this hand
                    wlms = [(lm.x, lm.y, lm.z) for lm in lms]
                    landmarks_3d.append(wlms)

                pts = {idx: (int(lm.x * w), int(lm.y * h)) for idx, lm in enumerate(lms)}
                for a, b in HAND_CONNECTIONS:
                    if a in pts and b in pts:
                        cv2.line(annotated, pts[a], pts[b], (0, 180, 255), 2, cv2.LINE_AA)
                for (x, y) in pts.values():
                    cv2.circle(annotated, (x, y), 3, (255, 255, 255), -1, cv2.LINE_AA)
        
        return annotated, count, landmarks_3d
