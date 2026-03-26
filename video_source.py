"""
video_source.py
Abstraction over OpenCV VideoCapture for both file paths and webcam indices.
"""

import cv2
import numpy as np
from typing import Optional


class VideoSource:
    """Opens either a video file or a webcam and provides frames on demand."""

    def __init__(self, source: int | str):
        """
        Args:
            source: An integer webcam index (e.g. 0) or a file path string.
        """
        self._source = source
        self._cap: Optional[cv2.VideoCapture] = None
        self._is_file = isinstance(source, str)

    # ------------------------------------------------------------------ #
    #  Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    def open(self) -> bool:
        """Open the video source. Returns True on success."""
        self._cap = cv2.VideoCapture(self._source)
        return self._cap.isOpened()

    def release(self) -> None:
        """Release the underlying VideoCapture."""
        if self._cap and self._cap.isOpened():
            self._cap.release()
        self._cap = None

    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    # ------------------------------------------------------------------ #
    #  Frame access                                                        #
    # ------------------------------------------------------------------ #

    def get_frame(self) -> Optional[np.ndarray]:
        """
        Read the next frame.

        Returns:
            BGR numpy array, or None if the stream ended / failed.
        """
        if not self.is_open():
            return None
        ret, frame = self._cap.read()
        if not ret:
            return None
        return frame

    # ------------------------------------------------------------------ #
    #  Metadata                                                            #
    # ------------------------------------------------------------------ #

    @property
    def fps(self) -> float:
        if not self.is_open():
            return 30.0
        return self._cap.get(cv2.CAP_PROP_FPS) or 30.0

    @property
    def width(self) -> int:
        if not self.is_open():
            return 0
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        if not self.is_open():
            return 0
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def total_frames(self) -> int:
        """Total frame count (only meaningful for file sources)."""
        if not self.is_open():
            return 0
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def is_file_source(self) -> bool:
        return self._is_file
