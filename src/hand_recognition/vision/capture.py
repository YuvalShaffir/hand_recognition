import time
from collections.abc import Iterator
from typing import Any

import cv2

from ..config import CameraConfig
from ..domain import Frame


class CaptureManager:
    """The webcam, as a stream of frames.

    Frames are mirrored so the view on screen moves with the user rather
    than against them, and each one carries a strictly increasing
    timestamp, which is what MediaPipe's live-stream mode requires."""

    def __init__(self, config: CameraConfig, mirror: bool = True) -> None:
        self._config = config
        self._mirror = mirror
        self._capture: Any = None
        self._last_timestamp_ms = -1

    def __enter__(self) -> "CaptureManager":
        capture = cv2.VideoCapture(index=self._config.index)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.height)
        self._capture = capture
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __iter__(self) -> Iterator[Frame]:
        if self._capture is None:
            raise RuntimeError("CaptureManager must be entered before iterating")
        while self._capture.isOpened():
            ok, image = self._capture.read()
            if not ok:
                return
            if self._mirror:
                cv2.flip(image, 1, dst=image)
            yield Frame(image=image, timestamp_ms=self._next_timestamp_ms())

    def _next_timestamp_ms(self) -> int:
        now_ms = int(time.monotonic() * 1000)
        self._last_timestamp_ms = max(now_ms, self._last_timestamp_ms + 1)
        return self._last_timestamp_ms
