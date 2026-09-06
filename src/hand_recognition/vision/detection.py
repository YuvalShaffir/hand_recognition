import os
import threading
from typing import Any

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("GLOG_logtostderr", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2  # noqa: E402
import mediapipe as mp  # noqa: E402
from mediapipe.tasks.python import BaseOptions  # noqa: E402
from mediapipe.tasks.python.vision import (  # noqa: E402
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

from ..config import LandmarkerConfig  # noqa: E402
from ..domain import Detection, Frame, Hand  # noqa: E402
from .model_asset import ensure_model  # noqa: E402
from ..stage import Stage  # noqa: E402


class _LatestHands:
    """Thread-safe latest-result holder, filled by MediaPipe's async
    detection callback and read once per frame by the pipeline."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._hands: tuple[Hand, ...] = ()

    def on_result(self, result: Any, output_image: Any, timestamp_ms: int) -> None:
        hands = tuple(
            Hand(
                landmarks=landmarks,
                world_landmarks=world_landmarks,
                timestamp_ms=timestamp_ms,
            )
            for landmarks, world_landmarks in zip(
                result.hand_landmarks, result.hand_world_landmarks
            )
        )
        with self._lock:
            self._hands = hands

    def get(self) -> tuple[Hand, ...]:
        with self._lock:
            return self._hands


class HandDetector(Stage[Frame, Detection]):
    """Frames in, detected hands out.

    MediaPipe's live-stream mode is asynchronous: a frame is submitted and
    the result arrives on a worker thread some time later, so what comes
    back with each frame is the most recent completed detection rather than
    that frame's own."""

    def __init__(self, model_path: str, config: LandmarkerConfig) -> None:
        self._model_path = model_path
        self._config = config
        self._latest = _LatestHands()
        self._landmarker: Any = None

    def __enter__(self) -> "HandDetector":
        ensure_model(self._model_path)
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self._model_path),
            running_mode=RunningMode.LIVE_STREAM,
            num_hands=self._config.num_hands,
            min_hand_detection_confidence=self._config.min_hand_detection_confidence,
            min_hand_presence_confidence=self._config.min_hand_presence_confidence,
            min_tracking_confidence=self._config.min_tracking_confidence,
            result_callback=self._latest.on_result,
        )
        self._landmarker = HandLandmarker.create_from_options(options)
        self._landmarker.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._landmarker.__exit__(exc_type, exc_val, exc_tb)
        self._landmarker = None

    def apply(self, item: Frame) -> Detection:
        if self._landmarker is None:
            raise RuntimeError("HandDetector must be entered before use")
        rgb = cv2.cvtColor(item.image, cv2.COLOR_BGR2RGB)
        self._landmarker.detect_async(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), item.timestamp_ms
        )
        return Detection(frame=item, hands=self._latest.get())
