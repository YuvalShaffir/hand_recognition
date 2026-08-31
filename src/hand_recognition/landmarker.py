import os
import threading
import urllib.error
import urllib.request
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("GLOG_logtostderr", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import mediapipe as mp  # noqa: E402
from mediapipe.tasks.python import BaseOptions  # noqa: E402
from mediapipe.tasks.python.vision import (  # noqa: E402
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

from .config import LandmarkerConfig

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"


def ensure_model(model_path: str) -> None:
    """Downloads the hand landmarker model on first run if it isn't already
    present locally - it's an ~8MB generated binary asset, kept out of git."""
    path = Path(model_path)
    if path.exists():
        return

    print(f"{path} not found, downloading from {MODEL_URL} ...")
    try:
        urllib.request.urlretrieve(MODEL_URL, path)
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"couldn't download the hand landmarker model: {e}\n"
            f"download it manually from {MODEL_URL} and save it to {path}"
        ) from e
    print(f"saved {path}")


class HandLandmarkBuffer:
    """Thread-safe latest-result holder, filled by MediaPipe's async
    detection callback and read once per frame by the main loop."""

    def __init__(self):
        self._lock = threading.Lock()
        self._landmarks: list[list] = []
        self._world_landmarks: list[list] = []

    def on_result(self, result, output_image, timestamp_ms) -> None:
        thread_name = threading.current_thread().name

        with self._lock:
            self._landmarks = list(result.hand_landmarks)
            self._world_landmarks = list(result.hand_world_landmarks)

        if not result.hand_landmarks:
            return
        top = result.handedness[0][0]
        print(f"{thread_name}: {top.category_name} ~ {top.score * 100:.0f}%")

    def latest(self):
        """Returns (first_hand, first_world_hand, all_hands) - image-space
        and world landmarks for the first detected hand (or None, None),
        plus every detected hand's image-space landmarks for drawing."""
        with self._lock:
            all_hands = list(self._landmarks)
            hand = all_hands[0] if all_hands else None
            world_hand = self._world_landmarks[0] if self._world_landmarks else None
        return hand, world_hand, all_hands


def create_landmarker(buffer: HandLandmarkBuffer, model_path: str, config: LandmarkerConfig) -> HandLandmarker:
    ensure_model(model_path)
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.LIVE_STREAM,
        num_hands=config.num_hands,
        min_hand_detection_confidence=config.min_hand_detection_confidence,
        min_hand_presence_confidence=config.min_hand_presence_confidence,
        min_tracking_confidence=config.min_tracking_confidence,
        result_callback=buffer.on_result,
    )
    return HandLandmarker.create_from_options(options)
