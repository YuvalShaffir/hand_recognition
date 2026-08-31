import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp

from .config import AppConfig
from .gesture_dtw import GestureMatcher, GestureTemplate, load_templates
from .hand_angles import hand_joint_angles
from .landmarker import HandLandmarkBuffer, create_landmarker
from .recorder import GestureRecorder


def _hand_centroid(landmarks) -> tuple[float, float]:
    """Mean (x, y) of all 21 image-space landmarks, normalized [0,1].

    Duplicated from `cursor_control.hand_centroid` rather than imported -
    `cursor_control.py` imports `pyautogui` at module level, which can fail
    outright on a headless container, and this engine must stay importable
    there (see `docs/ARCHITECTURE.md`'s "Runtime split" section).
    """
    n = len(landmarks)
    return sum(lm.x for lm in landmarks) / n, sum(lm.y for lm in landmarks) / n


@dataclass
class FrameResult:
    hands: list  # image-space landmarks, all hands (for overlay)
    matched: str | None  # gesture name that fired this frame, if any
    cursor_xy: tuple[float, float] | None  # normalized hand centroid, if cursor_mode
    recording: bool
    frame_count: int


class GestureEngine:
    """Transport-agnostic gesture-recognition pipeline: owns the recorder,
    matcher, templates, and the MediaPipe landmarker - everything `App`
    needs for recognition, minus display (`cv2.imshow`/HUD) and OS-input
    side effects (`pyautogui`). Lets a desktop `App` and a headless
    Streamlit video processor share the same recognition logic.
    """

    def __init__(self, config: AppConfig):
        self.config = config
        c = config

        self.match_threshold = c.match.threshold_default
        self.recorder = GestureRecorder(
            bin_size=c.quantize.bin_size_deg,
            hysteresis=c.quantize.hysteresis_deg,
            recordings_dir=Path(c.paths.recordings_dir),
        )
        self.templates: list[GestureTemplate] = load_templates(
            Path(c.paths.recordings_dir)
        )
        self.matcher = GestureMatcher(
            self.templates,
            threshold=self.match_threshold,
            cooldown_ms=c.match.cooldown_ms,
        )
        self.cursor_mode = False
        self.landmark_buffer = HandLandmarkBuffer()
        self._landmarker: Any = None
        self._last_timestamp_ms = -1

    def __enter__(self) -> "GestureEngine":
        self._landmarker = create_landmarker(
            self.landmark_buffer, self.config.paths.model_path, self.config.landmarker
        )
        self._landmarker.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._landmarker.__exit__(exc_type, exc_val, exc_tb)
        self._landmarker = None

    def process_frame(self, frame_bgr, now_ms: int) -> FrameResult:
        """Detects the hand in `frame_bgr` (flipped horizontally in place,
        mirror-view like the desktop tool), advances recording/matching
        state, and reports the result - without drawing anything or firing
        any OS action itself."""
        timestamp_ms = max(now_ms, self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms

        cv2.flip(frame_bgr, 1, dst=frame_bgr)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._landmarker.detect_async(mp_image, timestamp_ms)

        hand, world_hand, all_hands = self.landmark_buffer.latest()

        cursor_xy = None
        if self.cursor_mode and hand is not None:
            cursor_xy = _hand_centroid(hand)

        matched = None
        if hand is not None and world_hand is not None:
            matched = self._observe_gesture(world_hand, now_ms)

        return FrameResult(
            hands=all_hands,
            matched=matched,
            cursor_xy=cursor_xy,
            recording=self.recorder.recording,
            frame_count=self.recorder.frame_count,
        )

    def _observe_gesture(self, world_hand, now_ms: int) -> str | None:
        moved = self.recorder.observe(hand_joint_angles(world_hand), now_ms)
        quantized = self.recorder.quantized
        if moved and not self.recorder.recording and quantized is not None:
            return self.matcher.observe(quantized, now_ms)
        return None

    def start_recording(self) -> None:
        self.recorder.start(int(time.monotonic() * 1000))

    def stop_recording(self, name: str | None) -> GestureTemplate:
        """Finishes recording and, if any frames were captured, appends the
        resulting template (in memory only) to this session's template list
        and rebuilds the matcher - no disk I/O. Use `self.recorder.save(...)`
        separately for callers that also want it persisted to `recordings/`.
        """
        template = self.recorder.finish()
        template.name = (name or "").strip() or time.strftime("gesture_%Y%m%d_%H%M%S")
        if len(template.frames) > 0:
            self.templates.append(template)
            self.matcher = GestureMatcher(
                self.templates,
                threshold=self.match_threshold,
                cooldown_ms=self.config.match.cooldown_ms,
            )
        return template

    def set_cursor_mode(self, on: bool) -> None:
        self.cursor_mode = on

    def set_threshold(self, value: float) -> None:
        self.match_threshold = value
        self.matcher.threshold = value
