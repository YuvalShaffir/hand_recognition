import re
import time
from pathlib import Path

import numpy as np

from .gesture_dtw import GestureTemplate
from .quantize import DEFAULT_BIN_SIZE_DEG, DEFAULT_HYSTERESIS_DEG, quantize_angles

RECORDINGS_DIR = Path("recordings")


class GestureRecorder:
    """Tracks the hand's quantized pose over time and, while recording,
    appends a new frame only when that quantized pose changes (i.e. the
    hand moved past a bin) - giving a run-length-collapsed sequence for
    free instead of one entry per camera frame.
    """

    def __init__(
        self,
        bin_size: float = DEFAULT_BIN_SIZE_DEG,
        hysteresis: float = DEFAULT_HYSTERESIS_DEG,
        recordings_dir: Path = RECORDINGS_DIR,
    ):
        self.bin_size = bin_size
        self.hysteresis = hysteresis
        self.recordings_dir = Path(recordings_dir)
        self.recording = False
        self._quantized: np.ndarray | None = None
        self._frames: list[dict] = []
        self._start_ms = 0

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    @property
    def quantized(self) -> np.ndarray | None:
        """The current running quantized pose, tracked continuously
        regardless of whether a recording is in progress."""
        return self._quantized

    def start(self, now_ms: int) -> None:
        self.recording = True
        self._frames = []
        self._quantized = None
        self._start_ms = now_ms

    def observe(self, angles: np.ndarray, now_ms: int) -> bool:
        """Feed one frame's raw angles. Returns True if the quantized pose
        changed (a new bin was entered)."""
        new_quantized = quantize_angles(
            angles, self._quantized, self.bin_size, self.hysteresis
        )
        moved = self._quantized is None or not np.array_equal(
            new_quantized, self._quantized
        )
        self._quantized = new_quantized

        if self.recording and moved:
            self._frames.append(
                {"t_ms": now_ms - self._start_ms, "angles": new_quantized.tolist()}
            )
        return moved

    def finish(self) -> GestureTemplate:
        """Stops recording and builds a GestureTemplate from the buffered
        frames, entirely in memory - no disk I/O. `self._frames` is left
        intact (only `start()` clears it) so `save()` can still read the
        per-frame timing off it afterwards."""
        self.recording = False
        frames = np.array([f["angles"] for f in self._frames], dtype=np.float64)
        return GestureTemplate(name="", bin_size=self.bin_size, frames=frames)

    def save(self, template: GestureTemplate, name: str | None = None) -> Path:
        self.recordings_dir.mkdir(exist_ok=True)

        slug = (
            re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-") if name else ""
        )
        slug = slug or time.strftime("gesture_%Y%m%d_%H%M%S")

        path = self.recordings_dir / f"{slug}.npz"
        counter = 2
        while path.exists():
            path = self.recordings_dir / f"{slug}-{counter}.npz"
            counter += 1

        t_ms = np.array([f["t_ms"] for f in self._frames], dtype=np.int32)
        angle_bins = np.rint(template.frames / template.bin_size).astype(np.int16)
        np.savez_compressed(
            path,
            bin_size=np.float32(template.bin_size),
            hysteresis=np.float32(self.hysteresis),
            t_ms=t_ms,
            angle_bins=angle_bins,
        )
        return path

    def stop(self, name: str | None = None) -> Path:
        return self.save(self.finish(), name)
