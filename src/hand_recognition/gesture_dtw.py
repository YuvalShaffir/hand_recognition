from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

RECORDINGS_DIR = Path("recordings")


@dataclass
class GestureTemplate:
    name: str
    bin_size: float
    frames: np.ndarray  # (T, 15) quantized angle vectors, in degrees


def load_templates(directory: Path = RECORDINGS_DIR) -> list[GestureTemplate]:
    templates = []
    for path in sorted(directory.glob("*.npz")):
        data = np.load(path)
        bin_size = float(data["bin_size"])
        frames = data["angle_bins"].astype(np.float64) * bin_size
        if len(frames) == 0:
            continue
        templates.append(GestureTemplate(name=path.stem, bin_size=bin_size, frames=frames))
    return templates


def dtw_distance(a: np.ndarray, b: np.ndarray, bin_size: float) -> float:
    """Average per-step cost (in bins) of the best order-preserving,
    time-elastic alignment between two quantized angle sequences."""
    n, m = len(a), len(b)
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        step_cost = np.abs(b - a[i - 1]).mean(axis=1) / bin_size  # (m,)
        for j in range(1, m + 1):
            cost[i, j] = step_cost[j - 1] + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    return cost[n, m] / (n + m)


class GestureMatcher:
    """Matches a live stream of quantized event-frames (one per bin change,
    same as recording) against templates via DTW, and returns the best
    matching template name once its score clears the threshold.
    """

    def __init__(self, templates: list[GestureTemplate], threshold: float = 0.6, cooldown_ms: int = 1000):
        self.templates = templates
        self.threshold = threshold
        self.cooldown_ms = cooldown_ms
        max_len = max((len(t.frames) for t in templates), default=1)
        self._buffer: deque = deque(maxlen=max_len * 2)
        self._last_trigger_ms: dict[str, int] = {}

    def observe(self, quantized: np.ndarray, now_ms: int) -> str | None:
        self._buffer.append(quantized)

        best_name = None
        best_score = self.threshold
        for template in self.templates:
            last = self._last_trigger_ms.get(template.name, -10**9)
            if now_ms - last < self.cooldown_ms:
                continue

            window = list(self._buffer)[-len(template.frames) * 2 :]
            if len(window) < len(template.frames):
                continue

            score = dtw_distance(np.array(window), template.frames, template.bin_size)
            if score < best_score:
                best_score = score
                best_name = template.name

        if best_name is not None:
            self._last_trigger_ms[best_name] = now_ms
        return best_name
