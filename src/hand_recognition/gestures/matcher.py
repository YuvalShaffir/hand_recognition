import math

import numpy as np

from ..domain import GestureTemplate, Movement
from ..stage import OptionalStage
from .library import GestureLibrary


def dtw_distance(
    a: np.ndarray,
    b: np.ndarray,
    bin_size: float,
    abandon_above: float = math.inf,
) -> float:
    """Average per-step cost (in bins) of the best order-preserving,
    time-elastic alignment between two quantized pose sequences.

    Returns `inf` as soon as the alignment cannot come in under
    `abandon_above`: step costs are non-negative, so a row's cheapest cell
    is a lower bound on the final one and a template that has already lost
    needs no further work.
    """
    n, m = len(a), len(b)
    steps = np.abs(b[None, :, :] - a[:, None, :]).mean(axis=2) / bin_size  # (n, m)
    limit = abandon_above * (n + m)

    # The DP runs on Python floats, not on a numpy array: it is a serial
    # recurrence over ~n*m cells, and element-wise numpy indexing costs far
    # more per cell than the arithmetic does.
    previous = [math.inf] * (m + 1)
    previous[0] = 0.0
    for row in steps.tolist():
        current = [math.inf] * (m + 1)
        cheapest = math.inf
        for j in range(1, m + 1):
            cell = row[j - 1] + min(previous[j], current[j - 1], previous[j - 1])
            current[j] = cell
            if cell < cheapest:
                cheapest = cell
        if cheapest > limit:
            return math.inf
        previous = current
    return previous[m] / (n + m)


class GestureMatcher(OptionalStage[Movement, str]):
    """Movements in, gesture names out. Keeps a rolling window of the
    movements just performed and reports a template's name once that window
    aligns closely enough with it."""

    def __init__(
        self,
        library: GestureLibrary,
        threshold: float = 0.6,
        cooldown_ms: int = 1000,
    ) -> None:
        self._library = library
        self.threshold = threshold
        self.cooldown_ms = cooldown_ms
        self._window: list[np.ndarray] = []
        self._last_match_ms: dict[str, int] = {}

    def transform(self, item: Movement) -> str | None:
        self._window.append(item.angles)
        del self._window[: -2 * self._library.longest]
        window = np.array(self._window)

        best_name = None
        best_score = self.threshold
        for template in self._library.templates:
            if self._cooling_down(template, item.timestamp_ms):
                continue
            length = len(template.frames)
            if len(window) < length:
                continue
            score = dtw_distance(
                window[-2 * length :],
                template.frames,
                template.bin_size,
                abandon_above=best_score,
            )
            if score < best_score:
                best_score = score
                best_name = template.name

        if best_name is not None:
            self._last_match_ms[best_name] = item.timestamp_ms
        return best_name

    def _cooling_down(self, template: GestureTemplate, now_ms: int) -> bool:
        last = self._last_match_ms.get(template.name)
        if last is None:
            return False
        # Timestamps come from independent clocks (`CaptureManager`, the web
        # app); a backwards jump must expire the cooldown, not freeze it.
        return 0 <= now_ms - last < self.cooldown_ms
