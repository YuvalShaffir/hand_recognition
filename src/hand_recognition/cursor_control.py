import pyautogui

pyautogui.FAILSAFE = True


def hand_centroid(landmarks) -> tuple[float, float]:
    """Mean (x, y) of all 21 image-space landmarks, normalized [0,1].

    Used instead of a single fingertip so cursor position stays stable
    while fingers move through a gesture - letting cursor tracking and
    gesture recognition run at the same time off the same hand.
    """
    n = len(landmarks)
    return sum(lm.x for lm in landmarks) / n, sum(lm.y for lm in landmarks) / n


class CursorController:
    """Drives the real OS cursor from a normalized (image-space, [0,1])
    hand position (see `hand_centroid`).

    Three things a raw 1:1 mapping gets wrong, all corrected here:
    - the frame edges are hard to reach with a hand, so only a central
      region of the frame (`region_margin` from each edge) is remapped to
      the full screen
    - per-frame landmark noise makes the cursor visibly jitter even when
      the hand is still, so positions are EMA-smoothed before moving
    - EMA alone never fully settles under continued noise (it keeps
      chasing a jittering target), so small moves are additionally held
      by a dead zone - the same hysteresis trick `quantize.py` uses for
      angle-bin jitter - until the hand moves far enough to "commit"
    """

    def __init__(
        self,
        region_margin: float = 0.2,
        smoothing: float = 0.35,
        deadzone: float = 0.008,
    ):
        self.region_margin = region_margin
        self.smoothing = smoothing
        self.deadzone = deadzone
        self.screen_w, self.screen_h = pyautogui.size()
        self._committed: tuple[float, float] | None = None
        self._smoothed: tuple[float, float] | None = None

    def update(self, x: float, y: float) -> None:
        """Feed one frame's normalized hand (x, y) and move the cursor."""
        if self._committed is not None:
            cx, cy = self._committed
            if abs(x - cx) < self.deadzone and abs(y - cy) < self.deadzone:
                return
        self._committed = (x, y)

        span = 1 - 2 * self.region_margin
        rx = (x - self.region_margin) / span
        ry = (y - self.region_margin) / span
        rx = min(max(rx, 0.0), 1.0)
        ry = min(max(ry, 0.0), 1.0)

        if self._smoothed is None:
            self._smoothed = (rx, ry)
        else:
            sx, sy = self._smoothed
            self._smoothed = (
                sx + self.smoothing * (rx - sx),
                sy + self.smoothing * (ry - sy),
            )

        pyautogui.moveTo(
            self._smoothed[0] * self.screen_w,
            self._smoothed[1] * self.screen_h,
            _pause=False,
        )

    def reset(self) -> None:
        """Drop the smoothing/deadzone history so re-entering cursor mode
        doesn't snap/interpolate from a stale last position."""
        self._committed = None
        self._smoothed = None
