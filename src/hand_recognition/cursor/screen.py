from ..config import CursorConfig
from ..domain import NormalizedPoint, ScreenPoint
from ..stage import OptionalStage


class ScreenPointConverter(OptionalStage[NormalizedPoint, ScreenPoint]):
    """Hand centers in, screen pixels out.

    Three things a raw 1:1 mapping gets wrong, all corrected here:
    - the frame edges are hard to reach with a hand, so only a central
      region of the frame (`region_margin` from each edge) is remapped to
      the full screen
    - per-frame landmark noise makes the cursor visibly jitter even when
      the hand is still, so positions are EMA-smoothed
    - EMA alone never fully settles under continued noise (it keeps
      chasing a jittering target), so small moves are additionally held by
      a dead zone - the same hysteresis trick `MovementExtractor` uses for
      angle bins - until the hand moves far enough to "commit"
    """

    def __init__(self, config: CursorConfig, screen_size: tuple[int, int]) -> None:
        self._config = config
        self._screen_width, self._screen_height = screen_size
        self._committed: NormalizedPoint | None = None
        self._smoothed: NormalizedPoint | None = None

    @property
    def screen_size(self) -> tuple[int, int]:
        return self._screen_width, self._screen_height

    @screen_size.setter
    def screen_size(self, value: tuple[int, int]) -> None:
        """Safe to change mid-stream: smoothing and the dead zone both work
        in normalized space, and the size is applied as the last step."""
        self._screen_width, self._screen_height = value

    def transform(self, item: NormalizedPoint) -> ScreenPoint | None:
        if self._held(item):
            return None
        self._committed = item

        region = self._remap(item)
        self._smoothed = self._smooth(region)
        return ScreenPoint(
            x=self._smoothed.x * self._screen_width,
            y=self._smoothed.y * self._screen_height,
        )

    def reset(self) -> None:
        """Drops the smoothing/dead-zone history so re-entering cursor mode
        doesn't snap or interpolate from a stale last position."""
        self._committed = None
        self._smoothed = None

    def _held(self, point: NormalizedPoint) -> bool:
        if self._committed is None:
            return False
        deadzone = self._config.deadzone
        return (
            abs(point.x - self._committed.x) < deadzone
            and abs(point.y - self._committed.y) < deadzone
        )

    def _remap(self, point: NormalizedPoint) -> NormalizedPoint:
        margin = self._config.region_margin
        span = 1 - 2 * margin
        return NormalizedPoint(
            x=min(max((point.x - margin) / span, 0.0), 1.0),
            y=min(max((point.y - margin) / span, 0.0), 1.0),
        )

    def _smooth(self, point: NormalizedPoint) -> NormalizedPoint:
        if self._smoothed is None:
            return point
        weight = self._config.smoothing
        return NormalizedPoint(
            x=self._smoothed.x + weight * (point.x - self._smoothed.x),
            y=self._smoothed.y + weight * (point.y - self._smoothed.y),
        )
