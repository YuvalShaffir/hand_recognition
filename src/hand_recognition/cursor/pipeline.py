from collections.abc import Iterator

from ..config import CursorConfig
from ..domain import Hand, ScreenPoint
from ..stage import Stage
from .center import HandCenterExtractor
from .screen import ScreenPointConverter


class CursorPipeline(Stage[Hand | None, ScreenPoint | None]):
    """Hands in, screen positions out - while enabled. Disabled it yields
    nothing, and enabling it again starts from a clean position history
    rather than sliding over from where the hand was last seen."""

    def __init__(
        self,
        config: CursorConfig,
        screen_size: tuple[int, int],
        enabled: bool = False,
    ) -> None:
        self._converter = ScreenPointConverter(config, screen_size)
        self._stages: Stage[Hand | None, ScreenPoint | None] = (
            HandCenterExtractor() | self._converter
        )
        self._enabled = enabled

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        if value and not self._enabled:
            self._converter.reset()
        self._enabled = value

    def apply(self, item: Hand | None) -> ScreenPoint | None:
        if not self._enabled:
            return None
        return self._stages.apply(item)

    def __call__(self, source: Iterator[Hand | None]) -> Iterator[ScreenPoint | None]:
        return (self.apply(item) for item in source)
