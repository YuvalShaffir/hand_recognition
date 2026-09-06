from collections.abc import Iterator

from ..config import MatchConfig, QuantizeConfig
from ..domain import Hand
from ..stage import Stage
from .angles import AngleExtractor
from .library import GestureLibrary
from .matcher import GestureMatcher
from .movement import MovementExtractor
from .recorder import GestureRecorder


class GesturePipeline(Stage[Hand | None, str | None]):
    """Hands in, gesture names out.

    Hand -> pose -> movement -> gesture name, with the recorder able to
    divert the movements to itself while a new gesture is being recorded.
    """

    def __init__(
        self,
        library: GestureLibrary,
        quantize: QuantizeConfig,
        match: MatchConfig,
    ) -> None:
        self._library = library
        self._movements = MovementExtractor(
            bin_size=quantize.bin_size_deg, hysteresis=quantize.hysteresis_deg
        )
        self._recorder = GestureRecorder(bin_size=quantize.bin_size_deg)
        self._matcher = GestureMatcher(
            library, threshold=match.threshold_default, cooldown_ms=match.cooldown_ms
        )
        self._stages: Stage[Hand | None, str | None] = (
            AngleExtractor() | self._movements | self._recorder | self._matcher
        )

    def apply(self, item: Hand | None) -> str | None:
        return self._stages.apply(item)

    def __call__(self, source: Iterator[Hand | None]) -> Iterator[str | None]:
        return self._stages(source)

    @property
    def library(self) -> GestureLibrary:
        return self._library

    @property
    def recording(self) -> bool:
        return self._recorder.recording

    @property
    def recorded_frame_count(self) -> int:
        return self._recorder.frame_count

    @property
    def threshold(self) -> float:
        return self._matcher.threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        self._matcher.threshold = value

    def start_recording(self) -> None:
        """Starts from a clean pose history, so the template opens with the
        pose the hand is in now rather than with its first change."""
        self._movements.reset()
        self._recorder.start()

    def stop_recording(self, name: str = "", persist: bool = True) -> str | None:
        """Finishes the recording and admits it to the library, returning
        the name it was stored under - or None if the hand never moved and
        there is no gesture to store."""
        template = self._recorder.finish(name)
        if len(template.frames) == 0:
            return None
        return self._library.add(template, persist=persist)
