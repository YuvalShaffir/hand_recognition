import numpy as np

from ..domain import GestureTemplate, Movement
from ..stage import OptionalStage


class GestureRecorder(OptionalStage[Movement, Movement]):
    """Sits in the stream and, while recording, collects the movements
    passing through it instead of letting them on to the matcher - so a
    gesture being recorded can't fire a macro as it is performed."""

    def __init__(self, bin_size: float) -> None:
        self._bin_size = bin_size
        self._recording = False
        self._movements: list[Movement] = []

    @property
    def recording(self) -> bool:
        return self._recording

    @property
    def frame_count(self) -> int:
        return len(self._movements)

    def start(self) -> None:
        self._recording = True
        self._movements = []

    def finish(self, name: str = "") -> GestureTemplate:
        """Stops recording and builds a template from the movements
        collected, entirely in memory - persisting it is the gesture
        library's job."""
        self._recording = False
        frames = np.array([m.angles for m in self._movements], dtype=np.float64)
        return GestureTemplate(name=name, bin_size=self._bin_size, frames=frames)

    def transform(self, item: Movement) -> Movement | None:
        if not self._recording:
            return item
        self._movements.append(item)
        return None
