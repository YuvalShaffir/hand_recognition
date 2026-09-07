import logging

import numpy as np

from ..domain import MAX_TEMPLATE_FRAMES, GestureTemplate, Movement
from ..stage import OptionalStage

logger = logging.getLogger(__name__)


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
        self._movements = []
        return GestureTemplate(name=name, bin_size=self._bin_size, frames=frames)

    def transform(self, item: Movement) -> Movement | None:
        if not self._recording:
            return item
        # A recording nobody stops - a browser tab left open - would
        # otherwise grow a list until the process it shares runs out.
        if len(self._movements) < MAX_TEMPLATE_FRAMES:
            self._movements.append(item)
            if len(self._movements) == MAX_TEMPLATE_FRAMES:
                logger.warning(
                    "recording reached the %d-frame cap; further movements are "
                    "being dropped",
                    MAX_TEMPLATE_FRAMES,
                )
        return None
