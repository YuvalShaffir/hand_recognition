import numpy as np

from ..domain import Angles, Movement, Pose, QuantizedAngles
from ..stage import OptionalStage

DEFAULT_BIN_SIZE_DEG = 15.0
DEFAULT_HYSTERESIS_DEG = 4.0


def quantize_angles(
    angles: Angles,
    previous: QuantizedAngles | None,
    bin_size: float = DEFAULT_BIN_SIZE_DEG,
    hysteresis: float = DEFAULT_HYSTERESIS_DEG,
) -> QuantizedAngles:
    """Snap angles to bin centers, holding the previous bin until the raw
    angle moves past its edge (+ hysteresis margin), to stop boundary jitter
    from producing spurious bin changes.
    """
    if previous is None:
        return np.round(angles / bin_size) * bin_size

    low = previous - bin_size / 2 - hysteresis
    high = previous + bin_size / 2 + hysteresis
    moved = (angles < low) | (angles > high)
    snapped = np.round(angles / bin_size) * bin_size
    return np.where(moved, snapped, previous)


class MovementExtractor(OptionalStage[Pose, Movement]):
    """Poses in, movements out: a quantized pose is emitted only when it
    differs from the one before, so a hand held still produces nothing and
    a gesture becomes a record of changes rather than of elapsed frames."""

    def __init__(
        self,
        bin_size: float = DEFAULT_BIN_SIZE_DEG,
        hysteresis: float = DEFAULT_HYSTERESIS_DEG,
    ) -> None:
        self._bin_size = bin_size
        self._hysteresis = hysteresis
        self._current: QuantizedAngles | None = None

    @property
    def bin_size(self) -> float:
        return self._bin_size

    def transform(self, item: Pose) -> Movement | None:
        quantized = quantize_angles(
            item.angles, self._current, self._bin_size, self._hysteresis
        )
        moved = self._current is None or not np.array_equal(quantized, self._current)
        self._current = quantized
        if not moved:
            return None
        return Movement(angles=quantized, timestamp_ms=item.timestamp_ms)

    def reset(self) -> None:
        self._current = None
