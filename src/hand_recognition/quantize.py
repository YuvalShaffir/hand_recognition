import numpy as np

DEFAULT_BIN_SIZE_DEG = 15.0
DEFAULT_HYSTERESIS_DEG = 4.0


def quantize_angles(
    angles: np.ndarray,
    prev_quantized: np.ndarray | None,
    bin_size: float = DEFAULT_BIN_SIZE_DEG,
    hysteresis: float = DEFAULT_HYSTERESIS_DEG,
) -> np.ndarray:
    """Snap angles to bin centers, holding the previous bin until the raw
    angle moves past its edge (+ hysteresis margin), to stop boundary jitter
    from producing spurious bin changes.
    """
    if prev_quantized is None:
        return np.round(angles / bin_size) * bin_size

    low = prev_quantized - bin_size / 2 - hysteresis
    high = prev_quantized + bin_size / 2 + hysteresis
    moved = (angles < low) | (angles > high)
    snapped = np.round(angles / bin_size) * bin_size
    return np.where(moved, snapped, prev_quantized)
