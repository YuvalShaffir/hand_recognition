import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hand_recognition.gestures.movement import MovementExtractor, quantize_angles

from ..conftest import make_pose

BIN = 15.0
HYSTERESIS = 4.0
MARGIN = BIN / 2 + HYSTERESIS


def angles(value: float, count: int = 15) -> np.ndarray:
    return np.full(count, value, dtype=np.float64)


@pytest.fixture
def extractor():
    return MovementExtractor(bin_size=BIN, hysteresis=HYSTERESIS)


def drive(extractor, values, start_ms=0):
    return [
        extractor.apply(make_pose(angles(value), start_ms + index))
        for index, value in enumerate(values)
    ]


def test_first_pose_always_emits_a_movement(extractor):
    assert extractor.apply(make_pose(angles(0.0))) is not None


def test_unchanged_pose_emits_nothing(extractor):
    results = drive(extractor, [30.0] * 5)

    assert results[0] is not None
    assert results[1:] == [None] * 4


def test_changed_pose_emits_a_movement(extractor):
    results = drive(extractor, [0.0, 90.0])

    assert results[1] is not None


def test_angles_snap_to_bin_centres(extractor):
    movement = extractor.apply(make_pose(angles(37.0)))

    assert np.allclose(movement.angles, 30.0)
    assert np.allclose(np.remainder(movement.angles, BIN), 0.0)


def test_movement_carries_the_pose_timestamp(extractor):
    assert extractor.apply(make_pose(angles(0.0), 4321)).timestamp_ms == 4321


def test_jitter_inside_the_margin_does_not_change_the_bin(extractor):
    jitter = [0.0] + [MARGIN - 1, -(MARGIN - 1)] * 10

    emitted = [m for m in drive(extractor, jitter) if m is not None]

    assert len(emitted) == 1


def test_crossing_the_margin_commits_the_new_bin(extractor):
    emitted = [m for m in drive(extractor, [0.0, MARGIN + 1]) if m is not None]

    assert len(emitted) == 2
    assert np.allclose(emitted[1].angles, BIN)


def test_hysteresis_is_per_joint(extractor):
    extractor.apply(make_pose(angles(0.0)))
    moved = angles(0.0)
    moved[3] = 90.0

    movement = extractor.apply(make_pose(moved))

    assert movement is not None
    assert movement.angles[3] == 90.0
    assert np.allclose(np.delete(movement.angles, 3), 0.0)


@settings(max_examples=50)
@given(
    base=st.floats(0.0, 180.0),
    offsets=st.lists(st.floats(-MARGIN + 0.01, MARGIN - 0.01), min_size=1, max_size=20),
)
def test_hysteresis_property(base, offsets):
    """Any sequence that stays within the margin of the committed bin emits
    nothing after the first movement."""
    extractor = MovementExtractor(bin_size=BIN, hysteresis=HYSTERESIS)
    first = extractor.apply(make_pose(angles(base)))
    committed = float(first.angles[0])

    emitted = [
        extractor.apply(make_pose(angles(committed + offset))) for offset in offsets
    ]

    assert emitted == [None] * len(offsets)


def test_reset_clears_the_previous_pose(extractor):
    """What makes `start_recording` open the template with the hand's
    current pose rather than with its first change."""
    extractor.apply(make_pose(angles(0.0)))
    extractor.reset()

    assert extractor.apply(make_pose(angles(0.0))) is not None


def test_zero_bin_size_divides_by_zero():
    """Pinned; config validation makes this unreachable in the app."""
    with np.errstate(divide="ignore", invalid="ignore"):
        quantized = quantize_angles(angles(30.0), None, bin_size=0.0)

    assert np.isnan(quantized).all()


def test_zero_hysteresis_still_quantizes():
    extractor = MovementExtractor(bin_size=BIN, hysteresis=0.0)

    movement = extractor.apply(make_pose(angles(37.0)))

    assert np.allclose(movement.angles, 30.0)


def test_hysteresis_wider_than_the_bin_freezes_the_pose():
    extractor = MovementExtractor(bin_size=BIN, hysteresis=1000.0)

    emitted = [m for m in drive(extractor, [0.0, 90.0, 180.0]) if m is not None]

    assert len(emitted) == 1


def test_nan_angles_never_equal_themselves(extractor):
    """`np.array_equal` is False for nan, so a nan pose looks like a
    movement on every frame - which is why nan is rejected upstream."""
    nan_pose = angles(np.nan)

    emitted = [extractor.apply(make_pose(nan_pose)) for _ in range(3)]

    assert all(movement is not None for movement in emitted)


def test_passes_none_through(extractor):
    assert extractor.apply(None) is None


def test_wrong_angle_count_is_handled(extractor):
    movement = extractor.apply(make_pose(angles(30.0, count=14)))

    assert movement.angles.shape == (14,)
