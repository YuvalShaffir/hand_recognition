from types import SimpleNamespace

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hand_recognition.config import CursorConfig
from hand_recognition.cursor.screen import ScreenPointConverter
from hand_recognition.domain import NormalizedPoint

SCREEN = (1000, 800)
MARGIN = 0.2


def converter(**overrides):
    config = CursorConfig(region_margin=MARGIN, smoothing=1.0, deadzone=0.0)
    for key, value in overrides.items():
        setattr(config, key, value)
    return ScreenPointConverter(config, SCREEN)


def point(x, y):
    return NormalizedPoint(x=x, y=y)


def test_centre_of_the_active_region_maps_to_the_centre_of_the_screen():
    screen_point = converter().apply(point(0.5, 0.5))

    assert (screen_point.x, screen_point.y) == (500.0, 400.0)


@pytest.mark.parametrize(
    "normalized, expected",
    [((MARGIN, MARGIN), (0.0, 0.0)), ((1 - MARGIN, 1 - MARGIN), (1000.0, 800.0))],
)
def test_region_edges_map_to_screen_edges(normalized, expected):
    screen_point = converter().apply(point(*normalized))

    assert (screen_point.x, screen_point.y) == expected


@pytest.mark.parametrize(
    "normalized, expected",
    [
        ((0.0, 0.5), (0.0, 400.0)),
        ((1.0, 0.5), (1000.0, 400.0)),
        ((0.5, 0.0), (500.0, 0.0)),
        ((0.5, 1.0), (500.0, 800.0)),
    ],
)
def test_outside_the_region_clamps_to_the_screen_edge(normalized, expected):
    screen_point = converter().apply(point(*normalized))

    assert (screen_point.x, screen_point.y) == expected


def test_output_scales_by_the_screen_size():
    config = CursorConfig(region_margin=MARGIN, smoothing=1.0, deadzone=0.0)
    small = ScreenPointConverter(config, (100, 100)).apply(point(0.5, 0.5))
    large = ScreenPointConverter(config, (2000, 2000)).apply(point(0.5, 0.5))

    assert (large.x, large.y) == (small.x * 20, small.y * 20)


def test_first_point_is_emitted_unsmoothed():
    assert converter(smoothing=0.35).apply(point(0.5, 0.5)).x == 500.0


def test_a_move_inside_the_deadzone_emits_nothing():
    stage = converter(deadzone=0.01)
    stage.apply(point(0.5, 0.5))

    assert stage.apply(point(0.505, 0.505)) is None


def test_a_move_past_the_deadzone_emits():
    stage = converter(deadzone=0.01)
    stage.apply(point(0.5, 0.5))

    assert stage.apply(point(0.6, 0.6)) is not None


def test_deadzone_is_measured_against_the_committed_point_not_the_smoothed_one():
    """The two are deliberately different: the committed point is the raw
    hand centre, the smoothed one lags behind it in screen space."""
    stage = converter(deadzone=0.01, smoothing=0.35)
    stage.apply(point(0.5, 0.5))
    stage.apply(point(0.8, 0.8))

    assert stage.apply(point(0.805, 0.805)) is None


def test_smoothing_moves_a_fraction_of_the_way():
    stage = converter(smoothing=0.25)
    stage.apply(point(0.2, 0.2))

    moved = stage.apply(point(0.8, 0.8))

    assert moved.x == pytest.approx(0.25 * 1000)
    assert moved.y == pytest.approx(0.25 * 800)


@settings(max_examples=30)
@given(target=st.floats(0.21, 0.79), steps=st.integers(2, 12))
def test_repeated_identical_points_converge(target, steps):
    stage = converter(smoothing=0.35, deadzone=0.0)
    expected = (target - MARGIN) / (1 - 2 * MARGIN) * SCREEN[0]

    distances = [
        abs(stage.apply(point(target, target)).x - expected) for _ in range(steps)
    ]

    assert distances == sorted(distances, reverse=True)
    assert distances[-1] <= distances[0]


def test_reset_clears_both_histories():
    """Re-entering cursor mode must not snap or interpolate from a stale
    position."""
    stage = converter(smoothing=0.35, deadzone=0.01)
    stage.apply(point(0.2, 0.2))

    stage.reset()

    assert stage.apply(point(0.8, 0.8)).x == 1000.0


def test_zero_smoothing_freezes_the_cursor():
    stage = converter(smoothing=0.0)
    first = stage.apply(point(0.3, 0.3))

    assert stage.apply(point(0.7, 0.7)) == first


def test_smoothing_of_one_disables_smoothing():
    stage = converter(smoothing=1.0)
    stage.apply(point(0.3, 0.3))

    assert stage.apply(point(0.5, 0.5)).x == 500.0


def test_zero_deadzone_emits_every_frame():
    stage = converter(deadzone=0.0)
    stage.apply(point(0.5, 0.5))

    assert stage.apply(point(0.5, 0.5)) is not None


def test_region_margin_of_zero_maps_the_whole_frame():
    stage = converter(region_margin=0.0)

    screen_point = stage.apply(point(0.0, 1.0))

    assert (screen_point.x, screen_point.y) == (0.0, 800.0)


def test_region_margin_of_half_collapses_the_span():
    """Config validation forbids it precisely because the maths divides by
    zero here."""
    with pytest.raises(ValueError, match="region_margin"):
        CursorConfig(region_margin=0.5)

    unchecked = SimpleNamespace(region_margin=0.5, smoothing=1.0, deadzone=0.0)
    with pytest.raises(ZeroDivisionError):
        ScreenPointConverter(unchecked, SCREEN).apply(point(0.5, 0.5))


def test_passes_none_through():
    assert converter().apply(None) is None
