import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hand_recognition.gestures.library import GestureLibrary
from hand_recognition.gestures.matcher import GestureMatcher, dtw_distance

from ..conftest import make_movement, make_template

BIN = 15.0


def sequence(values, width=15):
    return np.array([np.full(width, value) for value in values], dtype=np.float64)


def sequences():
    return st.lists(
        st.lists(st.floats(-90.0, 90.0), min_size=3, max_size=3),
        min_size=1,
        max_size=6,
    ).map(lambda rows: np.array(rows, dtype=np.float64))


def test_identical_sequences_cost_zero():
    a = sequence([0.0, 15.0, 30.0])

    assert dtw_distance(a, a, BIN) == 0.0


def test_cost_is_in_bin_units():
    """The cost of a fixed offset depends on the offset in *bins*, not in
    degrees: doubling the bin size and the offset together leaves it
    unchanged."""
    a = sequence([0.0, 0.0, 0.0])

    one_bin = dtw_distance(a, a + BIN, BIN)

    assert one_bin == pytest.approx(0.5)
    assert dtw_distance(a, a + 2 * BIN, 2 * BIN) == pytest.approx(one_bin)
    assert dtw_distance(a, a + 2 * BIN, BIN) == pytest.approx(2 * one_bin)


def test_time_warping_is_free():
    """The property that lets the same gesture be performed fast or slow."""
    a = sequence([0.0, 15.0, 30.0, 45.0])
    stretched = np.repeat(a, 2, axis=0)

    assert dtw_distance(a, stretched, BIN) == 0.0


@settings(max_examples=50)
@given(a=sequences(), b=sequences())
def test_cost_is_symmetric(a, b):
    assert dtw_distance(a, b, BIN) == pytest.approx(dtw_distance(b, a, BIN))


@settings(max_examples=50)
@given(a=sequences(), b=sequences())
def test_cost_is_non_negative(a, b):
    assert dtw_distance(a, b, BIN) >= 0.0


def test_cost_grows_with_divergence():
    a = sequence([0.0, 15.0, 30.0])
    costs = [dtw_distance(a, a + offset * BIN, BIN) for offset in range(5)]

    assert costs == sorted(costs)
    assert len(set(costs)) == len(costs)


@settings(max_examples=50, deadline=None)
@given(a=sequences(), b=sequences(), limit=st.floats(0.0, 10.0))
def test_abandonment_matches_the_full_computation(a, b, limit):
    exact = dtw_distance(a, b, BIN)

    abandoned = dtw_distance(a, b, BIN, abandon_above=limit)

    assert abandoned == exact or (math.isinf(abandoned) and exact >= limit)


def test_abandonment_never_discards_a_winner():
    a = sequence([0.0, 15.0, 30.0])
    b = a + BIN
    exact = dtw_distance(a, b, BIN)

    assert dtw_distance(a, b, BIN, abandon_above=exact + 0.001) == exact


def test_infinite_limit_never_abandons():
    a, b = sequence([0.0, 15.0]), sequence([90.0, 120.0])

    assert math.isfinite(dtw_distance(a, b, BIN, abandon_above=math.inf))


def test_zero_limit_abandons_immediately():
    a, b = sequence([0.0, 15.0]), sequence([90.0, 120.0])

    assert dtw_distance(a, b, BIN, abandon_above=0.0) == math.inf


def test_single_frame_sequences():
    assert dtw_distance(sequence([0.0]), sequence([0.0]), BIN) == 0.0
    assert dtw_distance(sequence([0.0]), sequence([BIN]), BIN) == pytest.approx(0.5)


def test_length_one_against_length_many():
    a = sequence([0.0])
    b = sequence([0.0, 0.0, 0.0])

    assert dtw_distance(a, b, BIN) == 0.0


def test_empty_sequence_behaviour():
    """`n + m == 0` divides by zero; pinned rather than blessed."""
    empty = np.zeros((0, 15))

    assert dtw_distance(empty, sequence([0.0]), BIN) == math.inf
    with pytest.raises(ZeroDivisionError):
        dtw_distance(empty, empty, BIN)


def test_mismatched_widths_raise():
    with pytest.raises(ValueError):
        dtw_distance(sequence([0.0], width=15), sequence([0.0], width=7), BIN)


def test_zero_bin_size_divides_by_zero():
    a, b = sequence([0.0]), sequence([BIN])

    with np.errstate(divide="ignore", invalid="ignore"):
        assert not math.isfinite(dtw_distance(a, b, 0.0))


def test_nan_in_a_sequence_propagates():
    """A single nan poisons every comparison the sequence takes part in -
    which is why nan is rejected at the `.npz` boundary, not here."""
    a = sequence([0.0, 15.0])
    b = a.copy()
    b[0, 0] = np.nan

    assert not math.isfinite(dtw_distance(a, b, BIN))


# --- GestureMatcher ---------------------------------------------------------

WAVE = sequence([0.0, 15.0, 30.0, 45.0])
FAR = sequence([90.0, 90.0, 90.0, 90.0])


def library_of(*named):
    library = GestureLibrary()
    for name, frames in named:
        library.add(make_template(name=name, frames=frames, bin_size=BIN))
    return library


def drive(matcher, frames, start_ms=0, step_ms=100):
    """Every name the matcher fires over one performance. *Which* movement
    it fires on is not fixed - the window slides, so a gesture is recognised
    somewhere inside its performance rather than on its last frame."""
    fired = (
        matcher.apply(make_movement(row, start_ms + index * step_ms))
        for index, row in enumerate(frames)
    )
    return [name for name in fired if name is not None]


@pytest.fixture
def matcher():
    return GestureMatcher(library_of(("wave", WAVE)), threshold=0.6, cooldown_ms=1000)


def test_matches_a_template_performed_exactly(matcher):
    assert "wave" in drive(matcher, WAVE)


def test_matches_a_template_performed_at_a_different_speed(matcher):
    assert "wave" in drive(matcher, np.repeat(WAVE, 2, axis=0))


def test_returns_none_before_enough_movements_accumulate(matcher):
    assert drive(matcher, WAVE[:3]) == []


def test_returns_none_when_nothing_is_close_enough(matcher):
    assert drive(matcher, FAR) == []


def test_picks_the_closest_of_several_templates():
    matcher = GestureMatcher(library_of(("wave", WAVE), ("flat", FAR)))

    assert "wave" in drive(matcher, WAVE)


def test_an_empty_library_never_matches():
    matcher = GestureMatcher(GestureLibrary())

    assert drive(matcher, WAVE) == []


def test_tighter_threshold_rejects_a_loose_performance():
    matcher = GestureMatcher(library_of(("wave", WAVE)), threshold=0.1)

    assert drive(matcher, WAVE + BIN) == []


def test_looser_threshold_accepts_it():
    matcher = GestureMatcher(library_of(("wave", WAVE)), threshold=0.9)

    assert "wave" in drive(matcher, WAVE + BIN)


def test_threshold_is_mutable_mid_stream():
    matcher = GestureMatcher(library_of(("wave", WAVE)), threshold=0.1)
    assert drive(matcher, WAVE + BIN) == []

    matcher.threshold = 0.9

    assert "wave" in drive(matcher, WAVE + BIN, start_ms=10_000)


def test_a_match_suppresses_the_same_gesture_within_the_cooldown(matcher):
    drive(matcher, WAVE)

    assert drive(matcher, WAVE, start_ms=310, step_ms=1) == []


def test_the_same_gesture_matches_again_after_the_cooldown(matcher):
    drive(matcher, WAVE)

    assert "wave" in drive(matcher, WAVE, start_ms=10_000)


def test_cooldown_is_per_template():
    """Gesture B still matches while A is cooling down. B is performed
    twice so the window has room to clear A's movements out of it."""
    other = sequence([180.0, 165.0, 150.0, 135.0])
    matcher = GestureMatcher(library_of(("wave", WAVE), ("other", other)))
    drive(matcher, WAVE)

    fired = drive(matcher, np.vstack([other, other]), start_ms=310, step_ms=1)

    assert "other" in fired
    assert "wave" not in fired


def test_zero_cooldown_allows_consecutive_matches():
    matcher = GestureMatcher(library_of(("wave", WAVE)), cooldown_ms=0)
    drive(matcher, WAVE)

    assert "wave" in drive(matcher, WAVE, start_ms=400, step_ms=1)


def test_a_backwards_timestamp_does_not_permanently_suppress(matcher):
    """`CaptureManager` and `web.py` derive timestamps from independent
    clocks; a backwards jump must expire the cooldown, not freeze it."""
    drive(matcher, WAVE, start_ms=1_000_000)

    assert "wave" in drive(matcher, WAVE, start_ms=0)


def test_window_is_trimmed_to_twice_the_longest_template(matcher):
    drive(matcher, np.tile(WAVE, (10, 1)))

    assert len(matcher._window) <= 2 * matcher._library.longest


def test_window_survives_a_template_being_added_mid_stream(matcher):
    drive(matcher, WAVE)
    longer = np.repeat(WAVE, 3, axis=0)
    matcher._library.add(make_template(name="slow", frames=longer, bin_size=BIN))

    results = drive(matcher, longer, start_ms=50_000)

    assert len(matcher._window) <= 2 * matcher._library.longest
    assert results


def test_passes_none_through(matcher):
    assert matcher.apply(None) is None


def test_a_template_longer_than_the_window_never_matches(matcher):
    """A template longer than `2 x longest` cannot exist by construction;
    this asserts `longest` really is the bound the window is sized on."""
    lengths = [len(t.frames) for t in matcher._library.templates]

    assert max(lengths) == matcher._library.longest
