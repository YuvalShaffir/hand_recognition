"""Wiring, not maths - every stage has its own file. Real stages, fake
hands, no vision."""

import pytest

from hand_recognition.config import MatchConfig, QuantizeConfig
from hand_recognition.gestures import GestureLibrary, GesturePipeline

from ..conftest import make_folded_hand, make_hand

PERFORMANCE = (0.0, 0.5, 1.0, 0.5)


@pytest.fixture
def pipeline():
    return GesturePipeline(GestureLibrary(), QuantizeConfig(), MatchConfig())


def perform(pipeline, folds=PERFORMANCE, start_ms=0, step_ms=100):
    return [
        pipeline.apply(
            make_hand(
                landmarks=make_folded_hand(fold),
                world_landmarks=make_folded_hand(fold),
                timestamp_ms=start_ms + index * step_ms,
            )
        )
        for index, fold in enumerate(folds)
    ]


def record(pipeline, name="wave", start_ms=0):
    pipeline.start_recording()
    perform(pipeline, start_ms=start_ms)
    return pipeline.stop_recording(name)


def test_hand_to_gesture_name_end_to_end(pipeline):
    stored = record(pipeline)

    fired = perform(pipeline, start_ms=10_000) + perform(pipeline, start_ms=20_000)

    assert stored == "wave"
    assert stored in fired


def test_recording_diverts_movements_from_the_matcher(pipeline):
    record(pipeline)

    pipeline.start_recording()
    fired = perform(pipeline, start_ms=10_000)

    assert fired == [None] * len(PERFORMANCE)


def test_stop_recording_admits_the_template_and_returns_its_name(pipeline):
    stored = record(pipeline, name="Left Click")

    assert stored == "left-click"
    assert pipeline.library.names == ["left-click"]


def test_the_recorded_gesture_matches_when_performed_again(pipeline):
    """The whole point of the product, in one test."""
    stored = record(pipeline)

    fired = perform(pipeline, start_ms=10_000) + perform(pipeline, start_ms=20_000)

    assert stored in fired


def test_start_recording_resets_the_movement_extractor(pipeline):
    """The template opens with the hand's current pose, not its first
    change - so the first frame of a recording is always collected."""
    perform(pipeline, folds=(0.0,) * 3)

    pipeline.start_recording()
    perform(pipeline, folds=(0.0,), start_ms=1_000)

    assert pipeline.recorded_frame_count == 1


def test_threshold_property_reaches_the_matcher(pipeline):
    pipeline.threshold = 0.42

    assert pipeline.threshold == 0.42
    assert pipeline._matcher.threshold == 0.42


def test_recorded_frame_count_reflects_the_recorder(pipeline):
    pipeline.start_recording()
    perform(pipeline)

    assert pipeline.recorded_frame_count == pipeline._recorder.frame_count > 0


def test_recording_flag_tracks_the_recorder(pipeline):
    assert pipeline.recording is False
    pipeline.start_recording()
    assert pipeline.recording is True
    perform(pipeline)
    pipeline.stop_recording("wave")
    assert pipeline.recording is False


def test_stop_recording_returns_none_when_the_hand_never_moved(pipeline):
    perform(pipeline, folds=(0.0,))
    pipeline.start_recording()

    assert pipeline.stop_recording("wave") is None


def test_stop_recording_without_starting_returns_none(pipeline):
    assert pipeline.stop_recording("wave") is None


def test_none_hands_yield_none(pipeline):
    assert pipeline.apply(None) is None


def test_recording_leaves_the_filesystem_untouched(pipeline, tmp_path, monkeypatch):
    """The web path: nothing in the pipeline persists anything. Persisting
    is the front-end's business, so a browser session writes no files."""
    monkeypatch.chdir(tmp_path)

    record(pipeline)

    assert list(tmp_path.iterdir()) == []
