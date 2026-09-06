import numpy as np
import pytest

from hand_recognition.domain import MAX_TEMPLATE_FRAMES
from hand_recognition.gestures.recorder import GestureRecorder

from ..conftest import make_movement

BIN = 15.0


def movements(count: int):
    return [
        make_movement(np.full(15, index % 4 * BIN), index) for index in range(count)
    ]


@pytest.fixture
def recorder():
    return GestureRecorder(bin_size=BIN)


def test_passes_movements_through_when_not_recording(recorder):
    movement = make_movement(np.zeros(15))

    assert recorder.apply(movement) is movement


def test_swallows_movements_while_recording(recorder):
    recorder.start()

    assert recorder.apply(make_movement(np.zeros(15))) is None


def test_collects_movements_while_recording(recorder):
    recorder.start()
    for movement in movements(3):
        recorder.apply(movement)

    assert recorder.frame_count == 3


def test_finish_builds_a_template_of_the_collected_movements(recorder):
    recorder.start()
    for movement in movements(4):
        recorder.apply(movement)

    template = recorder.finish("wave")

    assert template.name == "wave"
    assert template.bin_size == BIN
    assert template.frames.shape == (4, 15)


def test_finish_stops_recording(recorder):
    recorder.start()
    recorder.finish("wave")

    assert recorder.recording is False


def test_start_discards_a_previous_partial_recording(recorder):
    recorder.start()
    for movement in movements(3):
        recorder.apply(movement)

    recorder.start()

    assert recorder.frame_count == 0


def test_finish_without_recording_returns_an_empty_template(recorder):
    template = recorder.finish("wave")

    assert template.frames.shape == (0,)


def test_finish_twice_returns_an_empty_second_template(recorder):
    recorder.start()
    recorder.apply(make_movement(np.zeros(15)))
    recorder.finish("wave")

    assert recorder.finish("wave").frames.shape == (0,)


def test_single_movement_recording(recorder):
    recorder.start()
    recorder.apply(make_movement(np.zeros(15)))

    assert recorder.finish("wave").frames.shape == (1, 15)


def test_passes_none_through_while_recording(recorder):
    recorder.start()

    assert recorder.apply(None) is None
    assert recorder.frame_count == 0


def test_empty_name_is_accepted(recorder):
    recorder.start()
    recorder.apply(make_movement(np.zeros(15)))

    assert recorder.finish("").name == ""


def test_recording_stops_collecting_at_the_cap(recorder):
    """A session that starts recording and walks away would otherwise grow
    a list forever, on a container it shares with other sessions."""
    recorder.start()
    for movement in movements(MAX_TEMPLATE_FRAMES + 500):
        recorder.apply(movement)

    assert recorder.frame_count == MAX_TEMPLATE_FRAMES
    assert recorder.finish("wave").frames.shape == (MAX_TEMPLATE_FRAMES, 15)
