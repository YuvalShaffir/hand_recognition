"""The invariant bought by making libraries session-scoped: N sessions in
one process do not see each other. Pure stages given their inputs, so the
threads probe locking and shared state rather than timing."""

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

from hand_recognition.config import CursorConfig, MatchConfig, QuantizeConfig
from hand_recognition.cursor.screen import ScreenPointConverter
from hand_recognition.domain import MAX_TEMPLATE_FRAMES, GestureTemplate
from hand_recognition.gestures import GestureLibrary, GesturePipeline
from hand_recognition.gestures.matcher import GestureMatcher
from hand_recognition.gestures.matcher import dtw_distance
from hand_recognition.gestures import matcher as matcher_module
from hand_recognition.gestures.movement import MovementExtractor
from hand_recognition.gestures.recorder import GestureRecorder
from hand_recognition.vision.detection import _LatestHands

from .conftest import make_folded_hand, make_hand, make_movement, make_template

SESSIONS = 16
BIN = 15.0


def performance(session: int) -> tuple[float, ...]:
    base = 0.05 * (session % 5)
    return (base, base + 0.4, base + 0.8, base + 0.4)


def perform(pipeline, session, start_ms=0):
    return [
        pipeline.apply(
            make_hand(
                landmarks=make_folded_hand(fold),
                world_landmarks=make_folded_hand(fold),
                timestamp_ms=start_ms + index * 100,
            )
        )
        for index, fold in enumerate(performance(session))
    ]


def record_session(session: int) -> tuple[list[str], int]:
    pipeline = GesturePipeline(GestureLibrary(), QuantizeConfig(), MatchConfig())
    pipeline.start_recording()
    perform(pipeline, session)
    pipeline.stop_recording(f"gesture-{session}")
    template = pipeline.library.templates[-1]
    return pipeline.library.names, len(template.frames)


def in_parallel(function, count=SESSIONS):
    with ThreadPoolExecutor(max_workers=count) as pool:
        return list(pool.map(function, range(count)))


def test_parallel_pipelines_do_not_share_templates():
    results = in_parallel(record_session)

    assert [names for names, _ in results] == [
        [f"gesture-{session}"] for session in range(SESSIONS)
    ]


def test_parallel_pipelines_do_not_share_movement_state():
    alone = [record_session(session) for session in range(SESSIONS)]

    together = in_parallel(record_session)

    assert together == alone


def test_matcher_cooldown_is_per_instance():
    frames = np.array([np.full(15, value) for value in (0.0, 15.0, 30.0, 45.0)])
    library = GestureLibrary([make_template(name="wave", frames=frames)])
    first, second = GestureMatcher(library), GestureMatcher(library)

    for index, row in enumerate(frames):
        first.apply(make_movement(row, index * 100))
    fired = [
        second.apply(make_movement(row, index * 100))
        for index, row in enumerate(frames)
    ]

    assert "wave" in fired


def test_latest_hands_is_safe_under_concurrent_writes():
    """Every value read must be one complete tuple that was written, never
    a mixture of two."""
    latest = _LatestHands()
    stop = threading.Event()
    seen: list[tuple[int, ...]] = []

    def write(session):
        for _ in range(200):
            result = SimpleNamespace(
                hand_landmarks=[[]] * (session % 3 + 1),
                hand_world_landmarks=[[]] * (session % 3 + 1),
            )
            latest.on_result(result, None, session)

    def read():
        while not stop.is_set():
            hands = latest.get()
            seen.append(tuple(hand.timestamp_ms for hand in hands))

    reader = threading.Thread(target=read)
    reader.start()
    try:
        in_parallel(write, count=8)
    finally:
        stop.set()
        reader.join()

    assert seen
    for stamps in seen:
        assert len(set(stamps)) <= 1
        assert not stamps or len(stamps) == stamps[0] % 3 + 1


def test_no_module_level_mutable_state():
    library_a, library_b = GestureLibrary(), GestureLibrary()
    library_a.add(make_template(name="wave"))

    extractor_a = MovementExtractor(bin_size=BIN)
    extractor_b = MovementExtractor(bin_size=BIN)
    extractor_a.apply(SimpleNamespace(angles=np.zeros(15), timestamp_ms=0))

    recorder_a, recorder_b = GestureRecorder(BIN), GestureRecorder(BIN)
    recorder_a.start()
    recorder_a.apply(make_movement(np.zeros(15)))

    converter_a = ScreenPointConverter(CursorConfig(), (10, 10))
    converter_b = ScreenPointConverter(CursorConfig(), (10, 10))
    converter_a.apply(SimpleNamespace(x=0.5, y=0.5))

    assert library_b.names == []
    assert extractor_b._current is None
    assert (recorder_b.recording, recorder_b.frame_count) == (False, 0)
    assert converter_b._committed is None


def test_matcher_window_never_exceeds_twice_the_longest_template():
    frames = np.array([np.full(15, value) for value in (0.0, 15.0, 30.0, 45.0)])
    library = GestureLibrary([make_template(name="wave", frames=frames)])
    matcher = GestureMatcher(library)

    for index in range(10_000):
        matcher.apply(make_movement(frames[index % 4], index))
        assert len(matcher._window) <= 2 * library.longest


@pytest.mark.parametrize("n, m", [(4, 4), (8, 4), (13, 7)])
def test_dtw_cell_count_stays_within_budget(mocker, n, m):
    spy = mocker.spy(matcher_module.np, "abs")
    a = np.zeros((n, 15))
    b = np.zeros((m, 15))

    dtw_distance(a, b, BIN)

    assert spy.spy_return.shape == (n, m, 15)


def test_recorder_memory_is_bounded_by_the_cap():
    recorder = GestureRecorder(BIN)
    recorder.start()

    for index in range(10_000):
        recorder.apply(make_movement(np.zeros(15), index))

    assert recorder.frame_count == MAX_TEMPLATE_FRAMES


def test_matcher_cost_is_bounded_by_the_template_length_cap():
    """A hostile 10,000-frame template cannot be admitted, so the window -
    and with it the per-frame cost - has a ceiling."""
    hostile = GestureTemplate("hostile", BIN, np.zeros((10_000, 15)))
    library = GestureLibrary()

    with pytest.raises(ValueError):
        library.add(hostile)
    assert library.longest <= MAX_TEMPLATE_FRAMES
