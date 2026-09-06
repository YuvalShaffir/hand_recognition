"""The two rules `CLAUDE.md` calls load-bearing: a stage that has nothing to
report returns `None` rather than skipping, and a stage works per-item and
per-stream alike. Both are invisible to per-stage tests - every unit test
still passes when a stage skips, while the `Fork` branches silently
desynchronise."""

import numpy as np
import pytest

from hand_recognition.config import CursorConfig, MatchConfig, QuantizeConfig
from hand_recognition.cursor import CursorPipeline
from hand_recognition.gestures import GestureLibrary, GesturePipeline
from hand_recognition.stage import Fork, OptionalStage, Stage

from .conftest import make_hand


class Append(Stage[str, str]):
    def __init__(self, suffix: str) -> None:
        self.suffix = suffix

    def apply(self, item: str) -> str:
        return item + self.suffix


class Reporting(OptionalStage[int, int]):
    """Reports `item` only when the predicate holds, and records every call
    to `transform` so a `None` reaching it is visible."""

    def __init__(self, reports) -> None:
        self.reports = reports
        self.seen: list[int] = []

    def transform(self, item: int) -> int | None:
        self.seen.append(item)
        return item if self.reports(item) else None


@pytest.fixture
def hands():
    return [make_hand(timestamp_ms=i * 10) for i in range(6)]


def test_chain_applies_left_to_right():
    assert (Append("a") | Append("b")).apply("") == "ab"


def test_chain_is_itself_a_stage():
    left = (Append("a") | Append("b")) | Append("c")
    right = Append("a") | (Append("b") | Append("c"))
    assert left.apply("_") == right.apply("_") == "_abc"


def test_call_maps_a_stream_one_to_one():
    assert list(Append("!")(iter(["a", "b", "c"]))) == ["a!", "b!", "c!"]


def test_fork_pairs_both_branch_results():
    fork = Fork(Append("a"), Append("b"))
    assert fork.apply("_") == ("_a", "_b")


def test_optional_stage_passes_none_through_untransformed():
    stage = Reporting(lambda item: True)
    assert stage.apply(None) is None
    assert stage.seen == []


def test_optional_stage_returning_none_still_yields_an_item():
    stage = Reporting(lambda item: False)
    assert list(stage(iter(range(5)))) == [None] * 5


def test_chain_of_optional_stages_preserves_stream_length():
    chain = (
        Reporting(lambda item: item % 2 == 0)
        | Reporting(lambda item: item % 3 == 0)
        | Reporting(lambda item: item % 5 == 0)
    )
    assert len(list(chain(iter(range(20))))) == 20


def test_fork_branches_stay_in_lockstep():
    left, right = Reporting(lambda i: i % 2 == 0), Reporting(lambda i: i % 3 == 0)
    pairs = list(Fork(left, right)(iter(range(12))))

    assert len(pairs) == 12
    for index, (first, second) in enumerate(pairs):
        assert first == (index if index % 2 == 0 else None)
        assert second == (index if index % 3 == 0 else None)


def test_gesture_pipeline_yields_one_result_per_hand(hands):
    pipeline = GesturePipeline(GestureLibrary(), QuantizeConfig(), MatchConfig())
    stream = [hands[0], None, hands[1], None, hands[2]]

    assert len(list(pipeline(iter(stream)))) == len(stream)


@pytest.mark.parametrize("enabled", [True, False])
def test_cursor_pipeline_yields_one_result_per_hand(hands, enabled):
    pipeline = CursorPipeline(CursorConfig(), (1920, 1080), enabled=enabled)
    stream = [hands[0], None, hands[1], None]

    assert len(list(pipeline(iter(stream)))) == len(stream)


def test_a_stage_that_reports_nothing_never_shortens_a_fork_stream():
    library = GestureLibrary()
    gestures = GesturePipeline(library, QuantizeConfig(), MatchConfig())
    cursor = CursorPipeline(CursorConfig(), (1920, 1080), enabled=True)
    still = make_hand(timestamp_ms=0)

    pairs = list(Fork(gestures, cursor)(iter([still, still, None, still])))

    assert len(pairs) == 4
    assert all(isinstance(pair, tuple) and len(pair) == 2 for pair in pairs)
    assert np.all([pair[0] is None for pair in pairs])
