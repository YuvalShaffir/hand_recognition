import pytest

from hand_recognition.config import CursorConfig
from hand_recognition.cursor import CursorPipeline

from ..conftest import make_folded_hand, make_hand

SCREEN = (1000, 800)


def hand_at(x):
    """A hand shifted bodily across the frame, so its centre moves with x."""
    landmarks = make_folded_hand(0.0)
    return make_hand(landmarks=[type(lm)(x=lm.x + x, y=0.5, z=0.0) for lm in landmarks])


@pytest.fixture
def config():
    return CursorConfig(region_margin=0.2, smoothing=1.0, deadzone=0.0)


def test_disabled_pipeline_yields_none(config):
    pipeline = CursorPipeline(config, SCREEN, enabled=False)

    assert pipeline.apply(make_hand()) is None


def test_enabled_pipeline_yields_screen_points(config):
    pipeline = CursorPipeline(config, SCREEN, enabled=True)

    screen_point = pipeline.apply(hand_at(0.5))

    assert screen_point is not None
    assert 0.0 <= screen_point.x <= SCREEN[0]


def test_enabling_resets_the_converter(config):
    """Re-entering cursor mode must not slide the cursor in from where the
    hand was last seen."""
    pipeline = CursorPipeline(config, SCREEN, enabled=True)
    pipeline.apply(hand_at(0.0))
    pipeline.enabled = False

    pipeline.enabled = True

    assert pipeline._converter._committed is None
    assert pipeline._converter._smoothed is None


def test_disabling_and_re_enabling_starts_clean(config):
    pipeline = CursorPipeline(config, SCREEN, enabled=True)
    first = pipeline.apply(hand_at(0.9))
    pipeline.enabled = False
    pipeline.enabled = True

    assert pipeline.apply(hand_at(0.9)) == first


def test_enabling_an_already_enabled_pipeline_does_not_reset(config):
    """The guard in the setter; a spurious reset would jump the cursor."""
    pipeline = CursorPipeline(config, SCREEN, enabled=True)
    pipeline.apply(hand_at(0.5))
    committed = pipeline._converter._committed

    pipeline.enabled = True

    assert pipeline._converter._committed is committed


def test_none_hands_yield_none_when_enabled(config):
    pipeline = CursorPipeline(config, SCREEN, enabled=True)

    assert pipeline.apply(None) is None


@pytest.mark.parametrize("enabled", [True, False])
def test_stream_length_is_preserved_in_both_states(config, enabled):
    pipeline = CursorPipeline(config, SCREEN, enabled=enabled)
    stream = [hand_at(0.1), None, hand_at(0.9), None, hand_at(0.5)]

    assert len(list(pipeline(iter(stream)))) == len(stream)
