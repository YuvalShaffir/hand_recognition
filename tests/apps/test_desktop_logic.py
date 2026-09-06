"""Omitted from coverage, tested anyway - pure logic that happens to live
in `apps/`."""

import pytest

from hand_recognition.apps.desktop import DesktopApp, _Keys
from hand_recognition.config import AppConfig, KeybindConfig, PathsConfig


@pytest.fixture
def app(tmp_path):
    config = AppConfig(paths=PathsConfig(recordings_dir=str(tmp_path / "recordings")))
    return DesktopApp(config)


def test_keybindings_map_to_ordinals():
    keys = _Keys(KeybindConfig(quit="q", toggle_cursor="c", toggle_record="r"))

    assert (keys.quit, keys.toggle_cursor, keys.toggle_record) == (113, 99, 114)
    assert (keys.tighten, keys.loosen) == (ord("["), ord("]"))


@pytest.mark.parametrize("binding", ["esc", ""])
def test_a_binding_that_is_not_one_character_raises(binding):
    """Config validation rejects these first; `ord` is what would fail if it
    ever stopped."""
    config = KeybindConfig()
    config.quit = binding

    with pytest.raises(TypeError):
        _Keys(config)


def test_tighten_decreases_by_the_step(app):
    app._step_threshold(-1)

    assert app.gestures.threshold == 0.55


def test_loosen_increases_by_the_step(app):
    app._step_threshold(1)

    assert app.gestures.threshold == 0.65


def test_clamps_at_the_minimum(app):
    app.gestures.threshold = app.config.match.threshold_min

    app._step_threshold(-1)

    assert app.gestures.threshold == app.config.match.threshold_min


def test_clamps_at_the_maximum(app):
    app.gestures.threshold = app.config.match.threshold_max

    app._step_threshold(1)

    assert app.gestures.threshold == app.config.match.threshold_max


def test_rounds_to_two_decimals(app):
    """Repeated steps must not accumulate float drift into the HUD."""
    for _ in range(7):
        app._step_threshold(1)
    for _ in range(7):
        app._step_threshold(-1)

    assert app.gestures.threshold == 0.6
