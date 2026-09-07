import json

import pytest

from hand_recognition.config import (
    ActionsConfig,
    AppConfig,
    CameraConfig,
    CursorConfig,
    KeybindConfig,
    LandmarkerConfig,
    MatchConfig,
    PathsConfig,
    QuantizeConfig,
    load_config,
)


def write(tmp_path, data):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_missing_file_yields_all_defaults(tmp_path):
    assert load_config(tmp_path / "nope.json") == AppConfig()


def test_empty_object_yields_all_defaults(tmp_path):
    assert load_config(write(tmp_path, {})) == AppConfig()


def test_partial_section_keeps_sibling_defaults(tmp_path):
    config = load_config(write(tmp_path, {"camera": {"width": 1280}}))

    assert config.camera == CameraConfig(width=1280)
    assert config.camera.height == 480


@pytest.mark.parametrize(
    "section, overrides, expected",
    [
        ("paths", {"model_path": "m.task"}, PathsConfig(model_path="m.task")),
        ("camera", {"index": 2}, CameraConfig(index=2)),
        ("landmarker", {"num_hands": 2}, LandmarkerConfig(num_hands=2)),
        ("quantize", {"bin_size_deg": 10.0}, QuantizeConfig(bin_size_deg=10.0)),
        ("match", {"cooldown_ms": 500}, MatchConfig(cooldown_ms=500)),
        ("cursor", {"smoothing": 0.5}, CursorConfig(smoothing=0.5)),
        ("actions", {"scroll_amount": 40}, ActionsConfig(scroll_amount=40)),
        ("keybindings", {"quit": "x"}, KeybindConfig(quit="x")),
    ],
)
def test_every_section_is_overridable(tmp_path, section, overrides, expected):
    config = load_config(write(tmp_path, {section: overrides}))

    assert getattr(config, section) == expected


def test_unknown_section_raises(tmp_path):
    with pytest.raises(ValueError, match="wobble"):
        load_config(write(tmp_path, {"wobble": {}}))


def test_unknown_key_raises(tmp_path):
    with pytest.raises(ValueError, match="depth"):
        load_config(write(tmp_path, {"camera": {"depth": 3}}))


def test_malformed_json_raises_json_decode_error(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        load_config(path)


def test_directory_passed_as_path_raises(tmp_path):
    """`load_config` stats the path and then reads it; a directory gets past
    the first and fails the second. Pinned rather than blessed - which of
    `IsADirectoryError` and `PermissionError` comes back is the platform's
    choice, so this asserts their common base."""
    with pytest.raises(OSError):
        load_config(tmp_path)


def test_accepts_str_and_path(tmp_path):
    path = write(tmp_path, {"camera": {"width": 800}})

    assert load_config(str(path)) == load_config(path)


def test_null_value_for_a_key(tmp_path):
    with pytest.raises(ValueError, match="CameraConfig.width"):
        load_config(write(tmp_path, {"camera": {"width": None}}))


@pytest.mark.parametrize(
    "data, message",
    [
        ({"quantize": {"bin_size_deg": 0}}, "bin_size_deg"),
        ({"quantize": {"bin_size_deg": 0.0001}}, "bin_size_deg"),
        ({"quantize": {"bin_size_deg": -15.0}}, "bin_size_deg"),
        ({"quantize": {"hysteresis_deg": -1.0}}, "hysteresis_deg"),
        ({"match": {"threshold_min": 1.5, "threshold_max": 0.5}}, "threshold_min"),
        ({"landmarker": {"num_hands": 100000}}, "num_hands"),
        ({"landmarker": {"num_hands": 0}}, "num_hands"),
        ({"cursor": {"deadzone": -0.1}}, "deadzone"),
        ({"cursor": {"smoothing": 1.5}}, "smoothing"),
        ({"cursor": {"smoothing": -0.5}}, "smoothing"),
        ({"cursor": {"region_margin": 0.5}}, "region_margin"),
        ({"camera": {"width": 0}}, "width"),
        ({"camera": {"height": -480}}, "height"),
        ({"camera": {"index": -1}}, "index"),
        ({"actions": {"scroll_amount": -1}}, "scroll_amount"),
        ({"paths": {"recordings_dir": ""}}, "recordings_dir"),
        ({"keybindings": {"quit": "esc"}}, "quit"),
        ({"match": {"threshold_step": 0.0}}, "threshold_step"),
        ({"match": {"cooldown_ms": -1}}, "cooldown_ms"),
        ({"landmarker": {"min_tracking_confidence": 1.2}}, "min_tracking_confidence"),
    ],
)
def test_out_of_range_values_are_rejected(tmp_path, data, message):
    with pytest.raises(ValueError, match=message):
        load_config(write(tmp_path, data))


@pytest.mark.parametrize(
    "data, message",
    [
        ({"camera": {"width": "wide"}}, "CameraConfig.width must be int"),
        ({"camera": {"width": True}}, "CameraConfig.width must be int"),
        ({"quantize": {"bin_size_deg": "15"}}, "bin_size_deg must be float"),
        ({"keybindings": {"quit": 3}}, "quit must be str"),
        ({"match": {"cooldown_ms": 1.5}}, "cooldown_ms must be int"),
    ],
)
def test_wrong_type_is_rejected(tmp_path, data, message):
    with pytest.raises(ValueError, match=message):
        load_config(write(tmp_path, data))


def test_an_integer_is_accepted_where_a_float_is_expected(tmp_path):
    """JSON has one number type; `15` and `15.0` are the same value."""
    config = load_config(write(tmp_path, {"quantize": {"bin_size_deg": 15}}))

    assert config.quantize.bin_size_deg == 15.0


def test_a_utf8_config_file_is_read_as_utf8(tmp_path):
    """JSON is UTF-8 by definition; reading it in the machine's locale
    encoding fails outright on a Windows console codepage."""
    path = tmp_path / "config.json"
    payload = json.dumps({"paths": {"model_path": "modèle.task"}}, ensure_ascii=False)
    path.write_bytes(payload.encode("utf-8"))

    assert load_config(path).paths.model_path == "modèle.task"


def test_the_shipped_config_is_valid():
    assert load_config("config.json").quantize.bin_size_deg > 0
