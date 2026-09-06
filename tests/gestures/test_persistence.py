"""The only file in the suite that touches the filesystem. A `.npz` is
attacker-supplied wherever persistence is deployed, so all of that hostility
is concentrated here."""

import io
import zipfile

import numpy as np
import pytest

from hand_recognition.domain import MAX_TEMPLATE_FRAMES, GestureTemplate
from hand_recognition.gestures.persistence import (
    MAX_FILENAME_STEM,
    load_templates,
    save_template,
)

from ..conftest import make_template

BIN = 15.0


def write_npz(path, bin_size=BIN, angle_bins=None, **extra):
    if angle_bins is None:
        angle_bins = np.zeros((3, 15), dtype=np.int16)
    payload = {"angle_bins": angle_bins, **extra}
    if bin_size is not None:
        payload["bin_size"] = np.float32(bin_size)
    np.savez_compressed(path, **payload)
    return path


def test_save_then_load_round_trips_a_template(tmp_path):
    frames = np.tile(np.arange(15) * BIN, (4, 1)).astype(np.float64)
    save_template(tmp_path, make_template(name="wave", frames=frames))

    (loaded,) = load_templates(tmp_path)

    assert loaded.name == "wave"
    assert loaded.bin_size == BIN
    assert np.allclose(loaded.frames, frames)


def test_load_returns_templates_sorted_by_name(tmp_path):
    for name in ("zulu", "alpha", "mike"):
        save_template(tmp_path, make_template(name=name))

    assert [t.name for t in load_templates(tmp_path)] == ["alpha", "mike", "zulu"]


def test_save_creates_the_directory_if_absent(tmp_path):
    target = tmp_path / "recordings" / "nested"

    path = save_template(target, make_template(name="wave"))

    assert path.exists()


def test_load_from_a_missing_directory_returns_empty(tmp_path):
    assert load_templates(tmp_path / "nope") == []


def test_load_ignores_non_npz_files(tmp_path):
    (tmp_path / "notes.txt").write_text("hello")
    save_template(tmp_path, make_template(name="wave"))

    assert [t.name for t in load_templates(tmp_path)] == ["wave"]


def test_quantization_survives_the_int16_round_trip(tmp_path):
    frames = np.tile(np.arange(15) * BIN, (3, 1)).astype(np.float64)
    save_template(tmp_path, make_template(name="wave", frames=frames))

    (loaded,) = load_templates(tmp_path)

    assert np.abs(loaded.frames - frames).max() < BIN / 2


def test_a_second_save_of_the_same_name_does_not_overwrite(tmp_path):
    first = save_template(tmp_path, make_template(name="wave"))
    second = save_template(tmp_path, make_template(name="wave"))

    assert first != second
    assert len(load_templates(tmp_path)) == 2


def test_empty_template_file_is_skipped(tmp_path, caplog):
    write_npz(tmp_path / "empty.npz", angle_bins=np.zeros((0, 15), dtype=np.int16))

    assert load_templates(tmp_path) == []
    assert "empty.npz" in caplog.text


def test_unreadable_directory_raises(tmp_path):
    locked = tmp_path / "locked"
    locked.mkdir()
    save_template(locked, make_template(name="wave"))
    locked.chmod(0o000)
    try:
        with pytest.raises(PermissionError):
            load_templates(locked)
    finally:
        locked.chmod(0o755)


def test_save_to_a_readonly_directory_raises_oserror(tmp_path):
    readonly = tmp_path / "readonly"
    readonly.mkdir()
    readonly.chmod(0o500)
    try:
        with pytest.raises(OSError):
            save_template(readonly, make_template(name="wave"))
    finally:
        readonly.chmod(0o755)


def test_missing_bin_size_key_is_rejected(tmp_path):
    write_npz(tmp_path / "bad.npz", bin_size=None)

    with pytest.raises(ValueError, match="bin_size"):
        load_templates(tmp_path)


def test_missing_angle_bins_key_is_rejected(tmp_path):
    np.savez_compressed(tmp_path / "bad.npz", bin_size=np.float32(BIN))

    with pytest.raises(ValueError, match="angle_bins"):
        load_templates(tmp_path)


def test_wrong_angle_width_is_rejected(tmp_path):
    write_npz(tmp_path / "bad.npz", angle_bins=np.zeros((3, 7), dtype=np.int16))

    with pytest.raises(ValueError, match="15 angles per frame"):
        load_templates(tmp_path)


def test_one_dimensional_angle_bins_is_rejected(tmp_path):
    write_npz(tmp_path / "bad.npz", angle_bins=np.zeros(15, dtype=np.int16))

    with pytest.raises(ValueError, match="2-dimensional"):
        load_templates(tmp_path)


def test_non_numeric_dtype_is_rejected(tmp_path):
    write_npz(tmp_path / "bad.npz", angle_bins=np.full((3, 15), "x", dtype="<U1"))

    with pytest.raises(ValueError, match="numeric"):
        load_templates(tmp_path)


@pytest.mark.parametrize("bin_size", [0.0, -15.0])
def test_zero_or_negative_bin_size_is_rejected(tmp_path, bin_size):
    write_npz(tmp_path / "bad.npz", bin_size=bin_size)

    with pytest.raises(ValueError, match="bin_size must be positive"):
        load_templates(tmp_path)


@pytest.mark.parametrize("poison", [np.nan, np.inf])
def test_nan_or_inf_in_frames_is_rejected(tmp_path, poison):
    frames = np.zeros((3, 15))
    frames[1, 2] = poison
    write_npz(tmp_path / "bad.npz", angle_bins=frames)

    with pytest.raises(ValueError, match="non-finite"):
        load_templates(tmp_path)


def test_oversized_array_is_rejected(tmp_path):
    write_npz(
        tmp_path / "bad.npz",
        angle_bins=np.zeros((MAX_TEMPLATE_FRAMES + 1, 15), dtype=np.int16),
    )

    with pytest.raises(ValueError, match="the limit is"):
        load_templates(tmp_path)


def test_decompression_bomb_is_rejected(tmp_path):
    """A few-KB `.npz` declaring gigabytes. The declared shape is checked
    from the `.npy` header, before anything is materialised."""
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        header, {"descr": "<i2", "fortran_order": False, "shape": (10**9, 15)}
    )
    path = tmp_path / "bomb.npz"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("angle_bins.npy", header.getvalue() + b"\x00" * 32)
        with io.BytesIO() as buffer:
            np.lib.format.write_array(buffer, np.float32(BIN))
            archive.writestr("bin_size.npy", buffer.getvalue())

    assert path.stat().st_size < 2048
    with pytest.raises(ValueError, match="the limit is"):
        load_templates(tmp_path)


def test_a_corrupt_file_is_rejected_by_name(tmp_path):
    """One unreadable recording must name itself rather than raising a
    `BadZipFile` from somewhere inside numpy."""
    (tmp_path / "corrupt.npz").write_bytes(b"not a zip at all")

    with pytest.raises(ValueError, match="corrupt.npz"):
        load_templates(tmp_path)


def test_pickle_payload_is_refused(tmp_path):
    """`allow_pickle` defaults to False; this pins that default, because it
    is the one place a bad file is code execution rather than a crash."""
    payload = np.empty((3, 15), dtype=object)
    payload[:] = None
    np.savez(tmp_path / "bad.npz", bin_size=np.float32(BIN), angle_bins=payload)

    with pytest.raises(ValueError):
        load_templates(tmp_path)


def test_very_long_filename_is_truncated(tmp_path):
    """A 5,000-character name exceeds the 255-byte filesystem limit, which
    `savez_compressed` reports as an uncaught `OSError`."""
    path = save_template(tmp_path, make_template(name="w" * 5000))

    assert len(path.name) <= MAX_FILENAME_STEM + len(".npz")
    assert path.exists()


@pytest.mark.parametrize("reserved", ["con", "CON", "nul", "lpt1"])
def test_windows_reserved_filename_is_avoided(tmp_path, reserved):
    """`con.npz` is unopenable on Windows, which is the desktop app's
    platform."""
    path = save_template(tmp_path, make_template(name=reserved))

    assert path.stem.lower() not in {"con", "nul", "lpt1", "prn", "aux"}


def test_a_template_named_only_dots_still_gets_a_filename(tmp_path):
    path = save_template(tmp_path, GestureTemplate("..", BIN, np.zeros((2, 15))))

    assert path.name == "gesture.npz"
