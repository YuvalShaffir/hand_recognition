import logging
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from ..domain import MAX_TEMPLATE_FRAMES, GestureTemplate
from .angles import NUM_ANGLES

logger = logging.getLogger(__name__)

# `savez_compressed` builds the name; the rest of the 255-byte budget goes to
# the `.npz` suffix and a collision counter.
MAX_FILENAME_STEM = 100

# Unopenable as filenames on Windows, whatever the extension - and Windows is
# the platform the desktop app targets.
WINDOWS_RESERVED = {"con", "prn", "aux", "nul"} | {
    f"{prefix}{digit}" for prefix in ("com", "lpt") for digit in range(1, 10)
}

_BIN_SIZE = "bin_size"
_ANGLE_BINS = "angle_bins"


def load_templates(directory: Path | str) -> list[GestureTemplate]:
    """Reads every `.npz` recording in a directory, newest state of the disk
    at the moment of the call. A missing directory is a first run, not an
    error.

    A recording that fails validation is logged and skipped rather than
    raised: one unreadable file must cost the user that one gesture, not
    every gesture and the program's ability to start."""
    directory = Path(directory)
    if not directory.is_dir():
        return []

    paths = sorted(p for p in directory.iterdir() if p.suffix == ".npz")
    templates = []
    for path in paths:
        try:
            template = _load_one(path)
        except ValueError as error:
            logger.warning("skipping %s: %s", path, error)
            continue
        if template is not None:
            templates.append(template)
    return templates


def save_template(directory: Path | str, template: GestureTemplate) -> Path:
    """Writes one template into a directory under a filename that is safe on
    the filesystems this runs on, and returns the path written."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    path = _free_path(directory, _filename_stem(template.name))
    # Written beside the target and renamed, so an interrupted save leaves
    # no truncated `.npz` behind for the next run to trip over.
    partial = path.with_name(path.name + ".part")
    try:
        with partial.open("wb") as handle:
            np.savez_compressed(
                handle,
                bin_size=np.float32(template.bin_size),
                angle_bins=np.rint(template.frames / template.bin_size).astype(
                    np.int32
                ),
            )
        partial.replace(path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return path


def _filename_stem(name: str) -> str:
    stem = name.strip().strip(".")[:MAX_FILENAME_STEM].strip("-") or "gesture"
    return f"{stem}-" if stem.lower() in WINDOWS_RESERVED else stem


def _free_path(directory: Path, stem: str) -> Path:
    path = directory / f"{stem}.npz"
    counter = 2
    while path.exists():
        path = directory / f"{stem}-{counter}.npz"
        counter += 1
    return path


def _load_one(path: Path) -> GestureTemplate | None:
    _check_declared_array(path)
    with np.load(path) as data:
        bin_size = _checked_bin_size(path, data)
        bins = data[_ANGLE_BINS]

    if len(bins) == 0:
        logger.warning("skipping %s: the recording has no frames", path)
        return None
    frames = bins.astype(np.float64) * bin_size
    if not np.isfinite(frames).all():
        raise ValueError(f"{path}: angle_bins contains non-finite values")
    return GestureTemplate(name=path.stem, bin_size=bin_size, frames=frames)


def _checked_bin_size(path: Path, data: Any) -> float:
    if _BIN_SIZE not in data.files:
        raise ValueError(f"{path}: missing the 'bin_size' entry")
    bin_size = float(data[_BIN_SIZE])
    if not np.isfinite(bin_size) or bin_size <= 0:
        raise ValueError(f"{path}: bin_size must be positive, got {bin_size!r}")
    return bin_size


def _check_declared_array(path: Path) -> None:
    """Validates `angle_bins`' shape and dtype from the `.npy` header alone.

    A recording is attacker-supplied wherever persistence is deployed, and a
    few-KB `.npz` can declare an array of gigabytes; reading it first is the
    bug this avoids."""
    try:
        archive = zipfile.ZipFile(path)
    except zipfile.BadZipFile as error:
        raise ValueError(f"{path}: not a readable .npz file ({error})") from error
    with archive:
        names = set(archive.namelist())
        if f"{_ANGLE_BINS}.npy" not in names:
            raise ValueError(f"{path}: missing the 'angle_bins' entry")
        if f"{_BIN_SIZE}.npy" not in names:
            raise ValueError(f"{path}: missing the 'bin_size' entry")
        with archive.open(f"{_BIN_SIZE}.npy") as member:
            bin_size_shape, _ = _read_header(path, member)
        with archive.open(f"{_ANGLE_BINS}.npy") as member:
            shape, dtype = _read_header(path, member)

    if int(np.prod(bin_size_shape)) != 1:
        raise ValueError(f"{path}: bin_size must be a single number")

    if len(shape) != 2:
        raise ValueError(f"{path}: angle_bins must be 2-dimensional, got {shape}")
    if shape[1] != NUM_ANGLES:
        raise ValueError(
            f"{path}: angle_bins must have {NUM_ANGLES} angles per frame, "
            f"got {shape[1]}"
        )
    if shape[0] > MAX_TEMPLATE_FRAMES:
        raise ValueError(
            f"{path}: angle_bins declares {shape[0]} frames; the limit is "
            f"{MAX_TEMPLATE_FRAMES}"
        )
    if dtype.kind not in "iuf":
        raise ValueError(f"{path}: angle_bins must be numeric, got dtype {dtype}")


def _read_header(path: Path, member: Any) -> tuple[tuple[int, ...], np.dtype]:
    version = np.lib.format.read_magic(member)
    if version == (1, 0):
        shape, _, dtype = np.lib.format.read_array_header_1_0(member)
    elif version == (2, 0):
        shape, _, dtype = np.lib.format.read_array_header_2_0(member)
    else:
        raise ValueError(f"{path}: unsupported .npy format version {version}")
    return shape, dtype
