import re
import time
from pathlib import Path

import numpy as np

from ..domain import GestureTemplate

RECORDINGS_DIR = Path("recordings")


def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")


class GestureLibrary:
    """Every gesture template the system knows, backed by a directory of
    `.npz` files. The authority on what a gesture is called: `add` resolves
    a requested name into the name the template actually ends up with."""

    def __init__(self, directory: Path | str = RECORDINGS_DIR) -> None:
        self._directory = Path(directory)
        self._templates: list[GestureTemplate] = self._load()

    def _load(self) -> list[GestureTemplate]:
        templates = []
        for path in sorted(self._directory.glob("*.npz")):
            data = np.load(path)
            bin_size = float(data["bin_size"])
            frames = data["angle_bins"].astype(np.float64) * bin_size
            if len(frames) == 0:
                continue
            templates.append(
                GestureTemplate(name=path.stem, bin_size=bin_size, frames=frames)
            )
        return templates

    @property
    def directory(self) -> Path:
        return self._directory

    @property
    def templates(self) -> tuple[GestureTemplate, ...]:
        return tuple(self._templates)

    @property
    def names(self) -> list[str]:
        return [template.name for template in self._templates]

    @property
    def longest(self) -> int:
        return max((len(t.frames) for t in self._templates), default=1)

    def add(self, template: GestureTemplate, persist: bool = True) -> str:
        """Admits a template under a name free of collisions with the ones
        already stored, and returns the name it was given."""
        if len(template.frames) == 0:
            raise ValueError("refusing to add a gesture template with no frames")

        template.name = self._free_name(template.name)
        if persist:
            self._save(template)
        self._templates.append(template)
        return template.name

    def _free_name(self, requested: str) -> str:
        stem = _slugify(requested) or time.strftime("gesture_%Y%m%d_%H%M%S")
        taken = set(self.names)
        if stem not in taken and not (self._directory / f"{stem}.npz").exists():
            return stem
        counter = 2
        while (
            f"{stem}-{counter}" in taken
            or (self._directory / f"{stem}-{counter}.npz").exists()
        ):
            counter += 1
        return f"{stem}-{counter}"

    def _save(self, template: GestureTemplate) -> Path:
        self._directory.mkdir(exist_ok=True)
        path = self._directory / f"{template.name}.npz"
        np.savez_compressed(
            path,
            bin_size=np.float32(template.bin_size),
            angle_bins=np.rint(template.frames / template.bin_size).astype(np.int16),
        )
        return path
