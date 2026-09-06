import re
import time
from collections.abc import Iterable

from ..domain import MAX_TEMPLATE_FRAMES, GestureTemplate

# Long enough for any name a person types, short enough to leave room for a
# collision counter inside a filesystem's name limit.
MAX_NAME_LENGTH = 64


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return slug[:MAX_NAME_LENGTH].strip("-")


class GestureLibrary:
    """Every gesture template the system knows, held in memory and empty at
    construction - loading and saving belong to `persistence.py`, so one
    library never inherits another session's gestures. The authority on what
    a gesture is called: `add` resolves a requested name into the name the
    template actually ends up with."""

    def __init__(self, templates: Iterable[GestureTemplate] = ()) -> None:
        self._templates: list[GestureTemplate] = []
        for template in templates:
            self.add(template)

    @property
    def templates(self) -> tuple[GestureTemplate, ...]:
        return tuple(self._templates)

    @property
    def names(self) -> list[str]:
        return [template.name for template in self._templates]

    @property
    def longest(self) -> int:
        return max((len(t.frames) for t in self._templates), default=1)

    def add(self, template: GestureTemplate) -> str:
        """Admits a template under a name free of collisions with the ones
        already stored, and returns the name it was given."""
        if len(template.frames) == 0:
            raise ValueError("refusing to add a gesture template with no frames")
        if len(template.frames) > MAX_TEMPLATE_FRAMES:
            raise ValueError(
                f"refusing to add a gesture template of {len(template.frames)} "
                f"frames; the limit is {MAX_TEMPLATE_FRAMES}"
            )

        template.name = self._free_name(template.name)
        self._templates.append(template)
        return template.name

    def _free_name(self, requested: str) -> str:
        stem = _slugify(requested) or time.strftime("gesture_%Y%m%d_%H%M%S")
        taken = set(self.names)
        if stem not in taken:
            return stem
        counter = 2
        while f"{stem}-{counter}" in taken:
            counter += 1
        return f"{stem}-{counter}"
