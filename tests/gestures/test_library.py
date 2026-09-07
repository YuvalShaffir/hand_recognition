"""The library is in-memory and holds no path. Every test here runs with no
filesystem at all - if one needs `tmp_path`, the persistence split has
regressed."""

import numpy as np
import pytest

from hand_recognition.domain import MAX_TEMPLATE_FRAMES, GestureTemplate
from hand_recognition.gestures import library as library_module
from hand_recognition.gestures.library import MAX_NAME_LENGTH, GestureLibrary

from ..conftest import make_template


@pytest.fixture
def frozen_clock(monkeypatch):
    monkeypatch.setattr(
        library_module.time, "strftime", lambda fmt: "gesture_20240101_120000"
    )
    return "gesture_20240101_120000"


def test_a_new_library_is_empty():
    assert GestureLibrary().templates == ()
    assert GestureLibrary().names == []


def test_add_returns_the_stored_name():
    assert GestureLibrary().add(make_template(name="wave")) == "wave"


def test_add_appends_to_templates():
    library = GestureLibrary()
    template = make_template(name="wave")

    library.add(template)

    assert library.templates == (template,)


def test_names_lists_stored_templates_in_insertion_order():
    library = GestureLibrary()
    for name in ("zulu", "alpha", "mike"):
        library.add(make_template(name=name))

    assert library.names == ["zulu", "alpha", "mike"]


def test_longest_reports_the_longest_template_length():
    library = GestureLibrary()
    library.add(make_template(name="short", frames=np.zeros((3, 15))))
    library.add(make_template(name="long", frames=np.zeros((9, 15))))

    assert library.longest == 9


def test_seeding_from_loaded_templates():
    loaded = [make_template(name="wave"), make_template(name="point")]

    assert GestureLibrary(loaded).names == ["wave", "point"]


def test_name_is_slugified():
    assert GestureLibrary().add(make_template(name="Left Click")) == "left-click"


def test_collision_appends_a_counter():
    library = GestureLibrary()
    library.add(make_template(name="wave"))

    assert library.add(make_template(name="wave")) == "wave-2"


def test_repeated_collisions_increment():
    library = GestureLibrary()
    stored = [library.add(make_template(name="wave")) for _ in range(4)]

    assert stored == ["wave", "wave-2", "wave-3", "wave-4"]


def test_blank_name_falls_back_to_a_timestamp(frozen_clock):
    assert GestureLibrary().add(make_template(name="  ")) == frozen_clock


def test_name_of_only_punctuation_falls_back_to_a_timestamp(frozen_clock):
    assert GestureLibrary().add(make_template(name="!!!")) == frozen_clock


def test_unicode_name_falls_back_to_a_timestamp(frozen_clock):
    """A name with no `[a-z0-9]` in it slugifies to nothing. Silently
    timestamping a name the user typed is surprising, and worth revisiting."""
    assert GestureLibrary().add(make_template(name="✋")) == frozen_clock


def test_add_rejects_a_template_with_no_frames():
    with pytest.raises(ValueError, match="no frames"):
        GestureLibrary().add(make_template(frames=np.zeros((0, 15))))


def test_add_rejects_a_template_longer_than_the_cap():
    oversized = GestureTemplate(
        name="hostile",
        bin_size=15.0,
        frames=np.zeros((MAX_TEMPLATE_FRAMES + 1, 15)),
    )

    with pytest.raises(ValueError, match="limit"):
        GestureLibrary().add(oversized)


def test_longest_defaults_to_one_when_empty():
    """The matcher sizes its window off this; zero would divide by zero."""
    assert GestureLibrary().longest == 1


def test_templates_property_is_a_tuple():
    """Callers cannot mutate the library's list through it."""
    library = GestureLibrary()
    library.add(make_template(name="wave"))

    stored = library.templates

    assert isinstance(stored, tuple)
    assert library.names == ["wave"]


def test_traversal_in_a_name_is_neutralised():
    stored = GestureLibrary().add(make_template(name="../../etc/passwd"))

    assert stored == "etc-passwd"


def test_dot_dot_falls_back_to_a_timestamp(frozen_clock):
    assert GestureLibrary().add(make_template(name="..")) == frozen_clock


def test_windows_reserved_names_are_not_special_cased():
    """`con` is still a reserved device name on Windows; handling that is
    `save_template`'s job, not the library's."""
    assert GestureLibrary().add(make_template(name="CON")) == "con"


def test_very_long_name_is_truncated():
    stored = GestureLibrary().add(make_template(name="w" * 5000))

    assert stored == "w" * MAX_NAME_LENGTH
