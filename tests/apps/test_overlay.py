"""`cv2` faked; the drawing calls are the only thing to assert."""

import numpy as np
import pytest

from hand_recognition.apps import overlay
from hand_recognition.apps.overlay import draw_cursor_marker
from hand_recognition.domain import ScreenPoint


@pytest.fixture
def image():
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def cv2(mocker):
    return mocker.patch.object(overlay, "cv2")


def test_marker_is_filled_in_the_marker_colour(cv2, image):
    draw_cursor_marker(image, ScreenPoint(x=10.0, y=20.0))

    assert cv2.fillPoly.call_args.kwargs["color"] == overlay.CURSOR_MARKER_COLOR


def test_marker_is_outlined_so_it_survives_a_pale_background(cv2, image):
    draw_cursor_marker(image, ScreenPoint(x=10.0, y=20.0))

    assert cv2.polylines.call_args.kwargs["color"] == (
        overlay.CURSOR_MARKER_OUTLINE_COLOR
    )


def test_the_polygon_tip_sits_on_the_point(cv2, image):
    draw_cursor_marker(image, ScreenPoint(x=10.4, y=20.6))

    (polygon,) = cv2.fillPoly.call_args.args[1]
    assert tuple(polygon[0]) == (10, 20)


def test_the_polygon_keeps_its_shape_wherever_it_is_drawn(cv2, image):
    draw_cursor_marker(image, ScreenPoint(x=100.0, y=200.0))

    (polygon,) = cv2.fillPoly.call_args.args[1]
    offsets = polygon - polygon[0]
    assert offsets.tolist() == [list(p) for p in overlay.CURSOR_MARKER_POINTS]


def test_the_fill_and_the_outline_use_the_same_polygon(cv2, image):
    draw_cursor_marker(image, ScreenPoint(x=5.0, y=6.0))

    filled = cv2.fillPoly.call_args.args[1][0]
    outlined = cv2.polylines.call_args.args[1][0]
    assert np.array_equal(filled, outlined)
