"""`cv2.VideoCapture` faked; no webcam anywhere in here."""

from unittest import mock

import cv2
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hand_recognition.config import CameraConfig
from hand_recognition.vision import capture as capture_module
from hand_recognition.vision.capture import CaptureManager


class FakeCapture:
    def __init__(self, frames=3, ok=True, opened=True):
        self.frames = frames
        self.ok = ok
        self.opened = opened
        self.released = False
        self.settings: list[tuple[int, float]] = []
        self.reads = 0

    def isOpened(self):
        return self.opened

    def read(self):
        if not self.ok or self.reads >= self.frames:
            return False, None
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        image[0, 0] = self.reads + 1
        self.reads += 1
        return True, image

    def set(self, prop, value):
        self.settings.append((prop, value))

    def release(self):
        self.released = True


@pytest.fixture
def fake_capture(mocker):
    fake = FakeCapture()
    mocker.patch.object(capture_module.cv2, "VideoCapture", return_value=fake)
    return fake


def test_iterating_yields_frames_from_the_capture(fake_capture):
    with CaptureManager(CameraConfig()) as camera:
        frames = list(camera)

    assert len(frames) == 3
    assert all(frame.image.shape == (2, 2, 3) for frame in frames)


def test_frames_are_mirrored_by_default(fake_capture):
    """So the view on screen moves with the user rather than against them."""
    with CaptureManager(CameraConfig()) as camera:
        frame = next(iter(camera))

    assert frame.image[0, 0].tolist() == [0, 0, 0]
    assert frame.image[0, 1].tolist() == [1, 1, 1]


def test_mirroring_can_be_disabled(fake_capture):
    with CaptureManager(CameraConfig(), mirror=False) as camera:
        frame = next(iter(camera))

    assert frame.image[0, 0].tolist() == [1, 1, 1]


def test_camera_config_is_applied(mocker):
    fake = FakeCapture()
    constructor = mocker.patch.object(
        capture_module.cv2, "VideoCapture", return_value=fake
    )

    with CaptureManager(CameraConfig(index=2, width=1280, height=720)):
        pass

    constructor.assert_called_once_with(index=2)
    assert fake.settings == [
        (cv2.CAP_PROP_FRAME_WIDTH, 1280),
        (cv2.CAP_PROP_FRAME_HEIGHT, 720),
    ]


def test_context_manager_releases_the_capture(fake_capture):
    with CaptureManager(CameraConfig()):
        pass

    assert fake_capture.released is True


def test_timestamps_strictly_increase(fake_capture):
    fake_capture.frames = 50

    with CaptureManager(CameraConfig()) as camera:
        stamps = [frame.timestamp_ms for frame in camera]

    assert stamps == sorted(set(stamps))
    assert len(stamps) == 50


def test_timestamps_increase_even_when_the_clock_does_not(fake_capture):
    """MediaPipe's live-stream mode rejects a repeated timestamp outright,
    which is why `_next_timestamp_ms` has a `max(...)` in it."""
    fake_capture.frames = 5

    with mock.patch.object(capture_module.time, "monotonic", return_value=1.0):
        with CaptureManager(CameraConfig()) as camera:
            stamps = [frame.timestamp_ms for frame in camera]

    assert stamps == [1000, 1001, 1002, 1003, 1004]


@settings(max_examples=25)
@given(clock=st.lists(st.floats(0.0, 10.0), min_size=1, max_size=20))
def test_timestamps_survive_a_backwards_clock(clock):
    fake = FakeCapture(frames=len(clock))
    readings = iter(clock)

    with (
        mock.patch.object(capture_module.cv2, "VideoCapture", return_value=fake),
        mock.patch.object(
            capture_module.time, "monotonic", side_effect=lambda: next(readings)
        ),
    ):
        with CaptureManager(CameraConfig()) as camera:
            stamps = [frame.timestamp_ms for frame in camera]

    assert stamps == sorted(set(stamps))
    assert len(stamps) == len(clock)


def test_iterating_before_entering_raises_runtimeerror():
    with pytest.raises(RuntimeError, match="entered"):
        next(iter(CaptureManager(CameraConfig())))


def test_a_failed_read_ends_the_stream(mocker):
    """An unplugged webcam stops the stream cleanly rather than raising."""
    mocker.patch.object(
        capture_module.cv2, "VideoCapture", return_value=FakeCapture(ok=False)
    )

    with CaptureManager(CameraConfig()) as camera:
        assert list(camera) == []


def test_a_closed_capture_yields_nothing(mocker):
    mocker.patch.object(
        capture_module.cv2, "VideoCapture", return_value=FakeCapture(opened=False)
    )

    with CaptureManager(CameraConfig()) as camera:
        assert list(camera) == []


def test_exit_is_safe_when_never_entered():
    CaptureManager(CameraConfig()).__exit__(None, None, None)


def test_exit_twice_is_safe(fake_capture):
    camera = CaptureManager(CameraConfig())
    camera.__enter__()
    camera.__exit__(None, None, None)
    camera.__exit__(None, None, None)

    assert fake_capture.released is True
