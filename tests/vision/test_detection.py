"""`mediapipe` faked at the `HandLandmarker` seam; no model file, no
download, no graph."""

from types import SimpleNamespace

import numpy as np
import pytest

from hand_recognition.config import LandmarkerConfig
from hand_recognition.domain import Frame
from hand_recognition.vision import detection as detection_module
from hand_recognition.vision.detection import HandDetector, _LatestHands

from ..conftest import make_landmark


def result_with(*hands):
    return SimpleNamespace(
        hand_landmarks=[landmarks for landmarks, _ in hands],
        hand_world_landmarks=[world for _, world in hands],
    )


def test_starts_empty():
    assert _LatestHands().get() == ()


def test_on_result_pairs_landmarks_with_world_landmarks():
    """Pairing these wrongly swaps the two coordinate spaces and breaks the
    angles and the cursor at once."""
    image = [make_landmark(0.1, 0.1)]
    world = [make_landmark(9.0, 9.0)]
    latest = _LatestHands()

    latest.on_result(result_with((image, world)), None, 7)

    (hand,) = latest.get()
    assert hand.landmarks is image
    assert hand.world_landmarks is world


def test_on_result_stamps_the_callback_timestamp():
    latest = _LatestHands()

    latest.on_result(result_with(([], [])), None, 4321)

    assert latest.get()[0].timestamp_ms == 4321


def test_a_result_with_no_hands_clears_the_previous_one():
    latest = _LatestHands()
    latest.on_result(result_with(([], [])), None, 1)

    latest.on_result(result_with(), None, 2)

    assert latest.get() == ()


def test_get_returns_a_tuple_not_a_live_reference():
    latest = _LatestHands()
    latest.on_result(result_with(([], [])), None, 1)
    held = latest.get()

    latest.on_result(result_with(), None, 2)

    assert isinstance(held, tuple)
    assert len(held) == 1


class FakeLandmarker:
    def __init__(self):
        self.submitted: list[tuple] = []
        self.entered = False
        self.exited = False

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, *args):
        self.exited = True

    def detect_async(self, image, timestamp_ms):
        self.submitted.append((image, timestamp_ms))


@pytest.fixture
def landmarker(mocker):
    fake = FakeLandmarker()
    mocker.patch.object(
        detection_module.HandLandmarker, "create_from_options", return_value=fake
    )
    mocker.patch.object(
        detection_module,
        "mp",
        SimpleNamespace(
            Image=lambda image_format, data: SimpleNamespace(
                image_format=image_format, data=data
            ),
            ImageFormat=SimpleNamespace(SRGB="srgb"),
        ),
    )
    mocker.patch.object(detection_module, "ensure_model")
    return fake


@pytest.fixture
def frame():
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    image[..., 0] = 10
    image[..., 2] = 30
    return Frame(image=image, timestamp_ms=99)


def detector(**overrides):
    return HandDetector("model.task", LandmarkerConfig(**overrides))


def test_apply_before_entering_raises_runtimeerror(frame):
    with pytest.raises(RuntimeError, match="entered"):
        detector().apply(frame)


def test_apply_submits_the_frame_asynchronously(landmarker, frame):
    with detector() as stage:
        stage.apply(frame)

    ((submitted, timestamp),) = landmarker.submitted
    assert timestamp == 99


def test_apply_converts_bgr_to_rgb(landmarker, frame):
    """OpenCV hands over BGR and MediaPipe expects RGB; getting this wrong
    degrades detection quietly rather than failing."""
    with detector() as stage:
        stage.apply(frame)

    ((submitted, _),) = landmarker.submitted
    assert submitted.image_format == "srgb"
    assert np.array_equal(submitted.data, frame.image[..., ::-1])


def test_apply_returns_the_latest_completed_result_not_this_frames(
    landmarker, frame, mocker
):
    """The documented asynchrony: a test that assumes otherwise passes by
    luck."""
    with detector() as stage:
        first = stage.apply(frame)
        stage._latest.on_result(result_with(([], [])), None, 1)
        second = stage.apply(frame)

    assert first.hands == ()
    assert len(second.hands) == 1


def test_landmarker_config_reaches_the_options(landmarker, mocker):
    create = detection_module.HandLandmarker.create_from_options

    with detector(num_hands=2, min_tracking_confidence=0.9):
        pass

    options = create.call_args.args[0]
    assert options.num_hands == 2
    assert options.min_tracking_confidence == 0.9
    assert options.base_options.model_asset_path == "model.task"


def test_enter_ensures_the_model_is_present(landmarker):
    with detector():
        pass

    detection_module.ensure_model.assert_called_once_with("model.task")


def test_exit_closes_the_landmarker(landmarker):
    with detector():
        pass

    assert landmarker.exited is True


def test_detection_carries_the_original_frame(landmarker, frame):
    with detector() as stage:
        assert stage.apply(frame).frame is frame
