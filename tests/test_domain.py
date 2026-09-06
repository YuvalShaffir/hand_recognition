import dataclasses

import numpy as np
import pytest

from hand_recognition.domain import (
    Detection,
    Frame,
    GestureTemplate,
    Landmark,
    Movement,
    Pose,
)
from hand_recognition.gestures.angles import joint_angles

from .conftest import make_flat_hand, make_hand, make_landmark


@pytest.fixture
def frame():
    return Frame(image=np.zeros((2, 2, 3), dtype=np.uint8), timestamp_ms=1)


def test_detection_primary_returns_the_first_hand(frame):
    first, second = make_hand(timestamp_ms=1), make_hand(timestamp_ms=2)

    assert Detection(frame=frame, hands=(first, second)).primary is first


def test_detection_primary_is_none_with_no_hands(frame):
    assert Detection(frame=frame, hands=()).primary is None


def test_landmark_protocol_accepts_a_plain_namespace():
    point = make_landmark(0.1, 0.2, 0.3)

    assert isinstance(point, Landmark)
    checked: Landmark = point
    assert (checked.x, checked.y, checked.z) == (0.1, 0.2, 0.3)
    assert len(joint_angles(make_flat_hand())) == 15


def test_landmark_protocol_rejects_an_object_without_the_coordinates():
    assert not isinstance(object(), Landmark)


@pytest.mark.parametrize(
    "instance, attribute",
    [
        (Frame(image=np.zeros(1), timestamp_ms=0), "timestamp_ms"),
        (make_hand(), "timestamp_ms"),
        (Pose(angles=np.zeros(15), timestamp_ms=0), "timestamp_ms"),
        (Movement(angles=np.zeros(15), timestamp_ms=0), "timestamp_ms"),
    ],
)
def test_frozen_dataclasses_reject_mutation(instance, attribute):
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(instance, attribute, 99)


def test_gesture_template_is_deliberately_mutable():
    """`GestureLibrary.add` rewrites `name` to resolve collisions, so this
    one is not frozen. Documented here so it is not 'fixed' by accident."""
    template = GestureTemplate(name="wave", bin_size=15.0, frames=np.zeros((1, 15)))

    template.name = "wave-2"

    assert template.name == "wave-2"
