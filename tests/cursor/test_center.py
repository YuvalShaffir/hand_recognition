import numpy as np
import pytest

from hand_recognition.cursor.center import HandCenterExtractor

from ..conftest import make_flat_hand, make_folded_hand, make_hand, make_landmark


@pytest.fixture
def extractor():
    return HandCenterExtractor()


def test_centre_is_the_mean_of_all_landmarks(extractor):
    landmarks = [make_landmark(0.0, 0.0), make_landmark(1.0, 0.5)]

    centre = extractor.apply(make_hand(landmarks=landmarks))

    assert (centre.x, centre.y) == (0.5, 0.25)


def test_centre_is_stable_while_fingers_move(extractor):
    """What lets cursor mode and gesture matching share one hand: folding
    the fingers barely moves the centre, while it moves a fingertip a long
    way."""
    flat, folded = make_flat_hand(), make_folded_hand(1.0)
    first = extractor.apply(make_hand(landmarks=flat))
    second = extractor.apply(make_hand(landmarks=folded))

    centre_shift = np.hypot(second.x - first.x, second.y - first.y)
    fingertip_shift = np.hypot(folded[8].x - flat[8].x, folded[8].y - flat[8].y)

    assert centre_shift < fingertip_shift / 5


def test_uses_image_landmarks_not_world_landmarks(extractor):
    hand = make_hand(
        landmarks=[make_landmark(0.25, 0.25)],
        world_landmarks=[make_landmark(9.0, 9.0)],
    )

    assert (extractor.apply(hand).x, extractor.apply(hand).y) == (0.25, 0.25)


def test_passes_none_through(extractor):
    assert extractor.apply(None) is None


def test_empty_landmarks_divides_by_zero(extractor):
    with pytest.raises(ZeroDivisionError):
        extractor.apply(make_hand(landmarks=[]))


def test_single_landmark_is_its_own_centre(extractor):
    centre = extractor.apply(make_hand(landmarks=[make_landmark(0.3, 0.7)]))

    assert (centre.x, centre.y) == (0.3, 0.7)
