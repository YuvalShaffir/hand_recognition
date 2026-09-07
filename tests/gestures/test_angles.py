import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hand_recognition.gestures.angles import (
    FINGER_CHAINS,
    NUM_ANGLES,
    AngleExtractor,
    joint_angles,
)

from ..conftest import landmark_arrays, make_fist_hand, make_flat_hand, make_hand
from ..conftest import make_landmark

# arccos is ill-conditioned near 0 and 180 degrees - the angles a real hand
# spends most of its time at - so an invariance holds to a fraction of a
# degree, not to machine precision.
INVARIANCE_TOLERANCE_DEG = 0.1

ROTATION_ANGLE = st.floats(
    min_value=-np.pi, max_value=np.pi, allow_nan=False, allow_infinity=False
)


def landmarks_of(points: np.ndarray):
    return [make_landmark(*point) for point in points]


def rotation_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    def about(axis: int, angle: float) -> np.ndarray:
        cos, sin = np.cos(angle), np.sin(angle)
        matrix = np.eye(3)
        other = [i for i in range(3) if i != axis]
        matrix[np.ix_(other, other)] = [[cos, -sin], [sin, cos]]
        return matrix

    return about(0, yaw) @ about(1, pitch) @ about(2, roll)


def test_flat_hand_gives_straight_angles():
    assert np.allclose(joint_angles(make_flat_hand()), 180.0)


def test_fist_hand_gives_folded_angles():
    assert np.allclose(joint_angles(make_fist_hand()), 0.0, atol=1e-4)


def test_returns_fifteen_angles():
    assert NUM_ANGLES == 15
    assert joint_angles(make_flat_hand()).shape == (15,)


def test_triples_match_the_finger_chains():
    """`_TRIPLES` is a precomputed optimisation; this is the definition it
    replaced, so the two cannot drift apart."""
    from hand_recognition.gestures.angles import _TRIPLES

    expected = [
        (a, b, c)
        for chain in FINGER_CHAINS.values()
        for a, b, c in zip(chain, chain[1:], chain[2:])
    ]

    assert [tuple(triple) for triple in _TRIPLES] == expected


def test_extractor_uses_world_landmarks_not_image_landmarks():
    hand = make_hand(landmarks=make_fist_hand(), world_landmarks=make_flat_hand())

    assert np.allclose(AngleExtractor().apply(hand).angles, 180.0)


def test_extractor_preserves_the_timestamp():
    assert AngleExtractor().apply(make_hand(timestamp_ms=1234)).timestamp_ms == 1234


def test_extractor_passes_none_through():
    assert AngleExtractor().apply(None) is None


@settings(max_examples=50)
@given(points=landmark_arrays())
def test_angles_are_always_within_zero_and_one_eighty(points):
    angles = joint_angles(landmarks_of(points))

    assert not np.isnan(angles).any()
    assert ((angles >= 0.0) & (angles <= 180.0)).all()


@settings(max_examples=50)
@given(
    points=landmark_arrays(),
    offset=st.tuples(*[st.floats(-5.0, 5.0)] * 3),
)
def test_angles_are_invariant_under_translation(points, offset):
    moved = points + np.array(offset)

    assert np.allclose(
        joint_angles(landmarks_of(points)),
        joint_angles(landmarks_of(moved)),
        atol=INVARIANCE_TOLERANCE_DEG,
    )


@settings(max_examples=50)
@given(
    points=landmark_arrays(),
    yaw=ROTATION_ANGLE,
    pitch=ROTATION_ANGLE,
    roll=ROTATION_ANGLE,
)
def test_angles_are_invariant_under_rotation(points, yaw, pitch, roll):
    rotated = points @ rotation_matrix(yaw, pitch, roll).T

    assert np.allclose(
        joint_angles(landmarks_of(points)),
        joint_angles(landmarks_of(rotated)),
        atol=INVARIANCE_TOLERANCE_DEG,
    )


@settings(max_examples=50)
@given(points=landmark_arrays(), scale=st.floats(0.1, 10.0))
def test_angles_are_invariant_under_uniform_scale(points, scale):
    assert np.allclose(
        joint_angles(landmarks_of(points)),
        joint_angles(landmarks_of(points * scale)),
        atol=INVARIANCE_TOLERANCE_DEG,
    )


def test_collinear_points_do_not_produce_nan():
    """`cos` of exactly +-1 lands outside [-1, 1] under float error; the
    `np.clip` guard is what keeps `arccos` from returning nan."""
    angles = joint_angles(make_flat_hand())

    assert not np.isnan(angles).any()


def test_coincident_points_produce_nan_not_a_crash():
    """MediaPipe should never emit duplicate landmarks, but a malformed
    `.npz` or a future local model could - so this documents the current
    behaviour rather than blessing it."""
    points = np.zeros((21, 3))
    points[5] = points[6] = (1.0, 0.0, 0.0)

    with np.errstate(invalid="ignore"):
        angles = joint_angles(landmarks_of(points))

    assert np.isnan(angles).any()


def test_too_few_landmarks_raises():
    with pytest.raises(IndexError):
        joint_angles(landmarks_of(np.zeros((20, 3))))


def test_empty_landmarks_raises():
    with pytest.raises(IndexError):
        joint_angles([])
