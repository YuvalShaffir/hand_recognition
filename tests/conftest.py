"""Fixtures and fake boundaries for the whole suite.

The four edges the suite is allowed to fake - `pyautogui`, `cv2`,
`mediapipe`, `urllib` - are faked here or in the test that owns them, and
nowhere else. `pyautogui` in particular cannot even be imported on a
headless machine, so a stand-in is installed before any test module pulls in
`actions.py` or `cursor/driver.py`.
"""

import sys
import types
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from hypothesis import strategies as st

from hand_recognition.domain import GestureTemplate, Hand, Landmark, Movement, Pose
from hand_recognition.gestures.angles import FINGER_CHAINS

NUM_LANDMARKS = 21


def _install_fake_pyautogui() -> types.ModuleType:
    module = types.ModuleType("pyautogui")

    class FailSafeException(Exception):
        pass

    module.FailSafeException = FailSafeException  # type: ignore[attr-defined]
    module.FAILSAFE = False  # type: ignore[attr-defined]
    module.click = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    module.doubleClick = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    module.scroll = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    module.moveTo = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    module.size = lambda: (1920, 1080)  # type: ignore[attr-defined]
    sys.modules["pyautogui"] = module
    return module


FAKE_PYAUTOGUI = _install_fake_pyautogui()


def make_landmark(x: float, y: float, z: float = 0.0) -> Landmark:
    return SimpleNamespace(x=x, y=y, z=z)


@pytest.fixture
def landmark():
    return make_landmark


# Every finger runs along its own ray from the wrist, so a joint's angle is
# fixed by where along the ray its three landmarks sit.
_DIRECTIONS = np.array(
    [
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.6, 0.8, 0.0),
        (0.0, 0.6, 0.8),
    ]
)


def _hand_points(offsets: tuple[float, ...]) -> np.ndarray:
    points = np.zeros((NUM_LANDMARKS, 3))
    for direction, chain in zip(_DIRECTIONS, FINGER_CHAINS.values()):
        for offset, index in zip(offsets, chain[1:]):
            points[index] = direction * offset
    return points


def _landmarks(points: np.ndarray) -> list[Landmark]:
    return [make_landmark(*point) for point in points]


def make_flat_hand() -> list[Landmark]:
    """Fingers fully extended: every joint's neighbours sit on opposite
    sides of it, so all 15 angles are 180 degrees by construction."""
    return _landmarks(_hand_points((1.0, 2.0, 3.0, 4.0)))


def make_fist_hand() -> list[Landmark]:
    """Fingers folded back on themselves: every joint's neighbours sit on
    the same side of it, so all 15 angles are 0 degrees."""
    return _landmarks(_hand_points((3.0, 0.5, 3.5, 1.0)))


_FLAT_OFFSETS = np.array((1.0, 2.0, 3.0, 4.0))
_FIST_OFFSETS = np.array((3.0, 0.5, 3.5, 1.0))


def make_folded_hand(fold: float) -> list[Landmark]:
    """A hand somewhere between flat (`fold=0`) and fisted (`fold=1`), so a
    test can perform a gesture as a sequence of folds."""
    offsets = _FLAT_OFFSETS + fold * (_FIST_OFFSETS - _FLAT_OFFSETS)
    return _landmarks(_hand_points(tuple(offsets)))


@pytest.fixture
def folded_hand():
    return make_folded_hand


@pytest.fixture
def flat_hand():
    return make_flat_hand


@pytest.fixture
def fist_hand():
    return make_fist_hand


def make_hand(
    landmarks: Any = None,
    world_landmarks: Any = None,
    timestamp_ms: int = 0,
) -> Hand:
    if landmarks is None:
        landmarks = make_flat_hand()
    if world_landmarks is None:
        world_landmarks = landmarks
    return Hand(
        landmarks=landmarks,
        world_landmarks=world_landmarks,
        timestamp_ms=timestamp_ms,
    )


@pytest.fixture
def hand():
    return make_hand


def make_pose(angles: Any, timestamp_ms: int = 0) -> Pose:
    return Pose(angles=np.asarray(angles, dtype=np.float64), timestamp_ms=timestamp_ms)


@pytest.fixture
def pose():
    return make_pose


def make_movement(angles: Any, timestamp_ms: int = 0) -> Movement:
    return Movement(
        angles=np.asarray(angles, dtype=np.float64), timestamp_ms=timestamp_ms
    )


@pytest.fixture
def movement():
    return make_movement


def make_template(
    name: str = "wave",
    frames: Any = ((0.0,) * 15,),
    bin_size: float = 15.0,
) -> GestureTemplate:
    return GestureTemplate(
        name=name,
        bin_size=bin_size,
        frames=np.asarray(frames, dtype=np.float64),
    )


@pytest.fixture
def template():
    return make_template


def _non_degenerate(points: np.ndarray) -> bool:
    """Rejects hands with coincident landmarks: a zero-length bone has no
    angle, and that edge case has a test of its own."""
    triples = [
        (a, b, c)
        for chain in FINGER_CHAINS.values()
        for a, b, c in zip(chain, chain[1:], chain[2:])
    ]
    return all(
        np.linalg.norm(points[a] - points[b]) > 1e-3
        and np.linalg.norm(points[c] - points[b]) > 1e-3
        for a, b, c in triples
    )


def landmark_arrays() -> st.SearchStrategy[np.ndarray]:
    """21 finite 3-vectors in a plausible landmark coordinate range."""
    coordinate = st.floats(
        min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False, width=32
    )
    return (
        st.lists(
            st.tuples(coordinate, coordinate, coordinate),
            min_size=NUM_LANDMARKS,
            max_size=NUM_LANDMARKS,
        )
        .map(lambda rows: np.array(rows, dtype=np.float64))
        .filter(_non_degenerate)
    )
