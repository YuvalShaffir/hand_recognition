from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

import numpy as np


@runtime_checkable
class Landmark(Protocol):
    """One of MediaPipe's 21 hand points. Named as a protocol so the stages
    downstream depend on the three coordinates they read rather than on the
    concrete type MediaPipe happens to hand over."""

    x: float
    y: float
    z: float


Landmarks = Sequence[Landmark]

# Joint angles of one hand, in degrees - (15,).
Angles = np.ndarray

# Angles snapped to bin centers - (15,), same units.
QuantizedAngles = np.ndarray


@dataclass(frozen=True)
class Frame:
    image: np.ndarray  # BGR, as OpenCV and MediaPipe hand it over
    timestamp_ms: int


@dataclass(frozen=True)
class Hand:
    landmarks: Landmarks  # image-space, normalized [0, 1]
    world_landmarks: Landmarks  # metric, relative to the hand itself
    timestamp_ms: int


@dataclass(frozen=True)
class Detection:
    frame: Frame
    hands: tuple[Hand, ...]

    @property
    def primary(self) -> Hand | None:
        return self.hands[0] if self.hands else None


@dataclass(frozen=True)
class Pose:
    angles: Angles
    timestamp_ms: int


@dataclass(frozen=True)
class Movement:
    """A quantized pose that differs from the one before it - the unit of
    change in the system, and the only thing gestures are made of."""

    angles: QuantizedAngles
    timestamp_ms: int


@dataclass(frozen=True)
class NormalizedPoint:
    x: float
    y: float


@dataclass(frozen=True)
class ScreenPoint:
    x: float
    y: float


# A recorded gesture is a sequence of *changes*, so a few hundred frames is
# already a long one. The ceiling exists because a recording that never stops
# - a browser tab left open, or a hostile `.npz` - otherwise grows without
# bound, and every matched frame is compared against every template.
MAX_TEMPLATE_FRAMES = 3000


@dataclass
class GestureTemplate:
    name: str
    bin_size: float
    frames: np.ndarray  # (T, 15) quantized poses, in degrees
