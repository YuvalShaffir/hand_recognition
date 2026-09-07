import numpy as np

from ..domain import Angles, Hand, Landmarks, Pose
from ..stage import OptionalStage

# Each chain is wrist -> ... -> fingertip. Consecutive triples give the
# joint angles along that finger.
FINGER_CHAINS = {
    "thumb": (0, 1, 2, 3, 4),
    "index": (0, 5, 6, 7, 8),
    "middle": (0, 9, 10, 11, 12),
    "ring": (0, 13, 14, 15, 16),
    "pinky": (0, 17, 18, 19, 20),
}

# The (a, b, c) landmark triples of every joint, as three index arrays -
# precomputed so each frame is a handful of whole-array operations rather
# than one small numpy call per joint.
_TRIPLES = np.array(
    [
        (a_idx, b_idx, c_idx)
        for chain in FINGER_CHAINS.values()
        for a_idx, b_idx, c_idx in zip(chain, chain[1:], chain[2:])
    ]
)
_A, _B, _C = _TRIPLES[:, 0], _TRIPLES[:, 1], _TRIPLES[:, 2]

NUM_ANGLES = len(_TRIPLES)


def joint_angles(landmarks: Landmarks) -> Angles:
    """Joint angles (degrees) for one hand's 21 landmarks.

    3 angles per finger (5 fingers = 15 values), invariant to the hand's
    position, scale, and rotation relative to the camera.
    """
    points = np.array([(lm.x, lm.y, lm.z) for lm in landmarks], dtype=np.float64)
    v1 = points[_A] - points[_B]
    v2 = points[_C] - points[_B]
    cos_angles = np.einsum("ij,ij->i", v1, v2) / (
        np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
    )
    return np.degrees(np.arccos(np.clip(cos_angles, -1.0, 1.0)))


class AngleExtractor(OptionalStage[Hand, Pose]):
    """Hands in, poses out. Reads the world landmarks, not the image-space
    ones, so a pose describes the hand's shape and not where the hand
    happens to be in front of the camera."""

    def transform(self, item: Hand) -> Pose:
        return Pose(
            angles=joint_angles(item.world_landmarks),
            timestamp_ms=item.timestamp_ms,
        )
