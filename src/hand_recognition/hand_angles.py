import numpy as np

# Each chain is wrist -> ... -> fingertip. Consecutive triples give the
# joint angles along that finger.
FINGER_CHAINS = {
    "thumb": (0, 1, 2, 3, 4),
    "index": (0, 5, 6, 7, 8),
    "middle": (0, 9, 10, 11, 12),
    "ring": (0, 13, 14, 15, 16),
    "pinky": (0, 17, 18, 19, 20),
}

NUM_ANGLES = sum(len(chain) - 2 for chain in FINGER_CHAINS.values())


def _angle(a, b, c) -> float:
    """Angle at point b between segments a-b and b-c, in degrees."""
    v1 = np.array([a.x - b.x, a.y - b.y, a.z - b.z])
    v2 = np.array([c.x - b.x, c.y - b.y, c.z - b.z])
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


def hand_joint_angles(landmarks) -> np.ndarray:
    """Joint angles (degrees) for one hand's 21 landmarks.

    3 angles per finger (5 fingers = 15 values), invariant to the hand's
    position, scale, and rotation relative to the camera.
    """
    angles = [
        _angle(landmarks[a_idx], landmarks[b_idx], landmarks[c_idx])
        for chain in FINGER_CHAINS.values()
        for a_idx, b_idx, c_idx in zip(chain, chain[1:], chain[2:])
    ]
    return np.array(angles, dtype=np.float64)
