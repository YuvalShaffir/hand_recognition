from collections.abc import Sequence

import cv2
import numpy as np

from ..domain import Hand

HAND_CONNECTIONS = ((0, 0), (1, 4), (5, 8), (9, 12), (13, 16), (17, 20))


def draw_landmarks(image: np.ndarray, hands: Sequence[Hand]) -> None:
    height, width = image.shape[:2]
    for hand in hands:
        points = [(int(lm.x * width), int(lm.y * height)) for lm in hand.landmarks]
        for start_idx, end_idx in HAND_CONNECTIONS:
            cv2.line(
                image,
                points[start_idx],
                points[end_idx],
                color=(0, 255, 0),
                thickness=2,
            )
        for x, y in points:
            cv2.circle(image, center=(x, y), radius=4, color=(0, 0, 255), thickness=-1)


def draw_hud(
    image: np.ndarray,
    *,
    match_threshold: float,
    cursor_mode: bool,
    recording: bool,
    frame_count: int,
) -> None:
    cv2.putText(
        image,
        f"thresh: {match_threshold:.2f}",
        (10, image.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        1,
    )

    overlay_y = 30
    if cursor_mode:
        cv2.putText(
            image,
            "CURSOR MODE",
            (10, overlay_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
        )
        overlay_y += 30
    if recording:
        cv2.putText(
            image,
            f"REC ({frame_count} frames)",
            (10, overlay_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )


def draw_recording_prompt_hint(image: np.ndarray) -> None:
    cv2.putText(
        image,
        "check terminal for name prompt",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )
