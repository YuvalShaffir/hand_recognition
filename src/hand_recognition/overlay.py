import cv2

HAND_CONNECTIONS = ((0, 0), (1, 4), (5, 8), (9, 12), (13, 16), (17, 20))


def draw_landmarks(frame, landmarks) -> None:
    height, width = frame.shape[:2]
    for hand in landmarks:
        points = [(int(lm.x * width), int(lm.y * height)) for lm in hand]
        for start_idx, end_idx in HAND_CONNECTIONS:
            cv2.line(
                frame,
                points[start_idx],
                points[end_idx],
                color=(0, 255, 0),
                thickness=2,
            )
        for x, y in points:
            cv2.circle(frame, center=(x, y), radius=4, color=(0, 0, 255), thickness=-1)


def draw_hud(frame, *, match_threshold: float, cursor_mode: bool, recording: bool, frame_count: int) -> None:
    cv2.putText(
        frame,
        f"thresh: {match_threshold:.2f}",
        (10, frame.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        1,
    )

    overlay_y = 30
    if cursor_mode:
        cv2.putText(
            frame,
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
            frame,
            f"REC ({frame_count} frames)",
            (10, overlay_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )


def draw_recording_prompt_hint(frame) -> None:
    cv2.putText(
        frame,
        "check terminal for name prompt",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )
