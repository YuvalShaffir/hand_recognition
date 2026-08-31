import time

import cv2

from .actions import build_actions, run_action
from .config import AppConfig, load_config
from .cursor_control import CursorController
from .engine import GestureEngine
from .gesture_dtw import GestureMatcher, load_templates
from .overlay import draw_hud, draw_landmarks, draw_recording_prompt_hint


class App:
    """Owns the desktop-only concerns - camera capture, display, keyboard
    handling, and `pyautogui` output (real cursor moves, real clicks) - and
    runs the camera/detection/key-handling loop. Recognition state (recorder,
    matcher, templates, landmarker) lives in a `GestureEngine`. All tunables
    come from an AppConfig, loaded from config.json (relative to cwd) if not
    given."""

    def __init__(self, config: AppConfig | None = None):
        self.config = config or load_config()
        c = self.config

        self.engine = GestureEngine(c)
        self.cursor = CursorController(
            region_margin=c.cursor.region_margin,
            smoothing=c.cursor.smoothing,
            deadzone=c.cursor.deadzone,
        )
        self.actions = build_actions(c.actions.scroll_amount)

        kb = c.keybindings
        self._key_quit = ord(kb.quit)
        self._key_toggle_cursor = ord(kb.toggle_cursor)
        self._key_toggle_record = ord(kb.toggle_record)
        self._key_tighten = ord(kb.threshold_tighten)
        self._key_loosen = ord(kb.threshold_loosen)

    def run(self) -> None:
        c = self.config
        capture = cv2.VideoCapture(index=c.camera.index)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, c.camera.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, c.camera.height)

        print(
            f"loaded {len(self.engine.templates)} recording(s): {[t.name for t in self.engine.templates]}"
        )
        print(f"known actions: {list(self.actions)}")
        print(
            f"match threshold: {self.engine.match_threshold:.2f} ([ stricter / ] looser)"
        )

        try:
            with self.engine:
                while capture.isOpened():
                    now_ms = int(time.monotonic() * 1000)
                    ok, frame = capture.read()
                    if not ok:
                        break

                    result = self.engine.process_frame(frame, now_ms)
                    draw_landmarks(frame, result.hands)

                    if result.cursor_xy is not None:
                        self.cursor.update(*result.cursor_xy)

                    if result.matched is not None:
                        fired = run_action(result.matched, self.actions)
                        print(
                            f"matched '{result.matched}'"
                            + (" -> action fired" if fired else " (no action mapped)")
                        )

                    draw_hud(
                        frame,
                        match_threshold=self.engine.match_threshold,
                        cursor_mode=self.engine.cursor_mode,
                        recording=result.recording,
                        frame_count=result.frame_count,
                    )
                    cv2.imshow("camera", frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == self._key_quit:
                        break
                    self._handle_key(key, frame, now_ms)
        finally:
            capture.release()
            cv2.destroyAllWindows()

    def _handle_key(self, key: int, frame, now_ms: int) -> None:
        if key == self._key_toggle_cursor:
            self.engine.set_cursor_mode(not self.engine.cursor_mode)
            if self.engine.cursor_mode:
                self.cursor.reset()
            print("cursor mode " + ("on" if self.engine.cursor_mode else "off"))
        elif key in (self._key_tighten, self._key_loosen):
            m = self.config.match
            step = -m.threshold_step if key == self._key_tighten else m.threshold_step
            new_threshold = round(
                min(
                    max(self.engine.match_threshold + step, m.threshold_min),
                    m.threshold_max,
                ),
                2,
            )
            self.engine.set_threshold(new_threshold)
            print(f"match threshold: {self.engine.match_threshold:.2f}")
        elif key == self._key_toggle_record:
            self._handle_record_toggle(frame, now_ms)

    def _handle_record_toggle(self, frame, now_ms: int) -> None:
        if self.engine.recorder.recording:
            draw_recording_prompt_hint(frame)
            cv2.imshow("camera", frame)
            cv2.waitKey(1)

            print("\n>>> switch to this terminal window <<<")
            name = input("action name (e.g. left-click, blank = untitled): ").strip()
            template = self.engine.stop_recording(name or None)
            path = self.engine.recorder.save(template, name or None)
            print(f"saved recording to {path}")
            self.engine.templates = load_templates(self.engine.recorder.recordings_dir)
            self.engine.matcher = GestureMatcher(
                self.engine.templates,
                threshold=self.engine.match_threshold,
                cooldown_ms=self.config.match.cooldown_ms,
            )
            print(
                f"reloaded {len(self.engine.templates)} recording(s): {[t.name for t in self.engine.templates]}"
            )
        else:
            self.engine.start_recording()
            print("recording started")
