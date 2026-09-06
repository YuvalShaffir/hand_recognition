import cv2

from ..actions import ActionDispatcher
from ..config import AppConfig, KeybindConfig, load_config
from ..cursor import CursorPipeline
from ..cursor.driver import CursorDriver
from ..domain import Detection
from ..gestures import GestureLibrary, GesturePipeline
from ..stage import Fork
from ..vision import CaptureManager, HandDetector
from .overlay import draw_hud, draw_landmarks, draw_recording_prompt_hint


class _Keys:
    def __init__(self, config: KeybindConfig) -> None:
        self.quit = ord(config.quit)
        self.toggle_cursor = ord(config.toggle_cursor)
        self.toggle_record = ord(config.toggle_record)
        self.tighten = ord(config.threshold_tighten)
        self.loosen = ord(config.threshold_loosen)


class DesktopApp:
    """The desktop program: the window, the keyboard, and the OS output.

    It wires the camera to the two pipelines and does nothing with the
    frames but draw them - all recognition state lives behind
    `GesturePipeline` and `CursorPipeline`."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or load_config()
        c = self.config

        self.gestures = GesturePipeline(
            GestureLibrary(c.paths.recordings_dir), c.quantize, c.match
        )
        self.driver = CursorDriver()
        self.cursor = CursorPipeline(c.cursor, self.driver.screen_size)
        self.actions = ActionDispatcher(c.actions)
        self.recognize = Fork(self.gestures, self.cursor)
        self._keys = _Keys(c.keybindings)

    def run(self) -> None:
        self._announce()
        camera = CaptureManager(self.config.camera)
        detector = HandDetector(self.config.paths.model_path, self.config.landmarker)
        try:
            with camera, detector:
                for detection in detector(iter(camera)):
                    gesture, point = self.recognize.apply(detection.primary)

                    if point is not None:
                        self.driver.move_to(point)
                    if gesture is not None:
                        self._fire(gesture)

                    self._render(detection)
                    if not self._handle_key(cv2.waitKey(1) & 0xFF, detection):
                        break
        finally:
            cv2.destroyAllWindows()

    def _announce(self) -> None:
        names = self.gestures.library.names
        print(f"loaded {len(names)} gesture(s): {names}")
        print(f"known actions: {self.actions.names}")
        print(f"match threshold: {self.gestures.threshold:.2f} ([ stricter / ] looser)")

    def _fire(self, gesture: str) -> None:
        fired = self.actions.dispatch(gesture)
        print(
            f"matched '{gesture}'"
            + (" -> action fired" if fired else " (no action mapped)")
        )

    def _render(self, detection: Detection) -> None:
        image = detection.frame.image
        draw_landmarks(image, detection.hands)
        draw_hud(
            image,
            match_threshold=self.gestures.threshold,
            cursor_mode=self.cursor.enabled,
            recording=self.gestures.recording,
            frame_count=self.gestures.recorded_frame_count,
        )
        cv2.imshow("camera", image)

    def _handle_key(self, key: int, detection: Detection) -> bool:
        if key == self._keys.quit:
            return False
        if key == self._keys.toggle_cursor:
            self.cursor.enabled = not self.cursor.enabled
            print("cursor mode " + ("on" if self.cursor.enabled else "off"))
        elif key in (self._keys.tighten, self._keys.loosen):
            self._step_threshold(-1 if key == self._keys.tighten else 1)
        elif key == self._keys.toggle_record:
            self._toggle_recording(detection)
        return True

    def _step_threshold(self, direction: int) -> None:
        m = self.config.match
        moved = self.gestures.threshold + direction * m.threshold_step
        self.gestures.threshold = round(
            min(max(moved, m.threshold_min), m.threshold_max), 2
        )
        print(f"match threshold: {self.gestures.threshold:.2f}")

    def _toggle_recording(self, detection: Detection) -> None:
        if not self.gestures.recording:
            self.gestures.start_recording()
            print("recording started")
            return

        draw_recording_prompt_hint(detection.frame.image)
        cv2.imshow("camera", detection.frame.image)
        cv2.waitKey(1)

        print("\n>>> switch to this terminal window <<<")
        name = input("action name (e.g. left-click, blank = untitled): ").strip()
        stored = self.gestures.stop_recording(name)
        if stored is None:
            print("nothing recorded - the hand never moved")
            return
        print(f"saved gesture '{stored}'")
        print(f"known gestures: {self.gestures.library.names}")
