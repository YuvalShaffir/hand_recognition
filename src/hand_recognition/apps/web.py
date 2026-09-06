"""Streamlit demo: `streamlit run src/hand_recognition/apps/web.py`.

Report-only in-browser gesture recognition - no real OS cursor/click control
(see docs/WEB_DEMO_PLAN.md). Must not import `actions.py` or
`cursor.driver`: both pull in `pyautogui` at module import time, which can
fail outright on a headless container this gets deployed to.
"""

import threading
import time
from dataclasses import dataclass

import av
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

from hand_recognition.config import load_config
from hand_recognition.cursor import CursorPipeline
from hand_recognition.vision import HandDetector
from hand_recognition.domain import Frame, ScreenPoint
from hand_recognition.gestures import GestureLibrary, GesturePipeline
from hand_recognition.apps.overlay import draw_hud, draw_landmarks
from hand_recognition.stage import Fork

RTC_CONFIGURATION = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}

# Only used to scale the reported cursor position; nothing is moved.
REPORTED_SCREEN_SIZE = (1920, 1080)


@dataclass
class Readout:
    gesture: str | None
    point: ScreenPoint | None


class GestureVideoProcessor(VideoProcessorBase):
    """One set of pipelines per browser session/tab - `streamlit-webrtc`
    constructs a fresh instance per session and calls `recv()` on its own
    worker thread, separate from Streamlit's script-rerun thread."""

    def __init__(self) -> None:
        config = load_config()
        self.config = config
        self.gestures = GesturePipeline(
            GestureLibrary(config.paths.recordings_dir), config.quantize, config.match
        )
        self.cursor = CursorPipeline(config.cursor, REPORTED_SCREEN_SIZE)
        self._recognize = Fork(self.gestures, self.cursor)
        self._detector = HandDetector(config.paths.model_path, config.landmarker)
        self._detector.__enter__()
        self._lock = threading.Lock()
        self._latest: Readout | None = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        detection = self._detector.apply(
            Frame(image=image, timestamp_ms=int(time.monotonic() * 1000))
        )
        gesture, point = self._recognize.apply(detection.primary)

        draw_landmarks(image, detection.hands)
        draw_hud(
            image,
            match_threshold=self.gestures.threshold,
            cursor_mode=self.cursor.enabled,
            recording=self.gestures.recording,
            frame_count=self.gestures.recorded_frame_count,
        )

        with self._lock:
            self._latest = Readout(gesture=gesture, point=point)

        return av.VideoFrame.from_ndarray(image, format="bgr24")

    def latest_readout(self) -> Readout | None:
        with self._lock:
            return self._latest

    def __del__(self) -> None:
        self._detector.__exit__(None, None, None)


def _render_controls(processor: GestureVideoProcessor) -> None:
    if "record_phase" not in st.session_state:
        st.session_state.record_phase = "idle"  # idle -> recording -> naming -> idle

    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.record_phase == "idle":
            if st.button("Start recording"):
                processor.gestures.start_recording()
                st.session_state.record_phase = "recording"
                st.rerun()
        elif st.session_state.record_phase == "recording":
            if st.button("Stop recording"):
                st.session_state.record_phase = "naming"
                st.rerun()
        else:
            name = st.text_input(
                "Gesture name", key="record_name", placeholder="e.g. wave"
            )
            if st.button("Save recording"):
                stored = processor.gestures.stop_recording(name)
                st.session_state.record_phase = "idle"
                if stored is None:
                    st.warning("Nothing recorded - the hand never moved.")
                else:
                    st.success(f"Recorded '{stored}'")
                st.rerun()

    with col2:
        processor.cursor.enabled = st.checkbox(
            "Cursor mode (report only)", value=processor.cursor.enabled
        )

    m = processor.config.match
    processor.gestures.threshold = st.slider(
        "Match threshold",
        min_value=m.threshold_min,
        max_value=m.threshold_max,
        value=processor.gestures.threshold,
        step=m.threshold_step,
    )

    st.caption(f"Loaded gestures: {processor.gestures.library.names}")


@st.fragment(run_every=0.2)
def _render_readout(processor: GestureVideoProcessor) -> None:
    readout = processor.latest_readout()
    if readout is None:
        st.info("Waiting for camera frames...")
    elif readout.gesture:
        st.success(f"recognized macro: {readout.gesture}")
    else:
        st.write("recognized macro: -")


def main() -> None:
    st.set_page_config(page_title="Hand Gesture Recognition Demo", page_icon="✋")
    st.title("Hand Gesture Recognition Demo")
    st.caption(
        "Record a hand gesture, then repeat it to see it recognized live. "
        "Everything runs in your browser via WebRTC - report-only, no OS "
        "cursor or click control."
    )

    webrtc_ctx = webrtc_streamer(
        key="hand-recognition",
        video_processor_factory=GestureVideoProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
    )

    if webrtc_ctx.video_processor:
        _render_controls(webrtc_ctx.video_processor)
        _render_readout(webrtc_ctx.video_processor)
    else:
        st.info("Click START above and grant camera access to begin.")


main()
