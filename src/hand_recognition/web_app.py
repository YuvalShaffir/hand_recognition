"""Streamlit demo: `streamlit run src/hand_recognition/web_app.py`.

Report-only in-browser gesture recognition - no real OS cursor/click control
(see docs/WEB_DEMO_PLAN.md). Must not import `cursor_control.py` or
`actions.py`: both pull in `pyautogui` at module import time, which can fail
outright on a headless container this gets deployed to.
"""

import threading
import time

import av
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

from hand_recognition.config import load_config
from hand_recognition.engine import FrameResult, GestureEngine
from hand_recognition.overlay import draw_hud, draw_landmarks

RTC_CONFIGURATION = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}


class GestureVideoProcessor(VideoProcessorBase):
    """One `GestureEngine` per browser session/tab - `streamlit-webrtc`
    constructs a fresh instance per session and calls `recv()` on its own
    worker thread, separate from Streamlit's script-rerun thread."""

    def __init__(self) -> None:
        self.engine = GestureEngine(load_config())
        self.engine.__enter__()
        self._lock = threading.Lock()
        self._latest: FrameResult | None = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        result = self.engine.process_frame(img, int(time.monotonic() * 1000))

        draw_landmarks(img, result.hands)
        draw_hud(
            img,
            match_threshold=self.engine.match_threshold,
            cursor_mode=self.engine.cursor_mode,
            recording=result.recording,
            frame_count=result.frame_count,
        )

        with self._lock:
            self._latest = result

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def latest_result(self) -> FrameResult | None:
        with self._lock:
            return self._latest

    def __del__(self) -> None:
        self.engine.__exit__(None, None, None)


def _render_controls(processor: GestureVideoProcessor) -> None:
    if "record_phase" not in st.session_state:
        st.session_state.record_phase = "idle"  # idle -> recording -> naming -> idle

    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.record_phase == "idle":
            if st.button("Start recording"):
                processor.engine.start_recording()
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
                template = processor.engine.stop_recording(name)
                st.session_state.record_phase = "idle"
                st.success(
                    f"Recorded '{template.name}' ({len(template.frames)} frames)"
                )
                st.rerun()

    with col2:
        cursor_mode = st.checkbox(
            "Cursor mode (report only)", value=processor.engine.cursor_mode
        )
        processor.engine.set_cursor_mode(cursor_mode)

    m = processor.engine.config.match
    threshold = st.slider(
        "Match threshold",
        min_value=m.threshold_min,
        max_value=m.threshold_max,
        value=processor.engine.match_threshold,
        step=m.threshold_step,
    )
    processor.engine.set_threshold(threshold)

    st.caption(f"Loaded gestures: {[t.name for t in processor.engine.templates]}")


@st.fragment(run_every=0.2)
def _render_readout(processor: GestureVideoProcessor) -> None:
    result = processor.latest_result()
    if result is None:
        st.info("Waiting for camera frames...")
    elif result.matched:
        st.success(f"recognized macro: {result.matched}")
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
