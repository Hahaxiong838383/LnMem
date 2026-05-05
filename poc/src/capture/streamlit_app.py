"""Day 1 PoC skeleton for streamlit UI. Day 8+ will add real capture."""

from __future__ import annotations

try:
    import streamlit as st  # type: ignore
    _STREAMLIT_AVAILABLE = True
except ImportError:
    st = None  # type: ignore[assignment]
    _STREAMLIT_AVAILABLE = False

try:
    from streamlit_webrtc import webrtc_streamer  # type: ignore
    _WEBRTC_AVAILABLE = True
except ImportError:
    webrtc_streamer = None  # type: ignore[assignment]
    _WEBRTC_AVAILABLE = False

from .multimodal_capture import MultimodalCapture


def main() -> None:
    """Render Streamlit page. / 渲染 Day 1 单页采集界面。"""
    if not _STREAMLIT_AVAILABLE:
        print("streamlit is not installed. please install requirements-day1.txt")
        return

    st.set_page_config(page_title="LnMem Day 1 Capture", layout="centered")
    st.title("LnMem PoC Day 1 - Multimodal Capture")
    st.caption("Day 1 PoC skeleton，等 Day 8+ 真实采集时补全")

    if not _WEBRTC_AVAILABLE:
        st.warning("streamlit-webrtc is not installed. UI runs in stub mode.")
    else:
        webrtc_streamer(key="lnmem-day1-capture")

    if "capture_instance" not in st.session_state:
        st.session_state["capture_instance"] = MultimodalCapture()

    open_clicked = st.button("Open Capture")
    capture_clicked = st.button("Capture One Frame")
    close_clicked = st.button("Close Capture")

    capture: MultimodalCapture = st.session_state["capture_instance"]

    if open_clicked:
        capture.open()
        st.success("capture opened")

    if capture_clicked:
        try:
            frame = capture.capture_frame()
            st.json(frame)
        except RuntimeError as exc:
            st.error(str(exc))

    if close_clicked:
        capture.close()
        st.info("capture closed")


if __name__ == "__main__":
    main()
