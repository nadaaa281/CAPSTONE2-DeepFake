import streamlit as st
import tempfile
from pathlib import Path
from src.app_predict import predict_video, predict_image, predict_audio
from src.llm_explainer import generate_explanation
from PIL import Image

st.set_page_config(page_title="Detection", page_icon="🔍", layout="wide")

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(BASE_DIR, "assets/cyber.css")) as f:

mode = st.selectbox("Select media type", ["Video", "Image", "Audio"])

st.markdown("""
<div class="df-card-plain">
  <h2 style="margin:0;">🔍 Detection</h2>
  <p style="color: rgba(230,230,230,.78); margin-top:6px;">
    Upload a file and run deepfake detection. The system will analyze and generate an AI explanation.
  </p>
</div>
""", unsafe_allow_html=True)

if "result" not in st.session_state:
    st.session_state["result"] = None

# =========================
# VIDEO
# =========================
if mode == "Video":
    uploaded = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi", "mkv"])

    if uploaded:
        st.video(uploaded.getvalue())

    run = st.button("🚀 Run Detection", disabled=(uploaded is None))

    if run and uploaded:
        with st.spinner("Processing video..."):
            with tempfile.TemporaryDirectory() as td:
                vid_path = Path(td) / uploaded.name
                vid_path.write_bytes(uploaded.getvalue())

                result = predict_video(str(vid_path))

                try:
                    explanation = generate_explanation(
                        result.get("transcript", ""),
                        result.get("prob_real", 0) * 100,
                        result.get("prob_fake", 0) * 100
                    )
                    result["llm_explanation"] = explanation

                except Exception as e:
                    error_msg = str(e)
                    result["llm_explanation"] = f"AI explanation unavailable: {error_msg}"
                    print("LLM ERROR (Video):", error_msg)

        st.session_state["result"] = result
        st.switch_page("pages/2_Results.py")

# =========================
# IMAGE
# =========================
if mode == "Image":
    img_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if img_file:
        pil_img = Image.open(img_file)
        st.image(pil_img, use_container_width=True)

    run_img = st.button("🖼️ Run Detection", disabled=(img_file is None))

    if run_img and img_file:
        with st.spinner("Analyzing image..."):
            result = predict_image(pil_img)

            try:
                explanation = generate_explanation(
                    "Image input (no transcript available)",
                    result.get("prob_real", 0) * 100,
                    result.get("prob_fake", 0) * 100
                )
                result["llm_explanation"] = explanation

            except Exception as e:
                error_msg = str(e)
                result["llm_explanation"] = f"AI explanation unavailable: {error_msg}"
                print("LLM ERROR (Image):", error_msg)

        st.session_state["result"] = result
        st.switch_page("pages/2_Results.py")

# =========================
# AUDIO
# =========================
if mode == "Audio":
    audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"])

    if audio_file:
        st.audio(audio_file.getvalue())

    run_audio = st.button("🎧 Run Detection", disabled=(audio_file is None))

    if run_audio and audio_file:
        with st.spinner("Processing audio..."):
            with tempfile.TemporaryDirectory() as td:
                aud_path = Path(td) / audio_file.name
                aud_path.write_bytes(audio_file.getvalue())

                result = predict_audio(str(aud_path))

                try:
                    explanation = generate_explanation(
                        result.get("transcript", ""),
                        result.get("prob_real", 0) * 100,
                        result.get("prob_fake", 0) * 100
                    )
                    result["llm_explanation"] = explanation

                except Exception as e:
                    error_msg = str(e)
                    result["llm_explanation"] = f"AI explanation unavailable: {error_msg}"
                    print("LLM ERROR (Audio):", error_msg)

        st.session_state["result"] = result
        st.switch_page("pages/2_Results.py")