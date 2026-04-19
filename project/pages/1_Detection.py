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
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
 
if "result" not in st.session_state:
    st.session_state["result"] = None
 
# ── Top bar ───────────────────────────────────────────────────────────────────
st.markdown("###### CYBERSECURITY &nbsp;/&nbsp; DETECTION")
 
col_title, col_status = st.columns([4, 1], vertical_alignment="top")
with col_title:
    st.title("🛡️ Detection")
    st.caption("Upload a file and run deepfake detection. The system will analyze and generate an AI explanation.")
with col_status:
    st.success("⬤  SYSTEM ONLINE")
    st.caption("FFmpeg • Whisper • v2.4.1")
 
st.markdown("---")
 
# ── Mode selector ─────────────────────────────────────────────────────────────
mode = st.radio(
    "Media type",
    ["Video", "Image", "Audio"],
    horizontal=True,
    label_visibility="collapsed",
)
 
st.markdown("")
 
# ── Step tracker ──────────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex;align-items:center;gap:0;margin-bottom:20px;">
  <div style="display:flex;align-items:center;gap:8px;padding:8px 14px;border-radius:8px;background:#0e1e14;border:0.5px solid #1a4d2a;">
    <div style="width:20px;height:20px;border-radius:50%;background:#1a4d2a;color:#22c55e;font-size:10px;font-weight:500;display:flex;align-items:center;justify-content:center;">✓</div>
    <span style="font-size:12px;color:#4ade80;">Select mode</span>
  </div>
  <span style="color:#2a2d35;padding:0 6px;font-size:14px;">›</span>
  <div style="display:flex;align-items:center;gap:8px;padding:8px 14px;border-radius:8px;background:#1a1d24;border:0.5px solid #3d4149;">
    <div style="width:20px;height:20px;border-radius:50%;background:#1e2533;color:#60a5fa;font-size:10px;font-weight:500;display:flex;align-items:center;justify-content:center;">2</div>
    <span style="font-size:12px;color:#e5e7eb;">Upload file</span>
  </div>
  <span style="color:#2a2d35;padding:0 6px;font-size:14px;">›</span>
  <div style="display:flex;align-items:center;gap:8px;padding:8px 14px;border-radius:8px;border:0.5px solid #1f2229;">
    <div style="width:20px;height:20px;border-radius:50%;background:#1f2229;color:#4b5563;font-size:10px;font-weight:500;display:flex;align-items:center;justify-content:center;">3</div>
    <span style="font-size:12px;color:#4b5563;">Run detection</span>
  </div>
  <span style="color:#2a2d35;padding:0 6px;font-size:14px;">›</span>
  <div style="display:flex;align-items:center;gap:8px;padding:8px 14px;border-radius:8px;border:0.5px solid #1f2229;">
    <div style="width:20px;height:20px;border-radius:50%;background:#1f2229;color:#4b5563;font-size:10px;font-weight:500;display:flex;align-items:center;justify-content:center;">4</div>
    <span style="font-size:12px;color:#4b5563;">View results</span>
  </div>
</div>
""", unsafe_allow_html=True)
 
# ── Upload + settings layout ──────────────────────────────────────────────────
upload_col, settings_col = st.columns([3, 1], vertical_alignment="top")
 
with settings_col:
    with st.container(border=True):
        st.caption("ANALYSIS SETTINGS")
        st.toggle("Audio forensics", value=True)
        st.toggle("Lip sync check", value=True)
        st.toggle("Face geometry", value=True)
        st.toggle("Transcript (Whisper)", value=True)
        st.caption("Clip length: 5–15 sec recommended")
        st.caption("Model: v2.4.1")
 
# =========================
# VIDEO
# =========================
if mode == "Video":
    with upload_col:
        uploaded = st.file_uploader(
            "Drop your video here or click to browse",
            type=["mp4", "mov", "avi", "mkv"],
            help="Max 200MB · MP4, MOV, AVI, MKV, MPEG4",
        )
        if uploaded:
            st.video(uploaded.getvalue())
 
    st.markdown("")
    run_col, info_col = st.columns([2, 3], vertical_alignment="center")
    with run_col:
        run = st.button("▶  Run Detection", disabled=(uploaded is None), use_container_width=True)
    with info_col:
        st.caption("ℹ️  Max **200MB** per file · results in ~30 sec")
 
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
                        result.get("prob_fake", 0) * 100,
                    )
                    result["llm_explanation"] = explanation
                except Exception as e:
                    result["llm_explanation"] = f"AI explanation unavailable: {e}"
                    print("LLM ERROR (Video):", e)
        st.session_state["result"] = result
        st.switch_page("pages/2_Results.py")
 
# =========================
# IMAGE  ← FIXED: now uses generate_image_explanation via predict_image()
# =========================
elif mode == "Image":
    with upload_col:
        img_file = st.file_uploader(
            "Drop your image here or click to browse",
            type=["jpg", "jpeg", "png"],
            help="JPG, JPEG, PNG",
        )
        if img_file:
            pil_img = Image.open(img_file)
            st.image(pil_img, use_container_width=True)
 
    st.markdown("")
    run_col, info_col = st.columns([2, 3], vertical_alignment="center")
    with run_col:
        run_img = st.button("▶  Run Detection", disabled=(img_file is None), use_container_width=True)
    with info_col:
        st.caption("ℹ️  JPG / PNG · results in ~10 sec")
 
    if run_img and img_file:
        with st.spinner("Analyzing image..."):
            result = predict_image(pil_img)
            # predict_image() already calls generate_image_explanation() internally
            # and stores the result in result["explanation"] — just pass it through
            result["llm_explanation"] = result.get("explanation", "AI explanation unavailable.")
        st.session_state["result"] = result
        st.switch_page("pages/2_Results.py")
 
# =========================
# AUDIO
# =========================
elif mode == "Audio":
    with upload_col:
        audio_file = st.file_uploader(
            "Drop your audio here or click to browse",
            type=["wav", "mp3", "m4a"],
            help="WAV, MP3, M4A",
        )
        if audio_file:
            st.audio(audio_file.getvalue())
 
    st.markdown("")
    run_col, info_col = st.columns([2, 3], vertical_alignment="center")
    with run_col:
        run_audio = st.button("▶  Run Detection", disabled=(audio_file is None), use_container_width=True)
    with info_col:
        st.caption("ℹ️  WAV, MP3, M4A · results in ~20 sec")
 
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
                        result.get("prob_fake", 0) * 100,
                    )
                    result["llm_explanation"] = explanation
                except Exception as e:
                    result["llm_explanation"] = f"AI explanation unavailable: {e}"
                    print("LLM ERROR (Audio):", e)
        st.session_state["result"] = result
        st.switch_page("pages/2_Results.py")
