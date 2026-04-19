import io
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import streamlit as st
st.set_page_config(page_title="Results", page_icon="📊", layout="wide")
 
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(BASE_DIR, "assets/cyber.css")) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
 
# =========================
# PDF GENERATOR
# =========================
def create_pdf(r, expl, llm_expl):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    content = f"""
    Modality: {r.get('modality')}
    Prediction: {r.get('prediction')}
    Confidence: {float(r.get('confidence', 0.0)) * 100:.2f}%
    Real Probability: {float(r.get('prob_real', 0.0)) * 100:.2f}%
    Fake Probability: {float(r.get('prob_fake', 0.0)) * 100:.2f}%
    System Explanation:
    {expl}
    AI Explanation:
    {llm_expl if llm_expl else "N/A"}
    Transcript:
    {r.get('transcript', 'N/A')}
    """
    elements = [
        Paragraph("<b>Deepfake Detection Report</b>", styles["Title"]),
        Spacer(1, 0.2 * inch),
        Paragraph(content, styles["Normal"]),
    ]
    doc.build(elements)
    buffer.seek(0)
    return buffer
 
# =========================
# DATA
# =========================
r = st.session_state.get("result")
if not r:
    st.warning("No result found. Go to Detection and run analysis first.")
    st.stop()
 
modality  = (r.get("modality") or "video").lower()
pred      = str(r.get("prediction", "UNKNOWN")).upper()
conf      = float(r.get("confidence", 0.0)) * 100.0
p_real    = float(r.get("prob_real",  0.0)) * 100.0
p_fake    = float(r.get("prob_fake",  0.0)) * 100.0
transcript = r.get("transcript", "") or ""
llm_expl  = r.get("llm_explanation", "")
 
if modality == "image":
    expl = "Prediction is computed from the uploaded image using your trained MobileNetV3 model."
elif modality == "audio":
    expl = "Prediction is computed using audio MFCC features with your trained classifier."
else:
    expl = "Prediction is computed using transcript + MFCC features with your trained MobileNetV3 model."
 
is_fake = pred == "FAKE"
verdict_color  = "#E24B4A" if is_fake else "#1D9E75"
verdict_bg     = "#1a0a0a" if is_fake else "#0a1a12"
verdict_border = "#4d1a1a" if is_fake else "#1a4d2a"
conf_color     = "#E24B4A" if conf >= 80 else "#BA7517" if conf >= 50 else "#1D9E75"
 
# =========================
# TOP BAR
# =========================
st.markdown("###### CYBERSECURITY &nbsp;/&nbsp; RESULTS")
 
top_l, top_r = st.columns([4, 1], vertical_alignment="top")
with top_l:
    st.title("📊 Results")
    st.caption(f"Analysis complete · {modality.capitalize()} · Model v2.4.1")
with top_r:
    st.success("⬤  ANALYSIS COMPLETE")
    st.caption("FFmpeg • Whisper • v2.4.1")
 
st.markdown("---")
 
# =========================
# STEP TRACKER (step 4 active)
# =========================
st.markdown("""
<div style="display:flex;align-items:center;gap:0;margin-bottom:20px;">
  <div style="display:flex;align-items:center;gap:8px;padding:8px 14px;border-radius:8px;background:#0e1e14;border:0.5px solid #1a4d2a;">
    <div style="width:20px;height:20px;border-radius:50%;background:#1a4d2a;color:#22c55e;font-size:10px;font-weight:500;display:flex;align-items:center;justify-content:center;">✓</div>
    <span style="font-size:12px;color:#4ade80;">Select mode</span>
  </div>
  <span style="color:#2a2d35;padding:0 6px;font-size:14px;">›</span>
  <div style="display:flex;align-items:center;gap:8px;padding:8px 14px;border-radius:8px;background:#0e1e14;border:0.5px solid #1a4d2a;">
    <div style="width:20px;height:20px;border-radius:50%;background:#1a4d2a;color:#22c55e;font-size:10px;font-weight:500;display:flex;align-items:center;justify-content:center;">✓</div>
    <span style="font-size:12px;color:#4ade80;">Upload file</span>
  </div>
  <span style="color:#2a2d35;padding:0 6px;font-size:14px;">›</span>
  <div style="display:flex;align-items:center;gap:8px;padding:8px 14px;border-radius:8px;background:#0e1e14;border:0.5px solid #1a4d2a;">
    <div style="width:20px;height:20px;border-radius:50%;background:#1a4d2a;color:#22c55e;font-size:10px;font-weight:500;display:flex;align-items:center;justify-content:center;">✓</div>
    <span style="font-size:12px;color:#4ade80;">Run detection</span>
  </div>
  <span style="color:#2a2d35;padding:0 6px;font-size:14px;">›</span>
  <div style="display:flex;align-items:center;gap:8px;padding:8px 14px;border-radius:8px;background:#1a1d24;border:0.5px solid #3d4149;">
    <div style="width:20px;height:20px;border-radius:50%;background:#1e2533;color:#60a5fa;font-size:10px;font-weight:500;display:flex;align-items:center;justify-content:center;">4</div>
    <span style="font-size:12px;color:#e5e7eb;">View results</span>
  </div>
</div>
""", unsafe_allow_html=True)
 
# =========================
# VERDICT BANNER
# =========================
verdict_icon = "✕" if is_fake else "✓"
st.markdown(f"""
<div style="background:{verdict_bg};border:1px solid {verdict_border};border-radius:14px;padding:20px 24px;margin-bottom:20px;display:flex;align-items:center;justify-content:space-between;">
  <div style="display:flex;align-items:center;gap:16px;">
    <div style="width:48px;height:48px;border-radius:50%;background:{verdict_color}22;border:1.5px solid {verdict_color};display:flex;align-items:center;justify-content:center;font-size:22px;font-weight:500;color:{verdict_color};">{verdict_icon}</div>
    <div>
      <div style="font-size:11px;color:#6b7280;text-transform:uppercase;letter-spacing:.08em;margin-bottom:2px;">Verdict</div>
      <div style="font-size:26px;font-weight:500;color:{verdict_color};line-height:1;">{pred}</div>
    </div>
  </div>
  <div style="text-align:right;">
    <div style="font-size:11px;color:#6b7280;text-transform:uppercase;letter-spacing:.08em;margin-bottom:2px;">Confidence</div>
    <div style="font-size:26px;font-weight:500;color:{conf_color};line-height:1;">{conf:.1f}%</div>
  </div>
</div>
""", unsafe_allow_html=True)
 
# =========================
# KPI CARDS
# =========================
k1, k2, k3 = st.columns(3)
with k1:
    with st.container(border=True):
        st.caption("REAL PROBABILITY")
        st.markdown(f"<div style='font-size:28px;font-weight:500;color:#1D9E75;'>{p_real:.2f}%</div>", unsafe_allow_html=True)
with k2:
    with st.container(border=True):
        st.caption("FAKE PROBABILITY")
        st.markdown(f"<div style='font-size:28px;font-weight:500;color:#E24B4A;'>{p_fake:.2f}%</div>", unsafe_allow_html=True)
with k3:
    with st.container(border=True):
        st.caption("MODALITY")
        st.markdown(f"<div style='font-size:28px;font-weight:500;color:#60a5fa;'>{modality.capitalize()}</div>", unsafe_allow_html=True)
 
st.markdown("")
 
# =========================
# SYSTEM EXPLANATION
# =========================
with st.container(border=True):
    st.caption("SYSTEM EXPLANATION")
    st.write(expl)
 
st.markdown("")
 
# =========================
# TRANSCRIPT
# =========================
if modality in ("video", "audio"):
    with st.container(border=True):
        st.caption("📝 TRANSCRIPT")
        if transcript:
            st.code(transcript, language=None)
        else:
            st.caption("No transcript extracted.")
 
    st.markdown("")
 
# =========================
# LLM EXPLANATION
# =========================
if llm_expl:
    with st.container(border=True):
        st.caption("🧠 AI EXPLANATION (LLM)")
        st.write(llm_expl)
 
    st.markdown("")
 
# =========================
# BUTTONS
# =========================
col1, col2 = st.columns(2)
with col1:
    if st.button("🔁  Analyze another file", use_container_width=True):
        st.session_state["result"] = None
        st.switch_page("pages/1_Detection.py")
with col2:
    pdf_buffer = create_pdf(r, expl, llm_expl)
    st.download_button(
        "⬇️  Download PDF report",
        data=pdf_buffer,
        file_name="deepfake_report.pdf",
        mime="application/pdf",
        use_container_width=True,
    )
