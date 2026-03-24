import io
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import streamlit as st

st.set_page_config(page_title="Results", page_icon="📊", layout="wide")

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(BASE_DIR, "assets/cyber.css")) as f:

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
        Paragraph(content, styles["Normal"])
    ]

    doc.build(elements)
    buffer.seek(0)
    return buffer


# =========================
# UI
# =========================
st.title("📊 Results")

r = st.session_state.get("result")
if not r:
    st.warning("No result found. Go to Detection and run analysis.")
    st.stop()

modality = (r.get("modality") or "video").lower()

pred = str(r.get("prediction", "UNKNOWN")).upper()
conf = float(r.get("confidence", 0.0)) * 100.0
p_real = float(r.get("prob_real", 0.0)) * 100.0
p_fake = float(r.get("prob_fake", 0.0)) * 100.0
transcript = r.get("transcript", "") or ""

badge_class = "fake" if pred == "FAKE" else "real"

# Rule-based explanation
if modality == "image":
    expl = "Prediction is computed from the uploaded image using your trained ResNet model."
elif modality == "audio":
    expl = "Prediction is computed using audio MFCC features with your trained classifier."
else:
    expl = "Prediction is computed using transcript + MFCC features with your trained model."

# LLM explanation
llm_expl = r.get("llm_explanation", "")

st.markdown(f"""
<div class="df-card">
  <div class="df-badge {badge_class}">VERDICT: {pred}</div>

  <div class="df-kpi">
    <div class="item"><div class="label">Confidence</div><div class="value">{conf:.2f}%</div></div>
    <div class="item"><div class="label">REAL probability</div><div class="value">{p_real:.2f}%</div></div>
    <div class="item"><div class="label">FAKE probability</div><div class="value">{p_fake:.2f}%</div></div>
  </div>

  <div class="df-hr"></div>

  <b>System Explanation</b><br>
  <span style="color: rgba(230,230,230,.78);">{expl}</span>
</div>
""", unsafe_allow_html=True)

# Transcript
if modality in ("video", "audio"):
    st.subheader("📝 Transcript")
    st.markdown(
        f"<div class='df-card-plain df-mono'>{transcript if transcript else 'No transcript extracted.'}</div>",
        unsafe_allow_html=True
    )

# LLM explanation
if llm_expl:
    st.markdown("### 🧠 AI Explanation (LLM)")
    st.markdown(
        f"<div class='df-card-plain'>{llm_expl}</div>",
        unsafe_allow_html=True
    )

# =========================
# BUTTONS
# =========================
col1, col2 = st.columns(2)

with col1:
    if st.button("🔁 Analyze another file", use_container_width=True):
        st.session_state["result"] = None
        st.switch_page("pages/1_Detection.py")

with col2:
    pdf_buffer = create_pdf(r, expl, llm_expl)

    st.download_button(
        "⬇️ Download PDF Report",
        data=pdf_buffer,
        file_name="deepfake_report.pdf",
        mime="application/pdf",
        use_container_width=True
    )