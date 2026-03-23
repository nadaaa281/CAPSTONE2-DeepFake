import streamlit as st

st.set_page_config(page_title="Deepfake Detector", page_icon="🛡️", layout="wide")

with open("assets/cyber.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Top bar
top_l, top_r = st.columns([3, 1], vertical_alignment="top")
with top_l:
    st.markdown("###### CYBERSECURITY • MEDIA FORENSICS")
    st.title("🛡️ Deepfake Detector")
    st.caption("Detect manipulated media using transcript + audio forensic features.")

with top_r:
    st.markdown("")
    st.success("SYSTEM ONLINE")
    st.caption("FFmpeg • Whisper • Trained Model")

st.markdown("---")

# Core value blocks (short)
c1, c2, c3 = st.columns(3, vertical_alignment="top")

with c1:
    with st.container(border=True):
        st.subheader("Analyze")
        st.write("Upload a video clip and run detection.")
        st.caption("Best: 5–15 seconds")

with c2:
    with st.container(border=True):
        st.subheader("Evidence")
        st.write("Transcript + probability breakdown.")
        st.caption("REAL vs FAKE")

with c3:
    with st.container(border=True):
        st.subheader("Export")
        st.write("Download the result report.")
        st.caption("For submission")

st.markdown("")

# One clean call-to-action
st.info("➡️ Open **Detection** from the left sidebar to start.")
