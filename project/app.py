import streamlit as st
st.set_page_config(page_title="Deepfake Detector", page_icon="🛡️", layout="wide")
 
import os
import plotly.graph_objects as go
 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
css_path = os.path.join(BASE_DIR, "assets/cyber.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
 
CLIPS = {
    "VID-9042": {
        "fake": 97.3, "real": 2.7,
        "detail": "High confidence", "verdict": "fake",
        "signals": {
            "Audio splice": 98, "Lip desync": 91, "MFCC variance": 87,
            "Face geometry": 72, "Transcript": 31, "Noise print": 44,
        },
        "events": [
            ("Audio splice at 0:03–0:07", "Confidence 98.1%", "🔴"),
            ("Lip desync detected", "Δ 84ms average lag", "🔴"),
            ("MFCC anomaly flagged", "Frames 47–62", "🟡"),
            ("Transcript parsed", "Whisper — 99.2% WER", "🟢"),
        ],
    },
    "VID-9041": {
        "fake": 4.2, "real": 95.8,
        "detail": "Likely authentic", "verdict": "real",
        "signals": {
            "Audio splice": 6, "Lip desync": 9, "MFCC variance": 12,
            "Face geometry": 8, "Transcript": 5, "Noise print": 11,
        },
        "events": [
            ("No splice detected", "Clean waveform", "🟢"),
            ("Lip sync nominal", "<10ms lag", "🟢"),
            ("MFCC normal", "All frames clear", "🟢"),
            ("Transcript parsed", "High coherence", "🟢"),
        ],
    },
    "VID-9040": {
        "fake": 62.1, "real": 37.9,
        "detail": "Below threshold — review needed", "verdict": "unsure",
        "signals": {
            "Audio splice": 55, "Lip desync": 48, "MFCC variance": 61,
            "Face geometry": 38, "Transcript": 42, "Noise print": 57,
        },
        "events": [
            ("Inconclusive audio", "Mixed signals", "🟡"),
            ("Lip lag detected", "32ms avg", "🟡"),
            ("MFCC elevated", "Suspicious range", "🟡"),
            ("Transcript gaps", "Minor anomalies", "🟡"),
        ],
    },
}
 
COLOR_MAP = {
    "fake":  {"score": "#E24B4A", "fill": "rgba(226,75,74,0.18)",  "line": "#E24B4A", "badge": "🔴 FAKE"},
    "real":  {"score": "#1D9E75", "fill": "rgba(29,158,117,0.18)", "line": "#1D9E75", "badge": "🟢 REAL"},
    "unsure":{"score": "#BA7517", "fill": "rgba(186,117,23,0.18)", "line": "#BA7517", "badge": "🟡 REVIEW"},
}
 
def signal_color(val, verdict):
    if verdict == "real":
        return "#1D9E75"
    if val >= 70:
        return "#E24B4A"
    if val >= 40:
        return "#BA7517"
    return "#1D9E75"
 
def build_radar(clip):
    labels = list(clip["signals"].keys())
    vals   = [v / 100 for v in clip["signals"].values()]
    c = COLOR_MAP[clip["verdict"]]
    fig = go.Figure(go.Scatterpolar(
        r=vals + [vals[0]],
        theta=labels + [labels[0]],
        fill="toself",
        fillcolor=c["fill"],
        line=dict(color=c["line"], width=2),
        marker=dict(size=5, color=c["line"]),
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1],
                            tickvals=[0.25, 0.5, 0.75, 1.0],
                            tickfont=dict(size=9),
                            gridcolor="rgba(128,128,128,0.2)"),
            angularaxis=dict(tickfont=dict(size=11),
                             gridcolor="rgba(128,128,128,0.15)"),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=False,
        margin=dict(l=50, r=50, t=30, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=340,
    )
    return fig
 
def build_bar(clip):
    labels = list(clip["signals"].keys())
    vals   = list(clip["signals"].values())
    colors = [signal_color(v, clip["verdict"]) for v in vals]
    fig = go.Figure(go.Bar(
        x=vals, y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{v}%" for v in vals],
        textposition="outside",
        textfont=dict(size=11),
    ))
    fig.update_layout(
        xaxis=dict(range=[0, 120], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(tickfont=dict(size=11)),
        margin=dict(l=10, r=40, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=240,
        showlegend=False,
    )
    return fig
 
 
# ── Top bar ──────────────────────────────────────────────────────────────────
top_l, top_r = st.columns([3, 1], vertical_alignment="top")
with top_l:
    st.markdown("###### CYBERSECURITY • MEDIA FORENSICS")
    st.title("🛡️ Deepfake Detector")
    st.caption("Detect manipulated media using transcript + audio forensic features.")
with top_r:
    st.success("⬤  SYSTEM ONLINE")
    st.caption("FFmpeg • Whisper • Trained Model")
 
st.markdown("---")
 
# ── Clip selector ─────────────────────────────────────────────────────────────
clip_id = st.radio(
    "Select clip",
    options=list(CLIPS.keys()),
    horizontal=True,
    label_visibility="collapsed",
)
clip = CLIPS[clip_id]
c = COLOR_MAP[clip["verdict"]]
 
st.markdown("")
 
# ── Main row: radar + verdict cards ──────────────────────────────────────────
radar_col, verdict_col = st.columns([3, 1], vertical_alignment="top")
 
with radar_col:
    with st.container(border=True):
        st.plotly_chart(build_radar(clip), use_container_width=True, config={"displayModeBar": False})
 
with verdict_col:
    with st.container(border=True):
        st.caption("FAKE PROBABILITY")
        st.markdown(f"<h2 style='color:{COLOR_MAP['fake']['score']};margin:0'>{clip['fake']:.1f}%</h2>", unsafe_allow_html=True)
        st.caption(clip["detail"])
        st.markdown(c["badge"])
 
    st.markdown("")
 
    with st.container(border=True):
        st.caption("REAL PROBABILITY")
        st.markdown(f"<h2 style='color:{COLOR_MAP['real']['score']};margin:0'>{clip['real']:.1f}%</h2>", unsafe_allow_html=True)
        st.caption("Authentic likelihood")
 
    st.markdown("")
 
    with st.container(border=True):
        st.caption("MODEL")
        st.markdown("<h4 style='margin:0'>v2.4.1</h4>", unsafe_allow_html=True)
        st.caption("Whisper large-v3 · FFmpeg")
 
st.markdown("")
 
# ── Signal bars ───────────────────────────────────────────────────────────────
with st.container(border=True):
    st.caption("FORENSIC SIGNAL BREAKDOWN")
    st.plotly_chart(build_bar(clip), use_container_width=True, config={"displayModeBar": False})
 
st.markdown("")
 
# ── Event log + actions ───────────────────────────────────────────────────────
log_col, action_col = st.columns([3, 1], vertical_alignment="top")
 
with log_col:
    with st.container(border=True):
        st.caption("EVENT LOG")
        for msg, sub, dot in clip["events"]:
            st.markdown(f"{dot} **{msg}** — {sub}")
 
with action_col:
    with st.container(border=True):
        st.caption("ACTIONS")
        st.button("Generate evidence report", use_container_width=True)
        st.button("Explain this detection", use_container_width=True)
        st.button("Compare clips", use_container_width=True)
        st.button("Upload new clip", use_container_width=True)
 
st.markdown("")
st.info("➡️ Open **Detection** from the left sidebar to start.")
