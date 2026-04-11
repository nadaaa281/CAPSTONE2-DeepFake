import streamlit as st
st.set_page_config(page_title="Deepfake Detector", page_icon="🛡️", layout="wide")

import plotly.graph_objects as go

# ── CSS: full dark panel design ───────────────────────────────────────────────
st.markdown("""
<style>
/* Page background */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="block-container"] {
    background-color: #0E0E11 !important;
    color: #E2E2E2 !important;
}

[data-testid="block-container"] {
    padding: 20px 28px 48px !important;
    max-width: 100% !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden !important; }
[data-testid="stToolbar"]  { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }

/* Kill default white borders on containers */
[data-testid="stVerticalBlock"] > [data-testid="element-container"] > div {
    background: transparent !important;
}

/* Column layout */
[data-testid="stHorizontalBlock"] {
    gap: 10px !important;
    align-items: stretch !important;
}
[data-testid="stColumn"] { padding: 0 !important; }

/* ── Radio as dark tab strip ── */
[data-testid="stRadio"] > div {
    flex-direction: row !important;
    gap: 3px !important;
    background: #1A1A22 !important;
    padding: 4px !important;
    border-radius: 9px !important;
    width: fit-content !important;
    border: 0.5px solid rgba(255,255,255,0.07) !important;
    margin-bottom: 12px !important;
}
[data-testid="stRadio"] label {
    background: transparent !important;
    border-radius: 6px !important;
    padding: 5px 18px !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    color: #666 !important;
    cursor: pointer !important;
    border: none !important;
}
[data-testid="stRadio"] label:has(input:checked) {
    background: #22223A !important;
    color: #AFA9EC !important;
    font-weight: 600 !important;
    border: 0.5px solid rgba(127,119,221,0.25) !important;
}
/* hide the radio circle */
[data-testid="stRadio"] [data-testid="stMarkdownContainer"] p { display: none; }
[data-testid="stRadio"] input[type="radio"] { display: none !important; }

/* ── Buttons ── */
.stButton > button {
    width: 100% !important;
    background: #1A1A22 !important;
    border: 0.5px solid rgba(255,255,255,0.08) !important;
    border-radius: 8px !important;
    color: #ABABBB !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    padding: 9px 14px !important;
    text-align: left !important;
}
.stButton > button:hover {
    background: #22222E !important;
    border-color: rgba(255,255,255,0.15) !important;
    color: #E2E2E2 !important;
}

/* ── Plotly chart containers ── */
[data-testid="stPlotlyChart"] {
    background: transparent !important;
    border: none !important;
}

/* ── Panel card ── */
.panel {
    background: #16161E;
    border: 0.5px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 18px 22px;
    height: 100%;
    box-sizing: border-box;
}

/* ── Micro label ── */
.micro {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: .08em;
    text-transform: uppercase;
    color: #444;
    margin-bottom: 12px;
}

/* ── Score ── */
.score-num {
    font-size: 62px;
    font-weight: 700;
    line-height: 1;
    letter-spacing: -2px;
    margin-bottom: 4px;
}
.score-lbl { font-size: 12px; color: #555; margin-bottom: 14px; }

/* ── Verdict pill ── */
.pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
}
.pill-fake   { background: rgba(226,75,74,0.15);  color: #F09595; }
.pill-real   { background: rgba(29,158,117,0.15); color: #5DCAA5; }
.pill-unsure { background: rgba(239,159,39,0.15); color: #FAC775; }

/* ── Meter ── */
.meter { margin-top: 18px; }
.meter-hd {
    display: flex;
    justify-content: space-between;
    font-size: 10px;
    color: #444;
    margin-bottom: 4px;
}
.meter-track { height: 3px; background: #1E1E28; border-radius: 2px; }
.meter-fill  { height: 100%; border-radius: 2px; }
.meter-vals  {
    display: flex;
    justify-content: space-between;
    margin-top: 4px;
    font-size: 12px;
    font-weight: 600;
}

/* ── Header bar ── */
.db-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #16161E;
    border: 0.5px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 14px 22px;
    margin-bottom: 12px;
}
.brand      { display: flex; align-items: center; gap: 12px; }
.shield {
    width: 38px; height: 38px; border-radius: 10px;
    background: #26215C;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
}
.brand-name { font-size: 15px; font-weight: 700; color: #E2E2E2; margin: 0; }
.brand-sub  { font-size: 11px; color: #444; margin: 0; }
.online-pill {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(99,153,34,0.15); color: #97C459;
    font-size: 11px; font-weight: 600;
    padding: 5px 14px; border-radius: 20px;
}
.live-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: #639922; display: inline-block;
    animation: blink 2s infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.2} }

/* ── Event log ── */
.ev-list { display: flex; flex-direction: column; }
.ev-item {
    display: flex; gap: 10px;
    padding: 10px 0;
    border-bottom: 0.5px solid rgba(255,255,255,0.04);
}
.ev-item:first-child { padding-top: 0; }
.ev-item:last-child  { border-bottom: none; padding-bottom: 0; }
.ev-dot   { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; margin-top: 5px; }
.ev-title { font-size: 12px; font-weight: 600; color: #D0D0DC; }
.ev-sub   { font-size: 11px; color: #555; margin-top: 1px; }

/* ── Footer strip ── */
.db-footer {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr 1fr;
    background: #16161E;
    border: 0.5px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 14px 22px;
    margin-top: 10px;
}
.foot-cell {
    padding: 0 20px;
    border-left: 0.5px solid rgba(255,255,255,0.05);
}
.foot-cell:first-child { padding-left: 0; border-left: none; }
.foot-key { font-size: 10px; font-weight: 600; letter-spacing:.07em; text-transform:uppercase; color:#444; margin-bottom:3px; }
.foot-val { font-size: 13px; font-weight: 600; color: #C0C0CC; }
.foot-sub { font-size: 11px; color: #555; }

.spacer { height: 10px; }
</style>
""", unsafe_allow_html=True)

# ── Data ──────────────────────────────────────────────────────────────────────
CLIPS = {
    "VID-9042": {
        "fake": 97.3, "real": 2.7, "verdict": "fake",
        "signals": {
            "Lip desync": 91, "MFCC variance": 87,
            "Face geometry": 72, "Audio splice": 98,
            "Transcript": 31,  "Noise print": 44,
        },
        "events": [
            ("Audio splice at 0:03–0:07", "Confidence 98.1%",     "#E24B4A"),
            ("Lip desync detected",        "4.84ms average lag",   "#E24B4A"),
            ("MFCC anomaly flagged",        "Frames 47–62",         "#EF9F27"),
            ("Transcript parsed",           "Whisper — 99.2% WER", "#639922"),
        ],
    },
    "VID-9041": {
        "fake": 4.2, "real": 95.8, "verdict": "real",
        "signals": {
            "Lip desync": 9,  "MFCC variance": 12,
            "Face geometry": 8,  "Audio splice": 6,
            "Transcript": 5,  "Noise print": 11,
        },
        "events": [
            ("No splice detected", "Clean waveform",   "#639922"),
            ("Lip sync nominal",   "< 10ms lag",        "#639922"),
            ("MFCC normal",        "All frames clear",  "#639922"),
            ("Transcript parsed",  "High coherence",    "#639922"),
        ],
    },
    "VID-9040": {
        "fake": 62.1, "real": 37.9, "verdict": "unsure",
        "signals": {
            "Lip desync": 48, "MFCC variance": 61,
            "Face geometry": 38, "Audio splice": 55,
            "Transcript": 42,  "Noise print": 57,
        },
        "events": [
            ("Inconclusive audio", "Mixed signals",     "#EF9F27"),
            ("Lip lag detected",   "32ms avg",           "#EF9F27"),
            ("MFCC elevated",      "Suspicious range",  "#EF9F27"),
            ("Transcript gaps",    "Minor anomalies",   "#EF9F27"),
        ],
    },
}

COLOR_MAP = {
    "fake":  {"score":"#E24B4A","fill":"rgba(226,75,74,0.18)",  "line":"#E24B4A","pill":"pill-fake",  "label":"Fake — high confidence"},
    "real":  {"score":"#1D9E75","fill":"rgba(29,158,117,0.18)","line":"#1D9E75","pill":"pill-real",  "label":"Authentic — all clear"},
    "unsure":{"score":"#EF9F27","fill":"rgba(239,159,39,0.18)","line":"#EF9F27","pill":"pill-unsure","label":"Probable fake — review needed"},
}

def sig_color(v, verdict):
    if verdict == "real": return "#1D9E75"
    if v >= 70: return "#E24B4A"
    if v >= 40: return "#EF9F27"
    return "#1D9E75"

# ── Plotly radar ──────────────────────────────────────────────────────────────
def build_radar(clip):
    cm = COLOR_MAP[clip["verdict"]]
    labels = list(clip["signals"].keys())
    vals   = [v / 100 for v in clip["signals"].values()]
    fig = go.Figure(go.Scatterpolar(
        r=vals+[vals[0]], theta=labels+[labels[0]],
        fill="toself", fillcolor=cm["fill"],
        line=dict(color=cm["line"], width=2),
        marker=dict(size=5, color=cm["line"]),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True, range=[0,1],
                tickvals=[.25,.5,.75,1],
                tickfont=dict(size=9, color="#444"),
                gridcolor="rgba(255,255,255,0.05)",
                linecolor="rgba(255,255,255,0.05)",
            ),
            angularaxis=dict(
                tickfont=dict(size=11, color="#777"),
                gridcolor="rgba(255,255,255,0.04)",
                linecolor="rgba(255,255,255,0.05)",
            ),
        ),
        showlegend=False,
        margin=dict(l=55,r=55,t=24,b=24),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=268, font=dict(color="#777"),
    )
    return fig

# ── Plotly horizontal bar ─────────────────────────────────────────────────────
def build_bar(clip):
    labels = list(clip["signals"].keys())
    vals   = list(clip["signals"].values())
    colors = [sig_color(v, clip["verdict"]) for v in vals]
    fig = go.Figure(go.Bar(
        x=vals, y=labels, orientation="h",
        marker=dict(color=colors, line_width=0),
        text=[f"  {v}%" for v in vals],
        textposition="outside",
        textfont=dict(size=11, color="#666"),
    ))
    fig.update_layout(
        xaxis=dict(range=[0,130], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(tickfont=dict(size=11, color="#777"), showgrid=False, tickcolor="rgba(0,0,0,0)"),
        margin=dict(l=8,r=52,t=4,b=4),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=226, showlegend=False, bargap=0.40,
        font=dict(color="#666"),
    )
    return fig

# ═══════════════════════════════════════════════════════════════════════════════
# LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════

# Header
st.markdown("""
<div class="db-header">
  <div class="brand">
    <div class="shield">🛡️</div>
    <div>
      <div class="brand-name">Deepfake Detector</div>
      <div class="brand-sub">Multimodal forensic analysis · v2.4.1</div>
    </div>
  </div>
  <div class="online-pill"><span class="live-dot"></span> System online</div>
</div>
""", unsafe_allow_html=True)

# Clip selector
clip_id = st.radio("", list(CLIPS.keys()), horizontal=True, label_visibility="collapsed")
clip    = CLIPS[clip_id]
cm      = COLOR_MAP[clip["verdict"]]

# Row 1
c1, c2, c3 = st.columns([1, 1.5, 1.7], gap="small")

with c1:
    st.markdown(f"""
    <div class="panel">
      <div class="micro">Verdict</div>
      <div class="score-num" style="color:{cm['score']}">{clip['fake']:.1f}%</div>
      <div class="score-lbl">Fake probability</div>
      <span class="pill {cm['pill']}">● &nbsp;{cm['label']}</span>
      <div class="meter">
        <div class="meter-hd"><span>Fake</span><span>Real</span></div>
        <div class="meter-track">
          <div class="meter-fill" style="width:{clip['fake']}%;background:{cm['score']}"></div>
        </div>
        <div class="meter-vals">
          <span style="color:{cm['score']}">{clip['fake']:.1f}%</span>
          <span style="color:#1D9E75">{clip['real']:.1f}%</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="micro">Signal radar</div>', unsafe_allow_html=True)
    st.plotly_chart(build_radar(clip), use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

with c3:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="micro">Forensic signal breakdown</div>', unsafe_allow_html=True)
    st.plotly_chart(build_bar(clip), use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)

# Row 2
c4, c5, c6 = st.columns([2, 1.2, 0.9], gap="small")

with c4:
    evts = "".join([
        f'<div class="ev-item">'
        f'<div class="ev-dot" style="background:{dot}"></div>'
        f'<div><div class="ev-title">{msg}</div>'
        f'<div class="ev-sub">{sub}</div></div></div>'
        for msg, sub, dot in clip["events"]
    ])
    st.markdown(f"""
    <div class="panel">
      <div class="micro">Detection events</div>
      <div class="ev-list">{evts}</div>
    </div>
    """, unsafe_allow_html=True)

with c5:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="micro">Actions</div>', unsafe_allow_html=True)
    st.button("📋  Generate evidence report", key="r", use_container_width=True)
    st.button("🔍  Explain this detection",   key="e", use_container_width=True)
    st.button("⚖️  Compare clips",             key="c", use_container_width=True)
    st.button("📤  Upload new clip",           key="u", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with c6:
    st.markdown(f"""
    <div class="panel">
      <div class="micro">Model</div>
      <div style="font-size:24px;font-weight:700;color:#D8D8E8;margin-bottom:2px">v2.4.1</div>
      <div style="font-size:11px;color:#444;margin-bottom:16px">Whisper large-v3 · FFmpeg</div>
      <div style="border-top:0.5px solid rgba(255,255,255,0.05);padding-top:14px;display:flex;flex-direction:column;gap:12px">
        <div>
          <div style="font-size:10px;font-weight:600;letter-spacing:.07em;text-transform:uppercase;color:#444;margin-bottom:2px">Frames scanned</div>
          <div style="font-size:18px;font-weight:700;color:#C0C0CC">47</div>
        </div>
        <div>
          <div style="font-size:10px;font-weight:600;letter-spacing:.07em;text-transform:uppercase;color:#444;margin-bottom:2px">Processing time</div>
          <div style="font-size:18px;font-weight:700;color:#C0C0CC">1.84s</div>
        </div>
        <div>
          <div style="font-size:10px;font-weight:600;letter-spacing:.07em;text-transform:uppercase;color:#444;margin-bottom:2px">Resolution</div>
          <div style="font-size:18px;font-weight:700;color:#C0C0CC">4K</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="db-footer">
  <div class="foot-cell">
    <div class="foot-key">Model version</div>
    <div class="foot-val">v2.4.1</div>
  </div>
  <div class="foot-cell">
    <div class="foot-key">Audio engine</div>
    <div class="foot-val">Whisper large-v3</div>
  </div>
  <div class="foot-cell">
    <div class="foot-key">Video pipeline</div>
    <div class="foot-val">FFmpeg + ResNet-50</div>
  </div>
  <div class="foot-cell">
    <div class="foot-key">Resolution</div>
    <div class="foot-val">4K</div>
    <div class="foot-sub">47 frames scanned</div>
  </div>
</div>
""", unsafe_allow_html=True)
