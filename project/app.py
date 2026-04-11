import streamlit as st
st.set_page_config(page_title="Deepfake Detector", page_icon="🛡️", layout="wide")

import os
import plotly.graph_objects as go

# ── Inline design CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Reset & base ── */
[data-testid="stAppViewContainer"] { background: #F8F7F4; }
[data-testid="stMain"] > div { padding-top: 0 !important; }
section.main > div { max-width: 1200px; margin: 0 auto; padding: 20px 24px 40px; }
[data-testid="stRadio"] label { font-size: 12px !important; font-weight: 500 !important; }

/* ── Panel wrapper ── */
.panel {
    background: #FFFFFF;
    border: 0.5px solid rgba(0,0,0,0.1);
    border-radius: 12px;
    overflow: hidden;
    margin-bottom: 10px;
}

/* ── Header bar ── */
.db-header {
    background: #FFFFFF;
    border: 0.5px solid rgba(0,0,0,0.1);
    border-radius: 12px;
    padding: 14px 22px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 10px;
}
.db-brand { display: flex; align-items: center; gap: 12px; }
.db-shield {
    width: 38px; height: 38px; border-radius: 10px;
    background: #26215C;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px; line-height: 1;
}
.db-brand-name { font-size: 15px; font-weight: 600; color: #1a1a1a; margin: 0; }
.db-brand-sub  { font-size: 11px; color: #888; margin: 0; }
.db-online {
    display: inline-flex; align-items: center; gap: 6px;
    background: #EAF3DE; color: #27500A;
    font-size: 11px; font-weight: 600;
    padding: 5px 12px; border-radius: 20px;
}
.db-online-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: #639922; display: inline-block;
    animation: blink 2s infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.3} }

/* ── Micro label ── */
.micro {
    font-size: 10px; font-weight: 600;
    letter-spacing: .07em; text-transform: uppercase;
    color: #999; margin-bottom: 10px;
}

/* ── Verdict score ── */
.score-num {
    font-size: 58px; font-weight: 600; line-height: 1;
    letter-spacing: -2px; margin-bottom: 4px;
}
.score-label { font-size: 12px; color: #888; margin-bottom: 12px; }
.verdict-pill {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 4px 12px; border-radius: 20px;
    font-size: 11px; font-weight: 600;
}
.pill-fake { background: #FCEBEB; color: #791F1F; }
.pill-real { background: #EAF3DE; color: #27500A; }
.pill-unsure { background: #FAEEDA; color: #633806; }

/* ── Meter ── */
.meter-wrap { margin-top: 16px; }
.meter-row  { display: flex; justify-content: space-between; font-size: 10px; color: #aaa; margin-bottom: 3px; }
.meter-track { height: 4px; background: #F1EFE8; border-radius: 2px; }
.meter-fill  { height: 100%; border-radius: 2px; }
.meter-vals  { display: flex; justify-content: space-between; margin-top: 3px; font-size: 12px; font-weight: 600; }

/* ── Event log ── */
.ev-list { display: flex; flex-direction: column; gap: 0; }
.ev-item {
    display: flex; align-items: flex-start; gap: 10px;
    padding: 10px 0; border-bottom: 0.5px solid rgba(0,0,0,0.07);
}
.ev-item:first-child { padding-top: 0; }
.ev-item:last-child  { border-bottom: none; padding-bottom: 0; }
.ev-dot  { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; margin-top: 4px; }
.ev-title { font-size: 12px; font-weight: 600; color: #1a1a1a; }
.ev-sub   { font-size: 11px; color: #888; margin-top: 1px; }

/* ── Action buttons ── */
.act-btn {
    display: flex; align-items: center; gap: 8px;
    width: 100%; padding: 8px 12px; margin-bottom: 7px;
    border-radius: 8px; border: 0.5px solid rgba(0,0,0,0.1);
    background: #FFFFFF; cursor: pointer;
    font-size: 12px; font-weight: 500; color: #1a1a1a;
    text-align: left; transition: background .12s;
}
.act-btn:hover { background: #F8F7F4; }
.act-btn:last-child { margin-bottom: 0; }

/* ── Cell grid ── */
.cell-grid {
    display: grid; gap: 1px;
    background: rgba(0,0,0,0.07);
}
.cell-grid-2 { grid-template-columns: 1fr 1fr; }
.cell-grid-3 { grid-template-columns: 1fr 1fr 1fr; }
.cell {
    background: #FFFFFF;
    padding: 18px 22px;
}
.cell:first-child { border-radius: 12px 0 0 12px; }
.cell:last-child  { border-radius: 0 12px 12px 0; }

/* ── Footer strip ── */
.db-footer {
    background: #FFFFFF;
    border: 0.5px solid rgba(0,0,0,0.1);
    border-radius: 12px;
    padding: 12px 22px;
    display: grid;
    grid-template-columns: 1fr 1fr 1fr 1fr;
    gap: 0;
    margin-top: 10px;
}
.foot-cell { padding: 0 18px; border-left: 0.5px solid rgba(0,0,0,0.08); }
.foot-cell:first-child { padding-left: 0; border-left: none; }
.foot-key { font-size: 10px; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; color: #aaa; margin-bottom: 2px; }
.foot-val { font-size: 13px; font-weight: 600; color: #1a1a1a; }
.foot-sub { font-size: 11px; color: #888; }

/* ── Streamlit overrides ── */
[data-testid="stPlotlyChart"] { border: none !important; }
div[data-testid="column"] { padding: 0 !important; }
.stButton button {
    border-radius: 8px !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    border: 0.5px solid rgba(0,0,0,0.12) !important;
    background: #FFFFFF !important;
    color: #1a1a1a !important;
}
.stButton button:hover { background: #F8F7F4 !important; }
hr { margin: 12px 0 !important; border-color: rgba(0,0,0,0.07) !important; }
</style>
""", unsafe_allow_html=True)

# ── Data ──────────────────────────────────────────────────────────────────────
CLIPS = {
    "VID-9042": {
        "fake": 97.3, "real": 2.7,
        "detail": "High confidence", "verdict": "fake",
        "signals": {
            "Audio splice": 98, "Lip desync": 91, "MFCC variance": 87,
            "Face geometry": 72, "Transcript": 31, "Noise print": 44,
        },
        "events": [
            ("Audio splice at 0:03–0:07", "Confidence 98.1%", "#E24B4A"),
            ("Lip desync detected",       "4.84ms average lag", "#E24B4A"),
            ("MFCC anomaly flagged",       "Frames 47–62",       "#EF9F27"),
            ("Transcript parsed",          "Whisper — 99.2% WER","#639922"),
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
            ("No splice detected", "Clean waveform",   "#639922"),
            ("Lip sync nominal",   "< 10ms lag",        "#639922"),
            ("MFCC normal",        "All frames clear",  "#639922"),
            ("Transcript parsed",  "High coherence",    "#639922"),
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
            ("Inconclusive audio", "Mixed signals",     "#EF9F27"),
            ("Lip lag detected",   "32ms avg",           "#EF9F27"),
            ("MFCC elevated",      "Suspicious range",  "#EF9F27"),
            ("Transcript gaps",    "Minor anomalies",   "#EF9F27"),
        ],
    },
}

COLOR_MAP = {
    "fake":  {"score": "#E24B4A", "fill": "rgba(226,75,74,0.15)",  "line": "#E24B4A", "pill": "pill-fake",   "label": "FAKE — high confidence"},
    "real":  {"score": "#1D9E75", "fill": "rgba(29,158,117,0.15)", "line": "#1D9E75", "pill": "pill-real",   "label": "Authentic — all signals clear"},
    "unsure":{"score": "#EF9F27", "fill": "rgba(239,159,39,0.15)", "line": "#EF9F27", "pill": "pill-unsure", "label": "Probable fake — review needed"},
}

def sig_color(val, verdict):
    if verdict == "real": return "#1D9E75"
    if val >= 70: return "#E24B4A"
    if val >= 40: return "#EF9F27"
    return "#1D9E75"

# ── Plotly radar ──────────────────────────────────────────────────────────────
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
            radialaxis=dict(
                visible=True, range=[0, 1],
                tickvals=[0.25, 0.5, 0.75, 1.0],
                tickfont=dict(size=9, color="#aaa"),
                gridcolor="rgba(0,0,0,0.07)",
                linecolor="rgba(0,0,0,0.07)",
            ),
            angularaxis=dict(
                tickfont=dict(size=11, color="#555"),
                gridcolor="rgba(0,0,0,0.06)",
                linecolor="rgba(0,0,0,0.08)",
            ),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=False,
        margin=dict(l=55, r=55, t=30, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=300,
    )
    return fig

# ── Plotly horizontal bars ────────────────────────────────────────────────────
def build_bar(clip):
    labels = list(clip["signals"].keys())
    vals   = list(clip["signals"].values())
    colors = [sig_color(v, clip["verdict"]) for v in vals]
    fig = go.Figure(go.Bar(
        x=vals, y=labels,
        orientation="h",
        marker_color=colors,
        marker_line_width=0,
        text=[f"{v}%" for v in vals],
        textposition="outside",
        textfont=dict(size=11, color="#555"),
    ))
    fig.update_layout(
        xaxis=dict(range=[0, 125], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(tickfont=dict(size=11, color="#555"), showgrid=False),
        margin=dict(l=10, r=50, t=6, b=6),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=220,
        showlegend=False,
        bargap=0.35,
    )
    return fig

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="db-header">
  <div class="db-brand">
    <div class="db-shield">🛡️</div>
    <div>
      <div class="db-brand-name">Deepfake Detector</div>
      <div class="db-brand-sub">Multimodal forensic analysis · v2.4.1</div>
    </div>
  </div>
  <div class="db-online">
    <span class="db-online-dot"></span> System online
  </div>
</div>
""", unsafe_allow_html=True)

# ── Clip selector ─────────────────────────────────────────────────────────────
clip_id = st.radio(
    "Select clip",
    options=list(CLIPS.keys()),
    horizontal=True,
    label_visibility="collapsed",
)
clip = CLIPS[clip_id]
c    = COLOR_MAP[clip["verdict"]]

st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

# ── Row 1: Verdict + Radar + Signals ─────────────────────────────────────────
col_verdict, col_radar, col_signals = st.columns([1, 1.6, 1.6], gap="small")

with col_verdict:
    st.markdown(f"""
    <div class="panel" style="padding:18px 22px;height:100%">
      <div class="micro">Verdict</div>
      <div class="score-num" style="color:{c['score']}">{clip['fake']:.1f}%</div>
      <div class="score-label">Fake probability</div>
      <span class="verdict-pill {c['pill']}">{c['label']}</span>
      <div class="meter-wrap">
        <div class="meter-row"><span>Fake</span><span>Real</span></div>
        <div class="meter-track">
          <div class="meter-fill" style="width:{clip['fake']}%;background:{c['score']}"></div>
        </div>
        <div class="meter-vals">
          <span style="color:{c['score']}">{clip['fake']:.1f}%</span>
          <span style="color:#1D9E75">{clip['real']:.1f}%</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

with col_radar:
    st.markdown('<div class="panel" style="padding:16px 20px">', unsafe_allow_html=True)
    st.markdown('<div class="micro">Signal radar</div>', unsafe_allow_html=True)
    st.plotly_chart(build_radar(clip), use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

with col_signals:
    st.markdown('<div class="panel" style="padding:16px 20px">', unsafe_allow_html=True)
    st.markdown('<div class="micro">Forensic signal breakdown</div>', unsafe_allow_html=True)
    st.plotly_chart(build_bar(clip), use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

# ── Row 2: Events + Actions + Model ──────────────────────────────────────────
col_ev, col_act, col_model = st.columns([2, 1.2, 1], gap="small")

with col_ev:
    events_html = "".join([
        f"""<div class="ev-item">
              <div class="ev-dot" style="background:{dot}"></div>
              <div>
                <div class="ev-title">{msg}</div>
                <div class="ev-sub">{sub}</div>
              </div>
            </div>"""
        for msg, sub, dot in clip["events"]
    ])
    st.markdown(f"""
    <div class="panel" style="padding:18px 22px">
      <div class="micro">Detection events</div>
      <div class="ev-list">{events_html}</div>
    </div>
    """, unsafe_allow_html=True)

with col_act:
    st.markdown('<div class="panel" style="padding:18px 22px">', unsafe_allow_html=True)
    st.markdown('<div class="micro">Actions</div>', unsafe_allow_html=True)
    st.button("📋  Generate evidence report", use_container_width=True, key="btn_report")
    st.button("🔍  Explain this detection",   use_container_width=True, key="btn_explain")
    st.button("⚖️  Compare clips",             use_container_width=True, key="btn_compare")
    st.button("📤  Upload new clip",           use_container_width=True, key="btn_upload")
    st.markdown('</div>', unsafe_allow_html=True)

with col_model:
    st.markdown(f"""
    <div class="panel" style="padding:18px 22px;height:100%">
      <div class="micro">Model</div>
      <div style="font-size:22px;font-weight:600;color:#1a1a1a;margin-bottom:2px">v2.4.1</div>
      <div style="font-size:11px;color:#888;margin-bottom:14px">Whisper large-v3 · FFmpeg</div>
      <div style="border-top:0.5px solid rgba(0,0,0,0.08);padding-top:12px">
        <div style="font-size:10px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:#aaa;margin-bottom:3px">Frames scanned</div>
        <div style="font-size:15px;font-weight:600;color:#1a1a1a">47</div>
      </div>
      <div style="margin-top:10px">
        <div style="font-size:10px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:#aaa;margin-bottom:3px">Processing time</div>
        <div style="font-size:15px;font-weight:600;color:#1a1a1a">1.84s</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── Footer strip ──────────────────────────────────────────────────────────────
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
