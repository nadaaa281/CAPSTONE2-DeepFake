import streamlit as st
st.set_page_config(page_title="Deepfake Detector", page_icon="🛡️", layout="wide")

import plotly.graph_objects as go

st.markdown("""
<style>
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="block-container"] {
    background-color: #0E0E11 !important;
    color: #E2E2E2 !important;
}
[data-testid="block-container"] {
    padding: 24px 32px 48px !important;
    max-width: 860px !important;
    margin: 0 auto !important;
}
#MainMenu, footer, header { visibility: hidden !important; }
[data-testid="stToolbar"]  { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }
[data-testid="stVerticalBlock"] > [data-testid="element-container"] > div {
    background: transparent !important;
}
[data-testid="stHorizontalBlock"] { gap: 0px !important; align-items: stretch !important; }
[data-testid="stColumn"] { padding: 0 !important; }

/* Radio as pill tabs */
[data-testid="stRadio"] > div {
    flex-direction: row !important;
    gap: 6px !important;
    background: transparent !important;
    padding: 0 !important;
    border: none !important;
    margin-bottom: 0 !important;
}
[data-testid="stRadio"] label {
    background: #1C1C26 !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 8px !important;
    padding: 6px 20px !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    color: #888 !important;
    cursor: pointer !important;
}
[data-testid="stRadio"] label:has(input:checked) {
    background: #1C1C26 !important;
    color: #E2E2E2 !important;
    border: 1px solid rgba(255,255,255,0.35) !important;
}
[data-testid="stRadio"] [data-testid="stMarkdownContainer"] p { display: none; }
[data-testid="stRadio"] input[type="radio"] { display: none !important; }

/* Buttons */
.stButton > button {
    background: #1A1A26 !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 8px !important;
    color: #ABABBB !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    padding: 8px 16px !important;
}
.stButton > button:hover {
    background: #22222E !important;
    border-color: rgba(255,255,255,0.22) !important;
    color: #E2E2E2 !important;
}

/* ── OUTER SHELL ── */
.shell {
    background: #17171F;
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 16px;
    overflow: hidden;
}

/* ── TOP BAR ── */
.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 20px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}
.brand { display: flex; align-items: center; gap: 10px; }
.shield-icon {
    width: 34px; height: 34px; border-radius: 8px;
    background: #26215C;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px;
}
.brand-name { font-size: 14px; font-weight: 700; color: #E2E2E2; margin: 0; line-height: 1.2; }
.brand-sub  { font-size: 11px; color: #444; margin: 0; }
.sys-online {
    display: inline-flex; align-items: center; gap: 5px;
    background: rgba(99,153,34,0.15); color: #97C459;
    font-size: 11px; font-weight: 600;
    padding: 4px 12px; border-radius: 20px;
}
.live-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: #639922; display: inline-block;
    animation: blink 2s infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.2} }

/* ── SECTION LABEL ── */
.sec-label {
    font-size: 10px; font-weight: 700;
    letter-spacing: .10em; text-transform: uppercase;
    color: #555; margin-bottom: 12px;
}

/* ── VERDICT PANEL ── */
.verdict-score {
    font-size: 68px; font-weight: 700;
    line-height: 1; letter-spacing: -3px;
    margin-bottom: 2px;
}
.verdict-sub { font-size: 12px; color: #555; margin-bottom: 14px; }
.pill {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 4px 12px; border-radius: 20px;
    font-size: 11px; font-weight: 600; margin-bottom: 16px;
}
.pill-fake   { background: rgba(226,75,74,0.15);  color: #F09595; }
.pill-real   { background: rgba(29,158,117,0.15); color: #5DCAA5; }
.pill-unsure { background: rgba(239,159,39,0.15); color: #FAC775; }

.meter-row { margin-top: 8px; }
.meter-hd {
    display: flex; justify-content: space-between;
    font-size: 10px; color: #444; margin-bottom: 4px;
}
.meter-track { height: 2px; background: #222230; border-radius: 2px; margin-bottom: 3px; }
.meter-fill  { height: 100%; border-radius: 2px; }
.meter-vals {
    display: flex; justify-content: space-between;
    font-size: 12px; font-weight: 600;
}

/* ── FORENSIC SIGNALS GRID ── */
.sig-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0;
}
.sig-cell {
    padding: 10px 14px;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    border-right: 1px solid rgba(255,255,255,0.04);
}
.sig-cell:nth-child(2n) { border-right: none; }
.sig-cell:nth-last-child(-n+2) { border-bottom: none; }
.sig-row {
    display: flex; justify-content: space-between;
    align-items: baseline; margin-bottom: 3px;
}
.sig-name { font-size: 13px; font-weight: 600; color: #D8D8E4; }
.sig-pct  { font-size: 13px; font-weight: 700; }
.sig-desc { font-size: 11px; color: #555; margin-bottom: 6px; }
.sig-bar-track { height: 2px; background: #1E1E2A; border-radius: 2px; }
.sig-bar-fill  { height: 100%; border-radius: 2px; }

/* ── DIVIDER ── */
.hdivider { height: 1px; background: rgba(255,255,255,0.05); }
.vdivider { width: 1px; background: rgba(255,255,255,0.05); }

/* ── DETECTION EVENTS ── */
.ev-item {
    display: flex; gap: 10px;
    padding: 10px 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
}
.ev-item:last-child { border-bottom: none; padding-bottom: 0; }
.ev-dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; margin-top: 4px; }
.ev-title { font-size: 12px; font-weight: 600; color: #D0D0DC; line-height: 1.3; }
.ev-sub   { font-size: 11px; color: #555; margin-top: 1px; }

/* ── CONFIDENCE BREAKDOWN ── */
.conf-item { margin-bottom: 12px; }
.conf-item:last-child { margin-bottom: 0; }
.conf-row {
    display: flex; justify-content: space-between;
    font-size: 13px; margin-bottom: 4px;
}
.conf-name { color: #C0C0CC; font-weight: 500; }
.conf-pct  { font-weight: 700; }
.conf-track { height: 2px; background: #1E1E2A; border-radius: 2px; }
.conf-fill  { height: 100%; border-radius: 2px; }

/* ── FOOTER STRIP ── */
.foot-strip {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr 1fr;
    border-top: 1px solid rgba(255,255,255,0.06);
    padding: 14px 20px;
}
.foot-cell { padding: 0 18px; border-left: 1px solid rgba(255,255,255,0.05); }
.foot-cell:first-child { padding-left: 0; border-left: none; }
.foot-key {
    font-size: 10px; font-weight: 700; letter-spacing:.08em;
    text-transform: uppercase; color: #444; margin-bottom: 3px;
}
.foot-val { font-size: 13px; font-weight: 600; color: #C0C0CC; }
.foot-sub { font-size: 11px; color: #555; }
</style>
""", unsafe_allow_html=True)

# ── Data ─────────────────────────────────────────────────────────────────────
CLIPS = {
    "VID-9042": {
        "fake": 97.3, "real": 2.7, "verdict": "fake",
        "signals": [
            ("Lip desync",     91, "#E24B4A", "4.84ms lag detected"),
            ("MFCC variance",  87, "#E24B4A", "Unnaturally smooth speech"),
            ("Face geometry",  72, "#EF9F27", "Drift at frames 47–62"),
            ("Audio splice",   68, "#EF9F27", "Cut at 0:03–0:07"),
            ("Noise print",    44, "#EF9F27", "Environment mismatch"),
            ("Transcript",     31, "#639922", "99.2% match — clean"),
        ],
        "confidence": [
            ("Audio",      79, "#E24B4A"),
            ("Visual",     81, "#E24B4A"),
            ("Sync",       91, "#E24B4A"),
            ("Transcript", 22, "#639922"),
        ],
        "events": [
            ("Audio splice detected",  "Timestamp 0:03–0:07 · 98.1% confidence", "#E24B4A"),
            ("Lip desync flagged",     "4.84ms average lag · 23 frames affected", "#EF9F27"),
            ("MFCC anomaly",           "Frames 47–62 · 6 coefficient bands",      "#EF9F27"),
            ("Transcript parsed",      "Whisper large-v3 · 99.2% accuracy",       "#639922"),
        ],
    },
    "VID-9041": {
        "fake": 4.2, "real": 95.8, "verdict": "real",
        "signals": [
            ("Lip desync",    9,  "#1D9E75", "< 10ms lag — nominal"),
            ("MFCC variance", 12, "#1D9E75", "Normal speech pattern"),
            ("Face geometry", 8,  "#1D9E75", "No drift detected"),
            ("Audio splice",  6,  "#1D9E75", "Clean waveform"),
            ("Noise print",   11, "#1D9E75", "Consistent environment"),
            ("Transcript",    5,  "#1D9E75", "High coherence"),
        ],
        "confidence": [
            ("Audio",      8,  "#1D9E75"),
            ("Visual",     6,  "#1D9E75"),
            ("Sync",       9,  "#1D9E75"),
            ("Transcript", 4,  "#1D9E75"),
        ],
        "events": [
            ("No splice detected",  "Clean waveform throughout",     "#639922"),
            ("Lip sync nominal",    "< 10ms lag across all frames",  "#639922"),
            ("MFCC normal",         "All frames within normal range","#639922"),
            ("Transcript parsed",   "High coherence score",          "#639922"),
        ],
    },
    "VID-9040": {
        "fake": 62.1, "real": 37.9, "verdict": "unsure",
        "signals": [
            ("Lip desync",    48, "#EF9F27", "32ms avg — suspicious"),
            ("MFCC variance", 61, "#EF9F27", "Elevated anomaly score"),
            ("Face geometry", 38, "#EF9F27", "Minor drift detected"),
            ("Audio splice",  55, "#EF9F27", "Possible cut at 0:11"),
            ("Noise print",   57, "#EF9F27", "Slight mismatch"),
            ("Transcript",    42, "#EF9F27", "Minor gaps noted"),
        ],
        "confidence": [
            ("Audio",      55, "#EF9F27"),
            ("Visual",     48, "#EF9F27"),
            ("Sync",       61, "#EF9F27"),
            ("Transcript", 38, "#EF9F27"),
        ],
        "events": [
            ("Inconclusive audio",  "Mixed signals — further review needed", "#EF9F27"),
            ("Lip lag detected",    "32ms avg — borderline threshold",       "#EF9F27"),
            ("MFCC elevated",       "Suspicious coefficient range",          "#EF9F27"),
            ("Transcript gaps",     "Minor anomalies detected",              "#EF9F27"),
        ],
    },
}

COLOR_MAP = {
    "fake":  {"score": "#E24B4A", "pill": "pill-fake",   "label": "Fake — high confidence"},
    "real":  {"score": "#1D9E75", "pill": "pill-real",   "label": "Authentic — all clear"},
    "unsure":{"score": "#EF9F27", "pill": "pill-unsure", "label": "Probable fake — review needed"},
}

# ── Header row: brand left, clip selector right ───────────────────────────────
st.markdown('<div class="shell">', unsafe_allow_html=True)

# Top bar with brand + clip tabs
col_brand, col_tabs, col_status = st.columns([2.2, 3, 1.2], gap="small")

with col_brand:
    st.markdown("""
    <div class="topbar" style="border-bottom:none;padding-right:0">
      <div class="brand">
        <div class="shield-icon">🛡️</div>
        <div>
          <div class="brand-name">Deepfake Detector</div>
          <div class="brand-sub">Multimodal forensic analysis · v2.4.1</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

with col_tabs:
    st.markdown('<div style="padding-top:16px">', unsafe_allow_html=True)
    clip_id = st.radio("", list(CLIPS.keys()), horizontal=True, label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

with col_status:
    st.markdown("""
    <div style="padding-top:18px;text-align:right">
      <span class="sys-online"><span class="live-dot"></span> System online</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="hdivider"></div>', unsafe_allow_html=True)

clip = CLIPS[clip_id]
cm   = COLOR_MAP[clip["verdict"]]

# ── Row 1: Verdict | Forensic Signals ────────────────────────────────────────
c_verdict, c_divv, c_signals = st.columns([1.1, 0.02, 2.4], gap="small")

with c_verdict:
    sigs_html = "".join([
        f'<div class="sig-cell">'
        f'<div class="sig-row"><span class="sig-name">{name}</span>'
        f'<span class="sig-pct" style="color:{col}">{val}%</span></div>'
        f'<div class="sig-desc">{desc}</div>'
        f'<div class="sig-bar-track"><div class="sig-bar-fill" style="width:{val}%;background:{col}"></div></div>'
        f'</div>'
        for name, val, col, desc in clip["signals"]
    ])
    st.markdown(f"""
    <div style="padding:18px 20px 12px">
      <div class="sec-label">Verdict</div>
      <div class="verdict-score" style="color:{cm['score']}">{clip['fake']:.1f}%</div>
      <div class="verdict-sub">Fake probability</div>
      <span class="pill {cm['pill']}">● &nbsp;{cm['label']}</span>
      <div class="meter-row">
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

with c_divv:
    st.markdown('<div class="vdivider" style="height:100%;min-height:220px"></div>', unsafe_allow_html=True)

with c_signals:
    sigs_html = "".join([
        f'<div class="sig-cell">'
        f'<div class="sig-row"><span class="sig-name">{name}</span>'
        f'<span class="sig-pct" style="color:{col}">{val}%</span></div>'
        f'<div class="sig-desc">{desc}</div>'
        f'<div class="sig-bar-track"><div class="sig-bar-fill" style="width:{val}%;background:{col}"></div></div>'
        f'</div>'
        for name, val, col, desc in clip["signals"]
    ])
    st.markdown(f"""
    <div style="padding:18px 14px 12px 14px">
      <div class="sec-label">Forensic Signals</div>
      <div class="sig-grid">{sigs_html}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="hdivider"></div>', unsafe_allow_html=True)

# ── Row 2: Detection Events | Confidence Breakdown ───────────────────────────
c_events, c_divh, c_conf = st.columns([1.6, 0.02, 1], gap="small")

with c_events:
    evts_html = "".join([
        f'<div class="ev-item">'
        f'<div class="ev-dot" style="background:{dot}"></div>'
        f'<div><div class="ev-title">{msg}</div>'
        f'<div class="ev-sub">{sub}</div></div></div>'
        for msg, sub, dot in clip["events"]
    ])
    st.markdown(f"""
    <div style="padding:16px 20px">
      <div class="sec-label">Detection Events</div>
      {evts_html}
    </div>
    """, unsafe_allow_html=True)

with c_divh:
    st.markdown('<div class="vdivider" style="height:100%;min-height:180px"></div>', unsafe_allow_html=True)

with c_conf:
    conf_html = "".join([
        f'<div class="conf-item">'
        f'<div class="conf-row"><span class="conf-name">{name}</span>'
        f'<span class="conf-pct" style="color:{col}">{val}%</span></div>'
        f'<div class="conf-track"><div class="conf-fill" style="width:{val}%;background:{col}"></div></div>'
        f'</div>'
        for name, val, col in clip["confidence"]
    ])
    st.markdown(f"""
    <div style="padding:16px 18px">
      <div class="sec-label">Confidence Breakdown</div>
      {conf_html}
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="hdivider"></div>', unsafe_allow_html=True)

# ── Row 3: Action buttons ─────────────────────────────────────────────────────
st.markdown('<div style="padding:12px 20px 14px;display:flex;gap:8px">', unsafe_allow_html=True)
b1, b2, b3, _ = st.columns([1.2, 1.2, 1, 2], gap="small")
with b1:
    st.button("▪ Evidence report ↗", key="r")
with b2:
    st.button("◎ Explain signals ↗", key="e")
with b3:
    st.button("⚖ Compare clips ↗", key="c")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="hdivider"></div>', unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="foot-strip">
  <div class="foot-cell">
    <div class="foot-key">Model</div>
    <div class="foot-val">v2.4.1</div>
  </div>
  <div class="foot-cell">
    <div class="foot-key">Audio Engine</div>
    <div class="foot-val">Whisper large-v3</div>
  </div>
  <div class="foot-cell">
    <div class="foot-key">Frames Scanned</div>
    <div class="foot-val">47</div>
    <div class="foot-sub">at 4K resolution</div>
  </div>
  <div class="foot-cell">
    <div class="foot-key">Processing Time</div>
    <div class="foot-val">1.84s</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # close .shell
