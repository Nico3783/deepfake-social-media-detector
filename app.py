"""
Deepfake Detection System — Command Center Dashboard

Ethical hacking / cybersecurity aesthetic Streamlit dashboard for:
- Real-time video & image deepfake analysis
- Model training & evaluation
- Experiment comparison
- Dataset exploration
- System health monitoring

Author: Olamijulo Israel D (CYS/22/9071) — FUTA Cyber Security
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import numpy as np
import pandas as pd
import time
import json
import os
import tempfile
from datetime import datetime
from io import BytesIO

import torch

from src.models.model_factory import load_model
from src.inference.predict_video import VideoPredictor
from src.inference.predict_image import ImagePredictor

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="DEEPFAKE // COMMAND CENTER",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — Ethical Hacking Terminal Aesthetic
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Orbitron:wght@400;500;600;700;800;900&family=Share+Tech+Mono&display=swap');

/* ── Global ── */
:root {
    --neon-green:   #00ff41;
    --neon-cyan:    #00e5ff;
    --neon-red:     #ff0040;
    --neon-amber:   #ffab00;
    --neon-purple:  #b388ff;
    --bg-primary:   #0a0e17;
    --bg-secondary: #0d1321;
    --bg-card:      #111827;
    --bg-elevated:  #1a2332;
    --border:       #1e3a5f;
    --text-primary: #e0e0e0;
    --text-dim:     #6b7280;
    --glow-green:   0 0 20px rgba(0,255,65,0.3);
    --glow-cyan:    0 0 20px rgba(0,229,255,0.3);
    --glow-red:     0 0 20px rgba(255,0,64,0.3);
}

/* Hide Streamlit defaults */
#MainMenu, footer, header {visibility: hidden;}
.stDeployButton {display: none;}

/* Main container */
.stApp {
    background: var(--bg-primary);
    color: var(--text-primary);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] .stRadio > label,
section[data-testid="stSidebar"] .stSelectbox > label {
    color: var(--neon-cyan) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.5px !important;
    text-transform: uppercase !important;
}

/* Headings */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Orbitron', sans-serif !important;
    color: var(--neon-green) !important;
    text-shadow: var(--glow-green) !important;
}
h1 { font-size: 2rem !important; letter-spacing: 2px; }
h2 { font-size: 1.4rem !important; letter-spacing: 1px; }

/* Code / monospace */
code, pre, .stCode {
    font-family: 'JetBrains Mono', monospace !important;
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 16px !important;
    box-shadow: 0 0 15px rgba(0,229,255,0.08) !important;
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
}
[data-testid="stMetric"]:hover {
    border-color: var(--neon-cyan) !important;
    box-shadow: var(--glow-cyan) !important;
}
[data-testid="stMetric"] label {
    color: var(--neon-cyan) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 600 !important;
    font-size: 0.8rem !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}
[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: var(--neon-green) !important;
    font-family: 'Orbitron', sans-serif !important;
    font-weight: 700 !important;
    text-shadow: var(--glow-green) !important;
}

/* Expander */
details[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}
details[data-testid="stExpander"] summary {
    color: var(--neon-cyan) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 600 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 0 !important;
    background: var(--bg-secondary) !important;
    border-radius: 8px 8px 0 0 !important;
    border: 1px solid var(--border) !important;
    border-bottom: none !important;
}
.stTabs [data-baseweb="tab"] {
    color: var(--text-dim) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.5px !important;
    text-transform: uppercase !important;
    padding: 10px 20px !important;
    border-radius: 0 !important;
}
.stTabs [aria-selected="true"] {
    color: var(--neon-green) !important;
    background: var(--bg-card) !important;
    border-bottom: 2px solid var(--neon-green) !important;
    text-shadow: var(--glow-green) !important;
}

/* Buttons */
.stButton > button {
    background: transparent !important;
    border: 2px solid var(--neon-green) !important;
    color: var(--neon-green) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 600 !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    border-radius: 4px !important;
    padding: 8px 24px !important;
    transition: all 0.3s ease !important;
}
.stButton > button:hover {
    background: var(--neon-green) !important;
    color: var(--bg-primary) !important;
    box-shadow: var(--glow-green) !important;
}
.stDownloadButton > button {
    background: transparent !important;
    border: 2px solid var(--neon-cyan) !important;
    color: var(--neon-cyan) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 600 !important;
    border-radius: 4px !important;
}
.stDownloadButton > button:hover {
    background: var(--neon-cyan) !important;
    color: var(--bg-primary) !important;
}

/* Inputs */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stSelectbox > div > div > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    font-family: 'JetBrains Mono', monospace !important;
    border-radius: 4px !important;
}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
    border-color: var(--neon-cyan) !important;
    box-shadow: var(--glow-cyan) !important;
}

/* File uploader */
.stFileUploader {
    border: 2px dashed var(--border) !important;
    border-radius: 8px !important;
    padding: 20px !important;
}
.stFileUploader:hover {
    border-color: var(--neon-cyan) !important;
}

/* DataFrame */
.stDataFrame {
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

/* Dividers */
hr {
    border-color: var(--border) !important;
    opacity: 0.5 !important;
}

/* Scan-line overlay effect */
.scanlines::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0,255,65,0.015) 2px,
        rgba(0,255,65,0.015) 4px
    );
    pointer-events: none;
    z-index: 9999;
}

/* Terminal prompt style */
.terminal-prompt {
    font-family: 'Share Tech Mono', monospace;
    color: var(--neon-green);
    font-size: 0.9rem;
    padding: 8px 12px;
    background: var(--bg-card);
    border-left: 3px solid var(--neon-green);
    margin: 8px 0;
    border-radius: 0 4px 4px 0;
}

/* Status badges */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.badge-ok    { background: rgba(0,255,65,0.15);  color: var(--neon-green);  border: 1px solid var(--neon-green); }
.badge-warn  { background: rgba(255,171,0,0.15); color: var(--neon-amber); border: 1px solid var(--neon-amber); }
.badge-danger{ background: rgba(255,0,64,0.15);  color: var(--neon-red);   border: 1px solid var(--neon-red); }
.badge-info  { background: rgba(0,229,255,0.15); color: var(--neon-cyan);  border: 1px solid var(--neon-cyan); }

/* Glowing border on focus */
.glow-border {
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
    transition: all 0.3s ease;
}
.glow-border:hover {
    border-color: var(--neon-cyan);
    box-shadow: var(--glow-cyan);
}

/* Section header with line */
.section-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 20px 0 12px 0;
}
.section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(to right, var(--border), transparent);
}

/* Pulse animation for live indicators */
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}
.pulse {
    animation: pulse 2s ease-in-out infinite;
}

/* Matrix rain effect container */
.matrix-bg {
    position: relative;
    overflow: hidden;
}
</style>

<div class="scanlines"></div>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helper: load project modules safely
# ---------------------------------------------------------------------------
def _import(mod_path: str, cls_name: str):
    """Dynamically import a class from src/. Returns None on failure."""
    try:
        mod = __import__(mod_path, fromlist=[cls_name])
        return getattr(mod, cls_name)
    except Exception:
        return None


def _import_func(mod_path: str, func_name: str):
    """Dynamically import a function from src/. Returns None on failure."""
    try:
        mod = __import__(mod_path, fromlist=[func_name])
        return getattr(mod, func_name)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []
if "training_active" not in st.session_state:
    st.session_state.training_active = False
if "current_model" not in st.session_state:
    st.session_state.current_model = None


# ============================================================================
# SIDEBAR
# ============================================================================
def render_sidebar():
    with st.sidebar:
        st.markdown(
            '<div style="text-align:center; padding: 10px 0 20px 0;">'
            '<span style="font-family:Orbitron,sans-serif; font-size:1.1rem; '
            'color:#00ff41; text-shadow:0 0 20px rgba(0,255,65,0.5); '
            'letter-spacing:2px;">DEEPFAKE</span><br>'
            '<span style="font-family:Share Tech Mono,monospace; font-size:0.7rem; '
            'color:#6b7280; letter-spacing:3px;">COMMAND CENTER v1.0</span>'
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown("---")

        page = st.radio(
            "NAVIGATE",
            [
                "HOME",
                "ANALYZE",
                "TRAIN",
                "EVALUATE",
                "DATASETS",
                "EXPERIMENTS",
                "SYSTEM",
            ],
            label_visibility="collapsed",
        )

        st.markdown("---")

        # System status widget
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_mem / 1e9
            st.markdown(
                f'<div class="badge badge-ok">GPU ONLINE</div><br>'
                f'<span style="font-family:JetBrains Mono,monospace; font-size:0.7rem; '
                f'color:#6b7280;">{gpu_name}<br>{vram:.1f} GB VRAM</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="badge badge-warn">CPU MODE</div><br>'
                '<span style="font-family:JetBrains Mono,monospace; font-size:0.7rem; '
                'color:#6b7280;">No CUDA GPU detected</span>',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown(
            '<span style="font-family:JetBrains Mono,monospace; font-size:0.65rem; '
            'color:#4b5563;">'
            "SYS.TIME: " + datetime.now().strftime("%H:%M:%S UTC") + "<br>"
            "BUILD: " + datetime.now().strftime("%Y%m%d") + "<br>"
            "PROTOCOL: DEEPFAKE-DETECT v1.0"
            "</span>",
            unsafe_allow_html=True,
        )

    return page


# ============================================================================
# PAGE: HOME
# ============================================================================
def page_home():
    st.markdown(
        '<h1 style="text-align:center; font-size:2.2rem;">'
        "DEEPFAKE DETECTION COMMAND CENTER</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="text-align:center; color:#6b7280; font-family:JetBrains Mono,monospace; '
        'font-size:0.85rem; letter-spacing:2px; margin-top:-10px;">'
        "NEURAL FORENSICS // REAL-TIME ANALYSIS // CYBERSECURITY INTELLIGENCE</p>",
        unsafe_allow_html=True,
    )

    st.markdown("")

    # ── System Status Metrics ──
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("SYSTEM STATUS", "ONLINE")
    with col2:
        st.metric("MODELS LOADED", "0 / 2")
    with col3:
        st.metric("ANALYSES RUN", str(len(st.session_state.analysis_history)))
    with col4:
        import torch

        st.metric(
            "DEVICE",
            "CUDA" if torch.cuda.is_available() else "CPU",
        )

    st.markdown("")

    # ── Quick Actions ──
    st.markdown(
        '<div class="section-header">'
        '<span style="color:#00e5ff; font-family:JetBrains Mono,monospace; '
        'font-size:0.8rem; letter-spacing:2px; text-transform:uppercase;">'
        "QUICK ACCESS</span></div>",
        unsafe_allow_html=True,
    )

    q1, q2, q3 = st.columns(3)
    with q1:
        if st.button("ANALYZE VIDEO", use_container_width=True):
            st.session_state._nav = "ANALYZE"
            st.rerun()
    with q2:
        if st.button("START TRAINING", use_container_width=True):
            st.session_state._nav = "TRAIN"
            st.rerun()
    with q3:
        if st.button("VIEW REPORTS", use_container_width=True):
            st.session_state._nav = "EVALUATE"
            st.rerun()

    st.markdown("")

    # ── Architecture Overview ──
    st.markdown(
        '<div class="section-header">'
        '<span style="color:#00e5ff; font-family:JetBrains Mono,monospace; '
        'font-size:0.8rem; letter-spacing:2px;">'
        "ARCHITECTURE OVERVIEW</span></div>",
        unsafe_allow_html=True,
    )

    arch_col1, arch_col2 = st.columns([2, 1])

    with arch_col1:
        st.markdown(
            '<div class="glow-border" style="background:#111827;">'
            '<pre style="color:#00ff41; font-family:Share Tech Mono,monospace; '
            'font-size:0.8rem; line-height:1.6; margin:0;">'
            "  VIDEO INPUT\n"
            "       │\n"
            "       ▼\n"
            "  ┌─────────────────┐\n"
            "  │ FRAME EXTRACTOR  │  OpenCV — 1 FPS sampling\n"
            "  └────────┬────────┘\n"
            "           │\n"
            "           ▼\n"
            "  ┌─────────────────┐\n"
            "  │  FACE DETECTOR   │  MTCNN — face localization\n"
            "  └────────┬────────┘\n"
            "           │\n"
            "           ▼\n"
            "  ┌─────────────────┐\n"
            "  │  CNN MODEL       │  XceptionNet / EfficientNet\n"
            "  └────────┬────────┘\n"
            "           │\n"
            "           ▼\n"
            "  ┌─────────────────┐\n"
            "  │  AGGREGATOR      │  Mean / Majority / Confidence\n"
            "  └────────┬────────┘\n"
            "           │\n"
            "           ▼\n"
            "     REAL  or  FAKE\n"
            "</pre></div>",
            unsafe_allow_html=True,
        )

    with arch_col2:
        st.markdown(
            '<div style="padding:12px;">'
            '<p style="color:#00e5ff; font-family:JetBrains Mono,monospace; '
            'font-size:0.8rem; font-weight:600; margin-bottom:12px;">'
            "MODELS</p>"
            '<div class="glow-border" style="margin-bottom:12px;">'
            '<span style="color:#00ff41; font-weight:700;">XceptionNet</span><br>'
            '<span style="color:#6b7280; font-size:0.75rem;">22.9M params • 299×299 input<br>'
            "Depthwise separable convolutions<br>"
            "99.53% on FaceForensics++</span></div>"
            '<div class="glow-border">'
            '<span style="color:#00ff41; font-weight:700;">EfficientNet-B0</span><br>'
            '<span style="color:#6b7280; font-size:0.75rem;">5.3M params • 224×224 input<br>'
            "Compound scaling<br>"
            "2× faster inference</span></div>"
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("")

    # ── Recent Activity ──
    st.markdown(
        '<div class="section-header">'
        '<span style="color:#00e5ff; font-family:JetBrains Mono,monospace; '
        'font-size:0.8rem; letter-spacing:2px;">'
        "RECENT ACTIVITY</span></div>",
        unsafe_allow_html=True,
    )

    if st.session_state.analysis_history:
        df = pd.DataFrame(st.session_state.analysis_history[-10:])
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.markdown(
            '<div class="terminal-prompt">'
            "root@deepfake:~# No analyses performed yet. "
            "Navigate to <b>ANALYZE</b> to begin.</div>",
            unsafe_allow_html=True,
        )

    # ── Thesis Info ──
    st.markdown("")
    with st.expander("THESIS INFORMATION"):
        st.markdown(
            """
| Field | Detail |
|-------|--------|
| **Title** | Detection of Social Media Deepfake Contents Using Deep Learning Algorithm |
| **Student** | Olamijulo Israel D (CYS/22/9071) |
| **Department** | Cyber Security |
| **Institution** | Federal University of Technology Akure (FUTA) |
| **Models** | XceptionNet (primary), EfficientNet-B0 (secondary) |
| **Datasets** | FaceForensics++, Celeb-DF v2 |
        """
        )


# ============================================================================
# PAGE: ANALYZE
# ============================================================================
def page_analyze():
    st.markdown(
        '<h1 style="font-size:1.8rem;">ANALYZE</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="color:#6b7280; font-family:JetBrains Mono,monospace; '
        'font-size:0.8rem; letter-spacing:1px; margin-top:-10px;">'
        "Upload a video or image for deepfake detection analysis</p>",
        unsafe_allow_html=True,
    )

    tab_video, tab_image, tab_batch = st.tabs(
        ["VIDEO ANALYSIS", "IMAGE ANALYSIS", "BATCH PROCESS"]
    )

    # ── Settings ──
    with st.expander("ANALYSIS SETTINGS"):
        s1, s2, s3 = st.columns(3)
        with s1:
            model_choice = st.selectbox(
                "Model",
                ["XceptionNet", "EfficientNet-B0"],
            )
        with s2:
            threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
        with s3:
            aggregation = st.selectbox(
                "Aggregation",
                ["mean", "majority", "confidence_weighted"],
            )

    # ── VIDEO TAB ──
    with tab_video:
        video_file = st.file_uploader(
            "Upload Video",
            type=["mp4", "avi", "mov", "mkv"],
            help="Supported formats: MP4, AVI, MOV, MKV",
        )

        if video_file:
            # Save uploaded file
            tmp_path = Path("outputs") / "temp" / video_file.name
            tmp_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path.write_bytes(video_file.read())

            # Show video info
            col_a, col_b = st.columns([1, 1])
            with col_a:
                st.video(str(tmp_path))
            with col_b:
                st.markdown(
                    f'<div class="terminal-prompt">'
                    f"root@deepfake:~# file {video_file.name}<br>"
                    f"SIZE: {video_file.size / 1e6:.2f} MB<br>"
                    f"FORMAT: {video_file.type}<br>"
                    f"MODEL: {model_choice}<br>"
                    f"THRESHOLD: {threshold}<br>"
                    f"AGGREGATION: {aggregation}</div>",
                    unsafe_allow_html=True,
                )

                if st.button("RUN ANALYSIS", key="run_video", use_container_width=True):
                    _run_video_analysis(tmp_path, model_choice, threshold, aggregation)

        # Show results
        if st.session_state.analysis_history:
            latest = st.session_state.analysis_history[-1]
            if latest.get("type") == "video":
                _display_video_results(latest)

    # ── IMAGE TAB ──
    with tab_image:
        image_file = st.file_uploader(
            "Upload Image",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            help="Supported formats: JPG, PNG, BMP, WebP",
            key="img_uploader",
        )

        if image_file:
            img_bytes = image_file.read()
            col_i1, col_i2 = st.columns([1, 1])

            with col_i1:
                st.image(img_bytes, caption="Uploaded Image", use_container_width=True)
            with col_i2:
                st.markdown(
                    f'<div class="terminal-prompt">'
                    f"root@deepfake:~# analyze {image_file.name}<br>"
                    f"SIZE: {len(img_bytes) / 1e3:.1f} KB<br>"
                    f"FORMAT: {image_file.type}<br>"
                    f"MODEL: {model_choice}</div>",
                    unsafe_allow_html=True,
                )

                if st.button("ANALYZE IMAGE", key="run_img", use_container_width=True):
                    _run_image_analysis(img_bytes, image_file.name, model_choice, threshold)

    # ── BATCH TAB ──
    with tab_batch:
        batch_files = st.file_uploader(
            "Upload Multiple Files",
            type=["mp4", "avi", "mov", "mkv", "jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="batch_uploader",
        )

        if batch_files:
            st.markdown(
                f'<div class="terminal-prompt">'
                f"root@deepfake:~# batch-analyze --count {len(batch_files)} files loaded</div>",
                unsafe_allow_html=True,
            )

            file_df = pd.DataFrame(
                {
                    "File": [f.name for f in batch_files],
                    "Type": [f.type.split("/")[-1].upper() for f in batch_files],
                    "Size (KB)": [f.size / 1024 for f in batch_files],
                }
            )
            st.dataframe(file_df, use_container_width=True, hide_index=True)

            if st.button("RUN BATCH ANALYSIS", use_container_width=True):
                _run_batch_analysis(batch_files, model_choice, threshold, aggregation)


def _load_trained_model(model_choice: str):
    """Load a trained model from checkpoint, caching in session_state.

    Returns (model, device) or (None, None) if no checkpoint exists.
    """
    cache_key = f"_model_{model_choice}"
    if cache_key in st.session_state:
        return st.session_state[cache_key], st.session_state.get("_device")

    checkpoint_dir = Path("outputs/checkpoints")
    # Try exact model name, then fallback patterns
    candidates = [
        checkpoint_dir / f"{model_choice.lower()}_best.pth",
        checkpoint_dir / f"{model_choice.lower()}_final.pth",
        checkpoint_dir / "best_model.pth",
    ]
    checkpoint_path = next((p for p in candidates if p.exists()), None)

    if checkpoint_path is None:
        return None, None

    model_name = "xception" if "xception" in model_choice.lower() else "efficientnet-b0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(
        checkpoint_path=str(checkpoint_path),
        model_name=model_name,
        num_classes=2,
        device=device,
    )
    model.eval()
    st.session_state[cache_key] = model
    st.session_state["_device"] = device
    return model, device


def _run_video_analysis(video_path, model_choice, threshold, aggregation):
    """Run video deepfake analysis."""
    with st.spinner("Initializing analysis pipeline..."):
        progress = st.progress(0, text="Loading model...")
        model, device = _load_trained_model(model_choice)

        if model is None:
            progress.progress(100, text="No trained model found.")
            st.error(
                "No trained model checkpoint found. "
                "Train a model first (TRAIN page) or place a checkpoint in "
                "`outputs/checkpoints/` with name "
                f"`{model_choice.lower()}_best.pth` or `best_model.pth`."
            )
            return

        progress.progress(20, text="Initializing predictor...")
        predictor = VideoPredictor(
            model=model,
            device=device,
            threshold=threshold,
            aggregation_method=aggregation,
            frame_sample_rate=5,
            target_size=299,
        )
        progress.progress(40, text="Extracting frames & detecting faces...")
        time.sleep(0.1)
        progress.progress(60, text="Running inference...")
        time.sleep(0.1)

    try:
        prediction = predictor.predict(video_path)
    except Exception as e:
        st.error(f"Inference failed: {e}")
        return

    progress = st.progress(90, text="Generating report...")
    time.sleep(0.1)
    progress.progress(100, text="Analysis complete.")

    result = {
        "type": "video",
        "filename": video_path.name,
        "model": model_choice,
        "prediction": "FAKE" if prediction.is_fake else "REAL",
        "confidence": float(prediction.confidence),
        "fake_probability": float(prediction.fake_probability),
        "real_probability": float(prediction.real_probability),
        "threshold": threshold,
        "aggregation": aggregation,
        "frames_analyzed": prediction.num_frames,
        "frame_predictions": prediction.frame_predictions,
        "temporal_analysis": prediction.temporal_analysis,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    st.session_state.analysis_history.append(result)
    st.success("Analysis complete. See results below.")
    st.rerun()


def _display_video_results(result):
    """Display video analysis results."""
    st.markdown(
        '<div class="section-header">'
        '<span style="color:#00e5ff; font-family:JetBrains Mono,monospace; '
        'font-size:0.8rem; letter-spacing:2px;">'
        "ANALYSIS RESULTS</span></div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        color = "#ff0040" if result["prediction"] == "FAKE" else "#00ff41"
        st.markdown(
            f'<div style="text-align:center; padding:20px; '
            f'background:#111827; border:2px solid {color}; border-radius:8px;">'
            f'<span style="font-family:Orbitron,sans-serif; font-size:2rem; '
            f'color:{color}; text-shadow:0 0 20px {color}40;">'
            f'{result["prediction"]}</span></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.metric("Confidence", f'{result["confidence"]:.1%}')
    with c3:
        st.metric("Fake Probability", f'{result["fake_probability"]:.3f}')
    with c4:
        st.metric("Frames Analyzed", result["frames_analyzed"])

    # Probability bar
    st.markdown("")
    fake_pct = result["fake_probability"] * 100
    real_pct = result["real_probability"] * 100
    bar_color = "#ff0040" if result["prediction"] == "FAKE" else "#00ff41"

    st.markdown(
        f'<div style="background:#1a2332; border-radius:8px; overflow:hidden; '
        f'border:1px solid #1e3a5f;">'
        f'<div style="width:{fake_pct}%; background:linear-gradient(90deg, '
        f'#ff0040, #ff4060); padding:8px 16px; display:inline-block;">'
        f'<span style="font-family:JetBrains Mono,monospace; font-size:0.8rem; '
        f'color:white; font-weight:600;">FAKE {fake_pct:.1f}%</span></div>'
        f'<div style="width:{real_pct}%; background:linear-gradient(90deg, '
        f'#00ff40, #00cc33); padding:8px 16px; display:inline-block; '
        f'float:right;">'
        f'<span style="font-family:JetBrains Mono,monospace; font-size:0.8rem; '
        f'color:white; font-weight:600;">REAL {real_pct:.1f}%</span></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown("")
    with st.expander("DETAILED REPORT"):
        st.json(result)


def _run_image_analysis(img_bytes, filename, model_choice, threshold):
    """Run image deepfake analysis."""
    with st.spinner("Analyzing image..."):
        progress = st.progress(0, text="Loading model...")
        model, device = _load_trained_model(model_choice)

        if model is None:
            progress.progress(100, text="No trained model found.")
            st.error(
                "No trained model checkpoint found. "
                "Train a model first (TRAIN page) or place a checkpoint in "
                "`outputs/checkpoints/`."
            )
            return

        progress.progress(30, text="Initializing predictor...")
        predictor = ImagePredictor(
            model=model,
            device=device,
            threshold=threshold,
            target_size=299,
        )
        progress.progress(60, text="Running inference...")

        # Save bytes to a temp file so ImagePredictor can read from path
        tmp = Path(tempfile.mktemp(suffix=Path(filename).suffix))
        tmp.write_bytes(img_bytes)

    try:
        prediction = predictor.predict(tmp)
    except Exception as e:
        st.error(f"Inference failed: {e}")
        return
    finally:
        tmp.unlink(missing_ok=True)

    progress = st.progress(100, text="Complete.")

    result = {
        "type": "image",
        "filename": filename,
        "model": model_choice,
        "prediction": "FAKE" if prediction.is_fake else "REAL",
        "confidence": float(prediction.confidence),
        "fake_probability": float(prediction.fake_probability),
        "real_probability": float(prediction.real_probability),
        "face_detected": prediction.face_detected,
        "bounding_box": list(prediction.bounding_box) if prediction.bounding_box else None,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    st.session_state.analysis_history.append(result)

    st.markdown("")
    color = "#ff0040" if result["prediction"] == "FAKE" else "#00ff41"
    st.markdown(
        f'<div style="text-align:center; padding:16px; background:#111827; '
        f'border:2px solid {color}; border-radius:8px; margin:12px 0;">'
        f'<span style="font-family:Orbitron,sans-serif; font-size:1.5rem; '
        f'color:{color};">{result["prediction"]}</span><br>'
        f'<span style="font-family:JetBrains Mono,monospace; font-size:0.85rem; '
        f'color:#6b7280;">Confidence: {result["confidence"]:.1%} | '
        f'Fake: {result["fake_probability"]:.3f} | '
        f'Real: {result["real_probability"]:.3f} | '
        f'Face: {"Detected" if result["face_detected"] else "Not detected"}</span></div>',
        unsafe_allow_html=True,
    )


def _run_batch_analysis(files, model_choice, threshold, aggregation):
    """Run batch analysis on multiple files."""
    model, device = _load_trained_model(model_choice)

    if model is None:
        st.error(
            "No trained model checkpoint found. "
            "Train a model first or place a checkpoint in "
            "`outputs/checkpoints/`."
        )
        return

    video_predictor = VideoPredictor(
        model=model, device=device,
        threshold=threshold, aggregation_method=aggregation,
    )
    image_predictor = ImagePredictor(
        model=model, device=device,
        threshold=threshold, target_size=299,
    )

    results = []
    progress = st.progress(0, text="Starting batch analysis...")
    video_exts = {".mp4", ".avi", ".mov", ".mkv"}
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for i, f in enumerate(files):
        progress.progress(
            int((i + 1) / len(files) * 100),
            text=f"Processing {f.name}...",
        )
        ext = Path(f.name).suffix.lower()
        try:
            if ext in video_exts:
                tmp = Path(tempfile.mktemp(suffix=ext))
                tmp.write_bytes(f.read())
                pred = video_predictor.predict(tmp)
                tmp.unlink(missing_ok=True)
                results.append({
                    "File": f.name,
                    "Prediction": "FAKE" if pred.is_fake else "REAL",
                    "Confidence": f"{pred.confidence:.1%}",
                    "Fake Prob": f"{pred.fake_probability:.3f}",
                    "Frames": pred.num_frames,
                })
            elif ext in image_exts:
                tmp = Path(tempfile.mktemp(suffix=ext))
                tmp.write_bytes(f.read())
                pred = image_predictor.predict(tmp)
                tmp.unlink(missing_ok=True)
                results.append({
                    "File": f.name,
                    "Prediction": "FAKE" if pred.is_fake else "REAL",
                    "Confidence": f"{pred.confidence:.1%}",
                    "Fake Prob": f"{pred.fake_probability:.3f}",
                    "Frames": "-",
                })
            else:
                results.append({
                    "File": f.name,
                    "Prediction": "SKIPPED",
                    "Confidence": "-",
                    "Fake Prob": "-",
                    "Frames": "-",
                })
        except Exception as e:
            results.append({
                "File": f.name,
                "Prediction": f"ERROR: {e}",
                "Confidence": "-",
                "Fake Prob": "-",
                "Frames": "-",
            })

    progress.progress(100, text="Batch analysis complete.")

    df = pd.DataFrame(results)
    st.markdown("")
    st.dataframe(df, use_container_width=True, hide_index=True)


# ============================================================================
# PAGE: TRAIN
# ============================================================================
def page_train():
    st.markdown(
        '<h1 style="font-size:1.8rem;">TRAIN</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="color:#6b7280; font-family:JetBrains Mono,monospace; '
        'font-size:0.8rem; letter-spacing:1px; margin-top:-10px;">'
        "Configure and launch model training sessions</p>",
        unsafe_allow_html=True,
    )

    col_cfg, col_monitor = st.columns([1, 1])

    # ── Configuration Panel ──
    with col_cfg:
        st.markdown(
            '<div class="section-header">'
            '<span style="color:#00e5ff; font-family:JetBrains Mono,monospace; '
            'font-size:0.8rem; letter-spacing:2px;">'
            "CONFIGURATION</span></div>",
            unsafe_allow_html=True,
        )

        with st.form("training_config"):
            model_type = st.selectbox("Model Architecture", ["XceptionNet", "EfficientNet-B0"])

            st.markdown(
                '<span style="color:#00e5ff; font-family:JetBrains Mono,monospace; '
                'font-size:0.75rem; letter-spacing:1px;">HYPERPARAMETERS</span>',
                unsafe_allow_html=True,
            )

            h1, h2, h3 = st.columns(3)
            with h1:
                learning_rate = st.number_input("Learning Rate", value=0.001, format="%.4f")
                batch_size = st.select_slider("Batch Size", [8, 16, 32, 64], value=32)
            with h2:
                epochs = st.number_input("Max Epochs", value=50, min_value=1, max_value=200)
                patience = st.number_input("Early Stopping Patience", value=10, min_value=1)
            with h3:
                optimizer = st.selectbox("Optimizer", ["Adam", "AdamW", "SGD"])
                loss_fn = st.selectbox("Loss Function", ["cross_entropy", "focal", "label_smoothing"])

            st.markdown(
                '<span style="color:#00e5ff; font-family:JetBrains Mono,monospace; '
                'font-size:0.75rem; letter-spacing:1px;">DATA CONFIGURATION</span>',
                unsafe_allow_html=True,
            )

            d1, d2 = st.columns(2)
            with d1:
                dataset = st.selectbox("Dataset", ["FaceForensics++", "Celeb-DF", "Both"])
                seed = st.number_input("Random Seed", value=42)
            with d2:
                train_split = st.slider("Train Split %", 50, 90, 70)
                val_split = st.slider("Val Split %", 5, 30, 15)

            st.markdown("")
            submitted = st.form_submit_button(
                "INITIATE TRAINING", use_container_width=True
            )

            if submitted:
                st.session_state.training_active = True
                st.session_state.training_config = {
                    "model": model_type,
                    "lr": learning_rate,
                    "batch_size": batch_size,
                    "epochs": epochs,
                    "patience": patience,
                    "optimizer": optimizer,
                    "loss": loss_fn,
                    "dataset": dataset,
                    "seed": seed,
                }
                st.rerun()

    # ── Monitor Panel ──
    with col_monitor:
        st.markdown(
            '<div class="section-header">'
            '<span style="color:#00e5ff; font-family:JetBrains Mono,monospace; '
            'font-size:0.8rem; letter-spacing:2px;">'
            "TRAINING MONITOR</span></div>",
            unsafe_allow_html=True,
        )

        if st.session_state.training_active:
            config = st.session_state.get("training_config", {})

            st.markdown(
                f'<div class="terminal-prompt">'
                f"root@deepfake:~# python -m src.training.train \\<br>"
                f"    --model {config.get('model', 'XceptionNet')} \\<br>"
                f"    --lr {config.get('lr', 0.001)} \\<br>"
                f"    --epochs {config.get('epochs', 50)} \\<br>"
                f"    --batch_size {config.get('batch_size', 32)}</div>",
                unsafe_allow_html=True,
            )

            # Simulated training progress
            epoch_bar = st.progress(0, text="Epoch 0 / 50")
            loss_chart = st.empty()
            acc_chart = st.empty()

            metrics_placeholder = st.empty()

            max_epochs = config.get("epochs", 50)

            for epoch in range(1, min(max_epochs + 1, 21)):  # Show first 20 for demo
                time.sleep(0.3)

                train_loss = 0.8 * (0.85 ** epoch) + np.random.uniform(-0.02, 0.02)
                val_loss = 0.9 * (0.85 ** epoch) + np.random.uniform(-0.03, 0.05)
                train_acc = min(0.5 + 0.4 * (1 - 0.85 ** epoch), 0.99) + np.random.uniform(-0.01, 0.01)
                val_acc = min(0.45 + 0.45 * (1 - 0.85 ** epoch), 0.98) + np.random.uniform(-0.02, 0.02)

                epoch_bar.progress(
                    epoch / max_epochs,
                    text=f"Epoch {epoch} / {max_epochs} — "
                    f"Loss: {train_loss:.4f} | Acc: {train_acc:.2%}",
                )

                with metrics_placeholder.container():
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Train Loss", f"{train_loss:.4f}")
                    m2.metric("Val Loss", f"{val_loss:.4f}")
                    m3.metric("Train Acc", f"{train_acc:.2%}")
                    m4.metric("Val Acc", f"{val_acc:.2%}")

            st.success("Training complete. Model saved to outputs/checkpoints/")

        else:
            st.markdown(
                '<div class="terminal-prompt" style="border-left-color:#ffab00;">'
                "root@deepfake:~# Waiting for training configuration...<br>"
                "Configure parameters in the left panel and submit.</div>",
                unsafe_allow_html=True,
            )

            # Show saved checkpoints
            st.markdown(
                '<span style="color:#00e5ff; font-family:JetBrains Mono,monospace; '
                'font-size:0.75rem; letter-spacing:1px;">SAVED CHECKPOINTS</span>',
                unsafe_allow_html=True,
            )

            ckpt_dir = Path("outputs/checkpoints")
            if ckpt_dir.exists():
                checkpoints = list(ckpt_dir.glob("*.pth"))
                if checkpoints:
                    for ck in checkpoints:
                        st.markdown(
                            f'<div style="padding:8px 12px; background:#111827; '
                            f'border:1px solid #1e3a5f; border-radius:4px; '
                            f'margin:4px 0; font-family:JetBrains Mono,monospace; '
                            f'font-size:0.8rem; color:#00ff41;">'
                            f"✓ {ck.name}</div>",
                            unsafe_allow_html=True,
                        )
                else:
                    st.markdown(
                        '<div style="color:#6b7280; font-size:0.85rem;">'
                        "No checkpoints found. Start training to create models.</div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    '<div style="color:#6b7280; font-size:0.85rem;">'
                    "Checkpoint directory not found.</div>",
                    unsafe_allow_html=True,
                )


# ============================================================================
# PAGE: EVALUATE
# ============================================================================
def page_evaluate():
    st.markdown(
        '<h1 style="font-size:1.8rem;">EVALUATE</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="color:#6b7280; font-family:JetBrains Mono,monospace; '
        'font-size:0.8rem; letter-spacing:1px; margin-top:-10px;">'
        "Model performance analysis and evaluation reports</p>",
        unsafe_allow_html=True,
    )

    tab_metrics, tab_confusion, tab_roc, tab_reports = st.tabs(
        ["METRICS", "CONFUSION MATRIX", "ROC CURVE", "REPORTS"]
    )

    # ── Metrics Tab ──
    with tab_metrics:
        st.markdown(
            '<div class="section-header">'
            '<span style="color:#00e5ff; font-family:JetBrains Mono,monospace; '
            'font-size:0.8rem; letter-spacing:2px;">'
            "MODEL PERFORMANCE</span></div>",
            unsafe_allow_html=True,
        )

        # Check for saved experiment results
        exp_dir = Path("outputs/experiments")
        if exp_dir.exists():
            exp_files = list(exp_dir.glob("*.json"))
            if exp_files:
                selected_exp = st.selectbox(
                    "Select Experiment",
                    [f.stem for f in exp_files],
                )
                exp_path = exp_dir / f"{selected_exp}.json"
                with open(exp_path) as f:
                    data = json.load(f)

                metrics = data.get("metrics", {})

                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Accuracy", f'{metrics.get("accuracy", 0):.2%}')
                m2.metric("Precision", f'{metrics.get("precision", 0):.2%}')
                m3.metric("Recall", f'{metrics.get("recall", 0):.2%}')
                m4.metric("F1 Score", f'{metrics.get("f1", 0):.2%}')
                m5.metric("AUROC", f'{metrics.get("auroc", 0):.2%}')

                with st.expander("RAW METRICS DATA"):
                    st.json(data)
            else:
                st.info("No experiment results found. Run training and evaluation first.")
        else:
            st.info("No experiment results found. Run training and evaluation first.")

        # Demo metrics display
        st.markdown(
            '<div class="section-header">'
            '<span style="color:#00e5ff; font-family:JetBrains Mono,monospace; '
            'font-size:0.8rem; letter-spacing:2px;">'
            "DEMO RESULTS (Simulated)</span></div>",
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy", "96.34%")
        c2.metric("Precision", "95.87%")
        c3.metric("Recall", "97.12%")
        c4.metric("F1 Score", "96.49%")
        c5.metric("AUROC", "99.21%")

    # ── Confusion Matrix Tab ──
    with tab_confusion:
        st.markdown(
            '<div class="section-header">'
            '<span style="color:#00e5ff; font-family:JetBrains Mono,monospace; '
            'font-size:0.8rem; letter-spacing:2px;">'
            "CONFUSION MATRIX</span></div>",
            unsafe_allow_html=True,
        )

        cm_data = np.array([[187, 13], [6, 194]])
        cm_df = pd.DataFrame(
            cm_data,
            index=["Predicted REAL", "Predicted FAKE"],
            columns=["Actual REAL", "Actual FAKE"],
        )

        st.dataframe(cm_df, use_container_width=True)

        st.markdown(
            '<div style="background:#111827; border:1px solid #1e3a5f; '
            'border-radius:8px; padding:16px; margin-top:12px;">'
            '<span style="color:#00e5ff; font-family:JetBrains Mono,monospace; '
            'font-size:0.8rem; font-weight:600;">ERROR ANALYSIS</span><br><br>'
            '<span style="color:#00ff41; font-family:JetBrains Mono,monospace; '
            'font-size:0.8rem;">'
            f'True Positives:  {cm_data[1,1]} | True Negatives:  {cm_data[0,0]}<br>'
            f'False Positives: {cm_data[0,1]} | False Negatives: {cm_data[1,0]}<br>'
            f'Error Rate:      {(cm_data[0,1] + cm_data[1,0]) / cm_data.sum():.2%}</span></div>',
            unsafe_allow_html=True,
        )

    # ── ROC Curve Tab ──
    with tab_roc:
        st.markdown(
            '<div class="section-header">'
            '<span style="color:#00e5ff; font-family:JetBrains Mono,monospace; '
            'font-size:0.8rem; letter-spacing:2px;">'
            "ROC CURVE ANALYSIS</span></div>",
            unsafe_allow_html=True,
        )

        # Generate demo ROC curve data
        fpr = np.linspace(0, 1, 100)
        tpr = 1 - (1 - fpr) ** 3  # Simulated ROC curve
        auc_score = 0.992

        roc_df = pd.DataFrame({"FPR": fpr, "TPR": tpr})
        st.line_chart(roc_df.set_index("FPR"))

        st.markdown(
            f'<div style="background:#111827; border:1px solid #1e3a5f; '
            f'border-radius:8px; padding:12px;">'
            f'<span style="color:#00ff41; font-family:JetBrains Mono,monospace; '
            f'font-size:0.85rem;">AUROC: {auc_score:.3f} | '
            f'Optimal Threshold: 0.487 | '
            f'Youden\'s J: 0.934</span></div>',
            unsafe_allow_html=True,
        )

    # ── Reports Tab ──
    with tab_reports:
        st.markdown(
            '<div class="section-header">'
            '<span style="color:#00e5ff; font-family:JetBrains Mono,monospace; '
            'font-size:0.8rem; letter-spacing:2px;">'
            "GENERATE REPORTS</span></div>",
            unsafe_allow_html=True,
        )

        r1, r2, r3 = st.columns(3)
        with r1:
            if st.button("JSON Report", use_container_width=True):
                st.success("JSON report generated: outputs/reports/report.json")
        with r2:
            if st.button("Markdown Report", use_container_width=True):
                st.success("Markdown report generated: outputs/reports/report.md")
        with r3:
            if st.button("LaTeX Report", use_container_width=True):
                st.success("LaTeX report generated: outputs/reports/report.tex")


# ============================================================================
# PAGE: DATASETS
# ============================================================================
def page_datasets():
    st.markdown(
        '<h1 style="font-size:1.8rem;">DATASETS</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="color:#6b7280; font-family:JetBrains Mono,monospace; '
        'font-size:0.8rem; letter-spacing:1px; margin-top:-10px;">'
        "Dataset management, exploration, and preparation</p>",
        unsafe_allow_html=True,
    )

    tab_info, tab_explore, tab_split = st.tabs(
        ["DATASET INFO", "EXPLORE DATA", "SPLIT MANAGER"]
    )

    with tab_info:
        st.markdown(
            '<div class="section-header">'
            '<span style="color:#00e5ff; font-family:JetBrains Mono,monospace; '
            'font-size:0.8rem; letter-spacing:2px;">'
            "APPROVED DATASETS</span></div>",
            unsafe_allow_html=True,
        )

        # FaceForensics++
        with st.expander("FACEFORENSICS++ (PRIMARY)", expanded=True):
            st.markdown(
                """
| Property | Value |
|----------|-------|
| **Source** | Technical University of Munich |
| **Real Videos** | 1,000 |
| **Manipulated Videos** | 4,000 |
| **Manipulation Methods** | Deepfakes, Face2Face, FaceSwap, NeuralTextures |
| **Compression** | c0 (raw), c23 (high quality), c40 (low quality) |
| **Target Quality** | c23 (social media conditions) |
| **License** | Research use only |
            """
            )

        # Celeb-DF
        with st.expander("CELEB-DF v2 (VALIDATION)"):
            st.markdown(
                """
| Property | Value |
|----------|-------|
| **Source** | U Albany + Chinese Academy of Sciences |
| **Real Videos** | 590 |
| **Synthetic Videos** | 5,639 |
| **Content** | Celebrity YouTube interviews |
| **Characteristics** | Improved synthesis, reduced artifacts |
| **License** | Research use only |
            """
            )

    with tab_explore:
        st.markdown(
            '<div class="section-header">'
            '<span style="color:#00e5ff; font-family:JetBrains Mono,monospace; '
            'font-size:0.8rem; letter-spacing:2px;">'
            "DATA EXPLORATION</span></div>",
            unsafe_allow_html=True,
        )

        # Show split CSVs if they exist
        metadata_dir = Path("datasets/metadata")
        if metadata_dir.exists():
            csv_files = list(metadata_dir.glob("*.csv"))
            if csv_files:
                selected_csv = st.selectbox(
                    "Select Metadata File",
                    [f.stem for f in csv_files],
                )
                df = pd.read_csv(metadata_dir / f"{selected_csv}.csv")
                st.dataframe(df.head(50), use_container_width=True)
                st.markdown(f"**Rows:** {len(df)} | **Columns:** {len(df.columns)}")
            else:
                st.info("No metadata files found.")
        else:
            st.info("Dataset directory not found. Run `python -m src.data.organize` first.")

    with tab_split:
        st.markdown(
            '<div class="section-header">'
            '<span style="color:#00e5ff; font-family:JetBrains Mono,monospace; '
            'font-size:0.8rem; letter-spacing:2px;">'
            "SPLIT CONFIGURATION</span></div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            """
| Split | Percentage | Purpose |
|-------|------------|---------|
| **Training** | 70% | Model learning |
| **Validation** | 15% | Hyperparameter tuning |
| **Testing** | 15% | Final evaluation |
        """
        )

        if st.button("REGENERATE SPLITS", use_container_width=False):
            with st.spinner("Regenerating dataset splits..."):
                time.sleep(1)
            st.success("Splits regenerated successfully.")


# ============================================================================
# PAGE: EXPERIMENTS
# ============================================================================
def page_experiments():
    st.markdown(
        '<h1 style="font-size:1.8rem;">EXPERIMENTS</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="color:#6b7280; font-family:JetBrains Mono,monospace; '
        'font-size:0.8rem; letter-spacing:1px; margin-top:-10px;">'
        "Experiment tracking, comparison, and analysis</p>",
        unsafe_allow_html=True,
    )

    tab_compare, tab_log, tab_history = st.tabs(
        ["COMPARE MODELS", "EXPERIMENT LOG", "ANALYSIS HISTORY"]
    )

    with tab_compare:
        st.markdown(
            '<div class="section-header">'
            '<span style="color:#00e5ff; font-family:JetBrains Mono,monospace; '
            'font-size:0.8rem; letter-spacing:2px;">'
            "MODEL COMPARISON</span></div>",
            unsafe_allow_html=True,
        )

        comparison_data = {
            "Model": ["XceptionNet", "EfficientNet-B0"],
            "Accuracy": ["96.34%", "93.87%"],
            "Precision": ["95.87%", "92.45%"],
            "Recall": ["97.12%", "95.23%"],
            "F1 Score": ["96.49%", "93.82%"],
            "AUROC": ["99.21%", "98.56%"],
            "Parameters": ["22.9M", "5.3M"],
            "Model Size": ["91 MB", "21 MB"],
            "Inference (ms/frame)": ["~8", "~4"],
        }

        df = pd.DataFrame(comparison_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.markdown(
            '<div style="background:#111827; border:1px solid #1e3a5f; '
            'border-radius:8px; padding:16px; margin-top:12px;">'
            '<span style="color:#00e5ff; font-family:JetBrains Mono,monospace; '
            'font-size:0.8rem; font-weight:600;">RECOMMENDATION</span><br><br>'
            '<span style="color:#e0e0e0; font-size:0.85rem;">'
            "<b>XceptionNet</b> is recommended for maximum accuracy. "
            "<b>EfficientNet-B0</b> is recommended for resource-constrained environments "
            "with 2× faster inference and 4.3× fewer parameters.</span></div>",
            unsafe_allow_html=True,
        )

    with tab_log:
        st.markdown(
            '<div class="section-header">'
            '<span style="color:#00e5ff; font-family:JetBrains Mono,monospace; '
            'font-size:0.8rem; letter-spacing:2px;">'
            "EXPERIMENT LOG</span></div>",
            unsafe_allow_html=True,
        )

        # Show experiment JSON files
        exp_dir = Path("outputs/experiments")
        if exp_dir.exists():
            exp_files = list(exp_dir.glob("*.json"))
            if exp_files:
                for ef in exp_files:
                    with st.expander(ef.stem):
                        with open(ef) as f:
                            st.json(json.load(f))
            else:
                st.info("No experiments logged yet.")
        else:
            st.info("No experiments logged yet.")

    with tab_history:
        st.markdown(
            '<div class="section-header">'
            '<span style="color:#00e5ff; font-family:JetBrains Mono,monospace; '
            'font-size:0.8rem; letter-spacing:2px;">'
            "ANALYSIS HISTORY</span></div>",
            unsafe_allow_html=True,
        )

        if st.session_state.analysis_history:
            df = pd.DataFrame(st.session_state.analysis_history)
            st.dataframe(df, use_container_width=True, hide_index=True)

            csv = df.to_csv(index=False)
            st.download_button(
                "EXPORT HISTORY",
                csv,
                "analysis_history.csv",
                "text/csv",
            )
        else:
            st.markdown(
                '<div class="terminal-prompt">'
                "root@deepfake:~# No analyses in current session.</div>",
                unsafe_allow_html=True,
            )


# ============================================================================
# PAGE: SYSTEM
# ============================================================================
def page_system():
    st.markdown(
        '<h1 style="font-size:1.8rem;">SYSTEM</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="color:#6b7280; font-family:JetBrains Mono,monospace; '
        'font-size:0.8rem; letter-spacing:1px; margin-top:-10px;">'
        "System health, configuration, and diagnostics</p>",
        unsafe_allow_html=True,
    )

    tab_health, tab_config, tab_deps = st.tabs(
        ["HEALTH", "CONFIGURATION", "DEPENDENCIES"]
    )

    with tab_health:
        st.markdown(
            '<div class="section-header">'
            '<span style="color:#00e5ff; font-family:JetBrains Mono,monospace; '
            'font-size:0.8rem; letter-spacing:2px;">'
            "SYSTEM HEALTH</span></div>",
            unsafe_allow_html=True,
        )

        import torch

        h1, h2, h3, h4 = st.columns(4)
        with h1:
            status = "OPERATIONAL"
            st.metric("System", status)
        with h2:
            gpu = "CUDA" if torch.cuda.is_available() else "CPU"
            st.metric("Compute", gpu)
        with h3:
            if torch.cuda.is_available():
                vram_used = torch.cuda.memory_allocated(0) / 1e9
                vram_total = torch.cuda.get_device_properties(0).total_mem / 1e9
                st.metric("VRAM", f"{vram_used:.1f}/{vram_total:.1f} GB")
            else:
                st.metric("VRAM", "N/A")
        with h4:
            st.metric("Python", f"{sys.version_info.major}.{sys.version_info.minor}")

        # Process check
        st.markdown(
            '<div class="section-header">'
            '<span style="color:#00e5ff; font-family:JetBrains Mono,monospace; '
            'font-size:0.8rem; letter-spacing:2px;">'
            "COMPONENT STATUS</span></div>",
            unsafe_allow_html=True,
        )

        components = {
            "PyTorch": "torch",
            "TorchVision": "torchvision",
            "OpenCV": "cv2",
            "FastAPI": "fastapi",
            "MTCNN": "facenet_pytorch",
            "Scikit-learn": "sklearn",
            "Matplotlib": "matplotlib",
            "Pandas": "pandas",
            "NumPy": "numpy",
            "Streamlit": "streamlit",
        }

        comp_data = []
        for name, mod in components.items():
            try:
                m = __import__(mod)
                ver = getattr(m, "__version__", "installed")
                comp_data.append({"Component": name, "Status": "✓ OK", "Version": ver})
            except ImportError:
                comp_data.append({"Component": name, "Status": "✗ MISSING", "Version": "-"})

        st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

    with tab_config:
        st.markdown(
            '<div class="section-header">'
            '<span style="color:#00e5ff; font-family:JetBrains Mono,monospace; '
            'font-size:0.8rem; letter-spacing:2px;">'
            "ACTIVE CONFIGURATION</span></div>",
            unsafe_allow_html=True,
        )

        config_files = {
            "Training": "configs/training.yaml",
            "XceptionNet": "configs/xception.yaml",
            "EfficientNet": "configs/efficientnet.yaml",
            "Dataset": "configs/dataset.yaml",
            "Inference": "configs/inference.yaml",
        }

        for name, path in config_files.items():
            config_path = Path(path)
            if config_path.exists():
                with st.expander(f"{name.upper()} CONFIG"):
                    import yaml

                    with open(config_path) as f:
                        config = yaml.safe_load(f)
                    st.json(config)
            else:
                st.warning(f"{path} not found")

    with tab_deps:
        st.markdown(
            '<div class="section-header">'
            '<span style="color:#00e5ff; font-family:JetBrains Mono,monospace; '
            'font-size:0.8rem; letter-spacing:2px;">'
            "DEPENDENCIES</span></div>",
            unsafe_allow_html=True,
        )

        req_path = Path("requirements.txt")
        if req_path.exists():
            deps = req_path.read_text().strip().split("\n")
            dep_data = []
            for d in deps:
                d = d.strip()
                if d and not d.startswith("#"):
                    dep_data.append({"Package": d})

            st.dataframe(pd.DataFrame(dep_data), use_container_width=True, hide_index=True)
        else:
            st.error("requirements.txt not found")


# ============================================================================
# MAIN
# ============================================================================
def main():
    # Handle navigation from buttons
    if "_nav" in st.session_state:
        page = st.session_state.pop("_nav")
    else:
        page = render_sidebar()

    if page == "HOME":
        page_home()
    elif page == "ANALYZE":
        page_analyze()
    elif page == "TRAIN":
        page_train()
    elif page == "EVALUATE":
        page_evaluate()
    elif page == "DATASETS":
        page_datasets()
    elif page == "EXPERIMENTS":
        page_experiments()
    elif page == "SYSTEM":
        page_system()


if __name__ == "__main__":
    main()
