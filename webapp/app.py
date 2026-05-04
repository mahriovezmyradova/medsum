"""
MedASR – Medical Conversation Summarizer
Streamlit web application

Flow:
  1. Doctor logs in (simple ID, no auth required for prototype)
  2. Record or upload a patient–doctor consultation
  3. Transcribe with the chosen ASR model (Whisper or Wav2Vec2)
  4. Optionally run GPT to clean the raw ASR text
  5. Summarize with the chosen summarizer
  6. Side-by-side comparison: raw vs AI-enhanced transcription & summary
  7. Save session to Supabase

Run:
    cd /Users/mahriovezmyradova/MedicalASR-Summarization
    streamlit run webapp/app.py
"""

from __future__ import annotations

import io
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import streamlit as st

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from webapp.database import SessionDB

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Page configuration
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="MedASR – Medical Summarizer",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Brand styles ──────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Brand colours ─────────────────────────── */
    :root {
        --brand-primary:   #1A5276;
        --brand-secondary: #2E86C1;
        --brand-accent:    #AED6F1;
        --brand-bg:        #F4F6F7;
        --brand-success:   #1E8449;
        --brand-warning:   #D68910;
    }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, var(--brand-primary), var(--brand-secondary));
        color: white;
        padding: 1.4rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    .main-header h1 { margin: 0; font-size: 1.8rem; }
    .main-header p  { margin: 0; opacity: 0.85; font-size: 0.95rem; }

    /* Step cards */
    .step-card {
        background: white;
        border: 1px solid #D5D8DC;
        border-left: 4px solid var(--brand-secondary);
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
    }
    .step-label {
        font-size: 0.75rem;
        font-weight: 700;
        color: var(--brand-secondary);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.3rem;
    }

    /* Result boxes */
    .result-box {
        background: #F8F9FA;
        border: 1px solid #DEE2E6;
        border-radius: 8px;
        padding: 1rem;
        font-family: 'Georgia', serif;
        font-size: 0.95rem;
        line-height: 1.6;
        min-height: 80px;
    }
    .result-box-enhanced {
        background: #EBF5FB;
        border-color: var(--brand-secondary);
    }

    /* Metric pill */
    .metric-pill {
        display: inline-block;
        background: var(--brand-accent);
        color: var(--brand-primary);
        border-radius: 12px;
        padding: 2px 10px;
        font-size: 0.78rem;
        font-weight: 600;
        margin: 2px 3px;
    }

    /* Recording indicator */
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
    .recording-indicator {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        color: #C0392B;
        font-weight: 600;
        animation: pulse 1.4s infinite;
    }
    .rec-dot {
        width: 10px; height: 10px;
        background: #C0392B;
        border-radius: 50%;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] { background: var(--brand-bg); }

    /* Hide Streamlit branding */
    #MainMenu, footer, header { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <div class="main-header">
        <div style="font-size:2.5rem">🩺</div>
        <div>
            <h1>MedASR – Medical Conversation Summarizer</h1>
            <p>Record · Transcribe · Enhance · Summarize patient–doctor interactions</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar – settings
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Settings")

    doctor_id = st.text_input("Doctor ID", value="doc_001", help="Used to tag saved sessions")

    st.markdown("---")
    st.markdown("**ASR Model**")
    asr_choice = st.selectbox(
        "Transcription model",
        options=[
            "whisper_small (recommended)",
            "whisper_base",
            "whisper_tiny",
            "wav2vec2_jonatasgrosman",
            "wav2vec2_facebook",
        ],
        index=0,
    )

    st.markdown("**Summarizer**")
    sum_choice = st.selectbox(
        "Summarization model",
        options=[
            "bart_german (recommended)",
            "mt5_base",
            "extractive_bert",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown("**AI Enhancement**")
    ai_enabled = st.toggle(
        "Enable GPT correction",
        value=True,
        help="Uses GPT-4o-mini to fix ASR errors before summarization. "
             "Requires OPENAI_API_KEY env variable.",
    )
    if ai_enabled:
        ai_model = st.selectbox(
            "GPT model",
            ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
            index=0,
        )
    else:
        ai_model = "none"

    st.markdown("---")
    st.markdown("**Max recording length**")
    max_seconds = st.slider("Seconds", 30, 600, 300, step=30)

    st.markdown("---")
    st.caption("Runs locally · Patient data stays on-device · Supabase stores session metadata only")

# ──────────────────────────────────────────────────────────────────────────────
# Session state
# ──────────────────────────────────────────────────────────────────────────────

for key, default in [
    ("pipeline", None),
    ("pipeline_result", None),
    ("audio_bytes", None),
    ("audio_duration", 0.0),
    ("processing", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ──────────────────────────────────────────────────────────────────────────────
# Pipeline builder (cached per settings combination)
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading models…")
def get_pipeline(asr_key: str, sum_key: str, ai_key: str, ai_on: bool):
    """Build and cache the MedicalPipeline for the current settings."""
    from src.pipeline.pipeline import MedicalPipeline

    asr_configs = {
        "whisper_small (recommended)": {
            "type": "whisper", "name": "whisper_small",
            "model_size": "small", "language": "de",
        },
        "whisper_base": {
            "type": "whisper", "name": "whisper_base",
            "model_size": "base", "language": "de",
        },
        "whisper_tiny": {
            "type": "whisper", "name": "whisper_tiny",
            "model_size": "tiny", "language": "de",
        },
        "wav2vec2_jonatasgrosman": {
            "type": "wav2vec2", "name": "wav2vec2_jonatasgrosman",
            "model_name": "jonatasgrosman/wav2vec2-large-xlsr-53-german",
            "language": "de",
        },
        "wav2vec2_facebook": {
            "type": "wav2vec2", "name": "wav2vec2_facebook",
            "model_name": "facebook/wav2vec2-large-xlsr-53-german",
            "language": "de",
        },
    }

    sum_configs = {
        "bart_german (recommended)": {
            "type": "bart", "name": "bart_german",
            "model_name": "philschmid/bart-large-german-samsum",
            "max_length": 180, "min_length": 30, "num_beams": 6,
        },
        "mt5_base": {
            "type": "mt5", "name": "mt5_base",
            "model_name": "google/mt5-base",
            "max_length": 150, "min_length": 20, "num_beams": 4,
        },
        "extractive_bert": {
            "type": "extractive", "name": "extractive_bert",
            "num_sentences": 3,
        },
    }

    return MedicalPipeline(
        asr_config=asr_configs[asr_key],
        summarizer_config=sum_configs[sum_key],
        ai_config={"enabled": ai_on, "model": ai_key},
        max_audio_seconds=max_seconds,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Step 1 – Audio input
# ──────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="step-label">Step 1 · Audio Input</div>', unsafe_allow_html=True)

tab_record, tab_upload = st.tabs(["🎙️ Record", "📁 Upload file"])

with tab_record:
    st.markdown(
        '<div class="step-card">'
        "Use your browser's microphone to record the consultation directly. "
        "Click <b>Start recording</b>, speak, then click <b>Stop</b>."
        "</div>",
        unsafe_allow_html=True,
    )
    # st.audio_input was added in Streamlit 1.33
    try:
        recorded = st.audio_input(
            "Record consultation",
            key="audio_recorder",
        )
        if recorded is not None:
            st.session_state["audio_bytes"] = recorded.read()
            st.markdown(
                '<div class="recording-indicator">'
                '<div class="rec-dot"></div> Audio captured – ready to process'
                "</div>",
                unsafe_allow_html=True,
            )
    except AttributeError:
        # Streamlit < 1.33 fallback
        st.info(
            "Live recording requires Streamlit ≥ 1.33. "
            "Please upload an audio file using the Upload tab, or run: pip install --upgrade streamlit"
        )

with tab_upload:
    uploaded = st.file_uploader(
        "Upload audio (WAV, MP3, OGG, M4A)",
        type=["wav", "mp3", "ogg", "m4a", "flac"],
        key="file_upload",
    )
    if uploaded is not None:
        st.session_state["audio_bytes"] = uploaded.read()
        st.audio(st.session_state["audio_bytes"])
        st.success(f"File loaded: **{uploaded.name}**  ({len(st.session_state['audio_bytes']) // 1024} KB)")

# ──────────────────────────────────────────────────────────────────────────────
# Step 2 – Process
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown('<div class="step-label">Step 2 · Process</div>', unsafe_allow_html=True)

col_btn, col_info = st.columns([1, 3])
with col_btn:
    process_btn = st.button(
        "▶ Transcribe & Summarize",
        disabled=st.session_state["audio_bytes"] is None,
        use_container_width=True,
        type="primary",
    )
with col_info:
    st.caption(
        f"ASR: **{asr_choice}**  ·  Summarizer: **{sum_choice}**  ·  "
        f"AI enhancement: **{'ON – ' + ai_model if ai_enabled else 'OFF'}**"
    )

if process_btn and st.session_state["audio_bytes"] is not None:
    progress = st.progress(0, text="Preparing audio…")
    try:
        # ── decode audio ──────────────────────────────────────────────────────
        import soundfile as sf

        audio_io = io.BytesIO(st.session_state["audio_bytes"])
        try:
            audio_arr, sr = sf.read(audio_io, dtype="float32")
        except Exception:
            # Try pydub for MP3/M4A
            try:
                from pydub import AudioSegment
                audio_io.seek(0)
                seg = AudioSegment.from_file(audio_io)
                seg = seg.set_frame_rate(16000).set_channels(1)
                samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
                audio_arr = samples / (2 ** (seg.sample_width * 8 - 1))
                sr = 16000
            except Exception as e:
                st.error(f"Could not decode audio: {e}")
                st.stop()

        if audio_arr.ndim > 1:
            audio_arr = audio_arr.mean(axis=1)

        # resample to 16 kHz if needed
        if sr != 16000:
            try:
                import librosa
                audio_arr = librosa.resample(audio_arr, orig_sr=sr, target_sr=16000)
            except ImportError:
                pass
            sr = 16000

        duration = len(audio_arr) / sr
        st.session_state["audio_duration"] = duration
        progress.progress(15, "Loading models…")

        # ── build pipeline ────────────────────────────────────────────────────
        pipeline = get_pipeline(asr_choice, sum_choice, ai_model, ai_enabled)
        progress.progress(35, "Transcribing…")

        # ── run ───────────────────────────────────────────────────────────────
        result = pipeline.run(audio_arr, sample_rate=sr)
        progress.progress(90, "Saving session…")

        st.session_state["pipeline_result"] = result

        # ── save to Supabase ──────────────────────────────────────────────────
        db = SessionDB()
        session_id = db.save_session(doctor_id, result, duration)
        if session_id:
            st.caption(f"Session saved · ID: `{session_id}`")

        progress.progress(100, "Done!")
        time.sleep(0.3)
        progress.empty()
        st.success(
            f"Processing complete in {result.total_time:.1f}s  "
            f"(ASR {result.asr_time:.1f}s · "
            f"AI {result.ai_time:.1f}s · "
            f"Summarizer {result.summary_raw_time:.1f}s)"
        )

    except Exception as exc:
        progress.empty()
        st.error(f"Processing failed: {exc}")
        logger.exception(exc)

# ──────────────────────────────────────────────────────────────────────────────
# Step 3 – Results
# ──────────────────────────────────────────────────────────────────────────────

result = st.session_state.get("pipeline_result")

if result is not None:
    st.markdown("---")
    st.markdown('<div class="step-label">Step 3 · Results</div>', unsafe_allow_html=True)

    # ── Transcription ─────────────────────────────────────────────────────────
    with st.expander("📝 Transcription", expanded=True):
        if result.ai_enabled:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Raw ASR output**")
                st.markdown(
                    f'<div class="result-box">{result.raw_transcription or "<em>empty</em>"}</div>',
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(f"**GPT-enhanced** ·  improvement score: `{result.ai_improvement_score:.3f}`")
                st.markdown(
                    f'<div class="result-box result-box-enhanced">{result.enhanced_transcription or "<em>empty</em>"}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown("**Transcription**")
            st.markdown(
                f'<div class="result-box">{result.raw_transcription or "<em>empty</em>"}</div>',
                unsafe_allow_html=True,
            )

    # ── Summary ───────────────────────────────────────────────────────────────
    with st.expander("📋 Summary", expanded=True):
        if result.ai_enabled and result.summary_raw != result.summary_enhanced:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Summary (from raw transcription)**")
                st.markdown(
                    f'<div class="result-box">{result.summary_raw or "<em>empty</em>"}</div>',
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown("**Summary (from AI-enhanced transcription)**")
                st.markdown(
                    f'<div class="result-box result-box-enhanced">{result.summary_enhanced or "<em>empty</em>"}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown("**Summary**")
            st.markdown(
                f'<div class="result-box result-box-enhanced">{result.summary_used or "<em>empty</em>"}</div>',
                unsafe_allow_html=True,
            )

    # ── Metrics (if reference available) ─────────────────────────────────────
    if result.metrics_raw or result.metrics_enhanced:
        with st.expander("📊 Metrics", expanded=False):
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("ROUGE-1 (raw)",      f"{result.metrics_raw.get('rouge1', 0):.3f}")
            mc2.metric("ROUGE-1 (enhanced)", f"{result.metrics_enhanced.get('rouge1', 0):.3f}",
                       delta=f"{result.metrics_enhanced.get('rouge1',0) - result.metrics_raw.get('rouge1',0):+.3f}")
            mc3.metric("Med. Preservation (raw)",
                       f"{result.metrics_raw.get('medical_preservation', 0):.3f}")
            mc4.metric("Med. Preservation (enhanced)",
                       f"{result.metrics_enhanced.get('medical_preservation', 0):.3f}",
                       delta=f"{result.metrics_enhanced.get('medical_preservation',0) - result.metrics_raw.get('medical_preservation',0):+.3f}")

    # ── Copy buttons ──────────────────────────────────────────────────────────
    st.markdown("---")
    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        st.download_button(
            "⬇ Download Transcription",
            data=result.transcription_used,
            file_name="transcription.txt",
            mime="text/plain",
        )
    with dl_col2:
        st.download_button(
            "⬇ Download Summary",
            data=result.summary_used,
            file_name="summary.txt",
            mime="text/plain",
        )

# ──────────────────────────────────────────────────────────────────────────────
# Session history (sidebar bottom)
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("---")
    st.markdown("### 🗂️ Recent Sessions")
    db = SessionDB()
    if db.available:
        sessions = db.get_recent_sessions(doctor_id, limit=5)
        if sessions:
            for s in sessions:
                created = s.get("created_at", "")[:16].replace("T", " ")
                rouge = s.get("rouge1_enhanced") or s.get("rouge1_raw") or 0
                st.markdown(
                    f"<small><b>{created}</b><br>"
                    f"ASR: {s.get('asr_model','-')} · "
                    f"ROUGE-1: {rouge:.3f}</small>",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No sessions yet for this doctor ID.")
    else:
        st.caption("Database not connected – sessions are not persisted.")
