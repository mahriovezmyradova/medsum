"""
End-to-end MedicalPipeline:

    audio → ASR → [AI enhancement] → summarizer → PipelineResult

The pipeline is the single object used by both the evaluation script and the
web application, so behaviour is identical in both contexts.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    # ASR
    raw_transcription: str
    asr_model: str
    asr_time: float

    # AI enhancement (may be empty when disabled)
    enhanced_transcription: str
    ai_model: str
    ai_time: float
    ai_improvement_score: float
    ai_enabled: bool

    # Summarization – without AI step
    summary_raw: str
    summarizer: str
    summary_raw_time: float

    # Summarization – with AI step (identical to summary_raw when AI disabled)
    summary_enhanced: str
    summary_enhanced_time: float

    # Metrics (populated by evaluation script, empty in live webapp)
    metrics_raw: Dict[str, float] = field(default_factory=dict)
    metrics_enhanced: Dict[str, float] = field(default_factory=dict)

    total_time: float = 0.0

    @property
    def transcription_used(self) -> str:
        """Return the text that should be shown as the final transcription."""
        return self.enhanced_transcription if self.ai_enabled else self.raw_transcription

    @property
    def summary_used(self) -> str:
        return self.summary_enhanced if self.ai_enabled else self.summary_raw


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────────────────

class MedicalPipeline:
    """
    Assembles the full processing chain.

    Parameters
    ----------
    asr_config : dict
        Passed to WhisperASR or Wav2Vec2ASR.
        Must include "type": "whisper" | "wav2vec2" and model-specific keys.
    summarizer_config : dict
        Passed to one of BartSummarizer / MT5Summarizer / MedicalSummarizer.
        Must include "type": "bart" | "mt5" | "medical".
    ai_config : dict
        Keys: enabled (bool), model (str), api_key (str|None).
    max_audio_seconds : int
        Hard cap for audio length.  Longer recordings are truncated.
    """

    def __init__(
        self,
        asr_config: Dict[str, Any],
        summarizer_config: Dict[str, Any],
        ai_config: Optional[Dict[str, Any]] = None,
        max_audio_seconds: int = 600,
    ):
        self.max_audio_seconds = max_audio_seconds
        self._asr = self._build_asr(asr_config)
        self._summarizer = self._build_summarizer(summarizer_config)
        ai_cfg = ai_config or {}
        from src.pipeline.ai_enhancer import AIEnhancer
        self._enhancer = AIEnhancer(
            model=ai_cfg.get("model", "gpt-4o-mini"),
            api_key=ai_cfg.get("api_key"),
            enabled=ai_cfg.get("enabled", True),
        )

    # ── builder helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _build_asr(cfg: Dict[str, Any]):
        asr_type = cfg.get("type", "whisper").lower()
        if asr_type == "whisper":
            from src.asr.whisper import WhisperASR
            m = WhisperASR(cfg)
        elif asr_type == "wav2vec2":
            from src.asr.wav2vec2 import Wav2Vec2ASR
            m = Wav2Vec2ASR(cfg)
        else:
            raise ValueError(f"Unknown ASR type: {asr_type}")
        m.load_model()
        return m

    @staticmethod
    def _build_summarizer(cfg: Dict[str, Any]):
        s_type = cfg.get("type", "bart").lower()
        if s_type == "bart":
            from src.summarization.medical_summarizer import BartSummarizer
            m = BartSummarizer(cfg)
        elif s_type == "mt5":
            from src.summarization.medical_summarizer import MT5Summarizer
            m = MT5Summarizer(cfg)
        elif s_type == "extractive":
            from src.summarization.medical_summarizer import ExtractiveSummarizer
            m = ExtractiveSummarizer(cfg)
        elif s_type == "medical":
            from src.summarization.medical_summarizer import MedicalSummarizer
            m = MedicalSummarizer(cfg)
        else:
            raise ValueError(f"Unknown summarizer type: {s_type}")
        m.load_model()
        return m

    # ── audio helpers ─────────────────────────────────────────────────────────

    def _truncate_audio(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        max_samples = self.max_audio_seconds * sample_rate
        if len(audio) > max_samples:
            logger.warning(
                "Audio truncated from %.1f s to %d s",
                len(audio) / sample_rate,
                self.max_audio_seconds,
            )
            return audio[:max_samples]
        return audio

    # ── main entry point ──────────────────────────────────────────────────────

    def run(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        reference_text: Optional[str] = None,
    ) -> PipelineResult:
        """
        Process one audio recording end-to-end.

        Parameters
        ----------
        audio : np.ndarray  – float32, mono, already at `sample_rate`
        sample_rate : int
        reference_text : str | None – when provided, ROUGE / medical metrics are computed
        """
        t_total = time.time()
        audio = self._truncate_audio(audio, sample_rate)

        # ── Step 1: ASR ───────────────────────────────────────────────────────
        t_asr = time.time()
        try:
            asr_result = self._asr.transcribe(audio, sample_rate=sample_rate)
            raw_text = asr_result.text
        except Exception as exc:
            logger.error("ASR failed: %s", exc)
            raw_text = ""
        asr_time = time.time() - t_asr

        # ── Step 2: AI enhancement ────────────────────────────────────────────
        t_ai = time.time()
        enh = self._enhancer.enhance(raw_text)
        ai_time = time.time() - t_ai

        # ── Step 3: Summarize raw transcription ───────────────────────────────
        t_sum_raw = time.time()
        sum_raw = self._summarizer.summarize(raw_text)
        sum_raw_time = time.time() - t_sum_raw

        # ── Step 4: Summarize enhanced transcription ──────────────────────────
        if enh.was_enhanced:
            t_sum_enh = time.time()
            sum_enh = self._summarizer.summarize(enh.enhanced_text)
            sum_enh_time = time.time() - t_sum_enh
        else:
            sum_enh = sum_raw
            sum_enh_time = 0.0

        # ── Step 5: Optional metric computation ───────────────────────────────
        metrics_raw: Dict[str, float] = {}
        metrics_enh: Dict[str, float] = {}
        if reference_text:
            from src.summarization.medical_summarizer import SummarizationEvaluator
            evaluator = SummarizationEvaluator()
            metrics_raw = evaluator.evaluate(reference_text, sum_raw.summary)
            metrics_enh = evaluator.evaluate(reference_text, sum_enh.summary)

        return PipelineResult(
            raw_transcription=raw_text,
            asr_model=self._asr.config.get("name", "asr"),
            asr_time=asr_time,
            enhanced_transcription=enh.enhanced_text,
            ai_model=enh.model_used,
            ai_time=ai_time,
            ai_improvement_score=enh.improvement_score,
            ai_enabled=enh.was_enhanced,
            summary_raw=sum_raw.summary,
            summarizer=self._summarizer.name,
            summary_raw_time=sum_raw_time,
            summary_enhanced=sum_enh.summary,
            summary_enhanced_time=sum_enh_time,
            metrics_raw=metrics_raw,
            metrics_enhanced=metrics_enh,
            total_time=time.time() - t_total,
        )

    def run_from_file(self, audio_path: str | Path) -> PipelineResult:
        """Convenience wrapper that loads an audio file and calls run()."""
        import soundfile as sf
        audio, sr = sf.read(str(audio_path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return self.run(audio, sample_rate=sr)
