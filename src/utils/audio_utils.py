"""
Audio processing utilities.

Handles loading/decoding from every format the MultiMed dataset uses
(OGG Opus bytes inside a dict, raw bytes, numpy arrays) and resamples
everything to 16 kHz mono float32 for ASR models.
"""

from __future__ import annotations

import io
import logging
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


@dataclass
class AudioSample:
    array: np.ndarray
    sampling_rate: int
    duration: float
    metadata: Optional[Dict] = None

    def __post_init__(self):
        if self.array is not None:
            self.duration = len(self.array) / self.sampling_rate

    def __repr__(self):
        return (
            f"AudioSample(duration={self.duration:.2f}s, "
            f"sr={self.sampling_rate}, shape={self.array.shape})"
        )


class AudioProcessor:
    """
    Decode audio from any format the MultiMed parquet dataset uses and
    return a 16 kHz mono float32 AudioSample.

    Dataset audio column format:
        {"bytes": b"OggS..."}   # OGG Opus at 48 kHz
    """

    def __init__(self, target_sr: int = 16000):
        self.target_sr = target_sr
        logger.info("Initialized AudioProcessor with target SR: %d", target_sr)

    # ── public entry point ────────────────────────────────────────────────────

    def process_audio_item(self, audio_item: Any) -> Optional[AudioSample]:
        """Accept any audio representation and return a normalised AudioSample."""
        try:
            if audio_item is None:
                logger.warning("Audio item is None")
                return None
            if isinstance(audio_item, dict):
                return self._from_dict(audio_item)
            if isinstance(audio_item, AudioSample):
                return self._ensure_target(audio_item)
            if isinstance(audio_item, np.ndarray):
                return self._from_array(audio_item, self.target_sr)
            if isinstance(audio_item, bytes):
                return self._from_bytes(audio_item)
            logger.warning("Unsupported audio type: %s", type(audio_item))
            return None
        except Exception as exc:
            logger.error("process_audio_item failed: %s", exc)
            return None

    # ── format handlers ───────────────────────────────────────────────────────

    def _from_dict(self, d: Dict) -> Optional[AudioSample]:
        """Handle {"bytes": ...} and {"array": ..., "sampling_rate": ...} dicts."""
        if "bytes" in d and d["bytes"] is not None:
            return self._from_bytes(d["bytes"])
        if "array" in d and "sampling_rate" in d:
            arr = np.array(d["array"], dtype=np.float32)
            sr  = int(d["sampling_rate"])
            return self._from_array(arr, sr)
        logger.warning("Dict has no recognised audio keys: %s", list(d.keys()))
        return None

    def _from_bytes(self, raw: bytes) -> Optional[AudioSample]:
        """Decode raw bytes (OGG, WAV, FLAC …) via soundfile."""
        try:
            with io.BytesIO(raw) as buf:
                arr, sr = sf.read(buf, dtype="float32")
            return self._from_array(arr, sr)
        except Exception as exc:
            logger.error("Failed to decode audio bytes: %s", exc)
            return None

    def _from_array(self, arr: np.ndarray, sr: int) -> AudioSample:
        """Mono-convert, resample to target_sr, normalise."""
        arr = arr.astype(np.float32)
        if arr.ndim > 1:
            arr = arr.mean(axis=1)         # stereo → mono
        if sr != self.target_sr:
            arr = self.resample_audio(arr, sr, self.target_sr)
        arr = self.normalize_audio(arr)
        return AudioSample(array=arr, sampling_rate=self.target_sr,
                           duration=len(arr) / self.target_sr)

    def _ensure_target(self, sample: AudioSample) -> AudioSample:
        """If an existing AudioSample is at the wrong SR, resample it."""
        if sample.sampling_rate == self.target_sr:
            return sample
        return self._from_array(sample.array, sample.sampling_rate)

    # ── signal processing ─────────────────────────────────────────────────────

    def resample_audio(self, arr: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        if orig_sr == target_sr:
            return arr
        try:
            tensor = torch.from_numpy(arr).unsqueeze(0)   # (1, T)
            resampled = torchaudio.functional.resample(tensor, orig_sr, target_sr)
            return resampled.squeeze(0).numpy()
        except Exception as exc:
            logger.warning("torchaudio resample failed (%s) – trying librosa", exc)
            try:
                import librosa
                return librosa.resample(arr, orig_sr=orig_sr, target_sr=target_sr)
            except Exception as exc2:
                logger.error("librosa resample also failed: %s – returning original", exc2)
                return arr

    @staticmethod
    def normalize_audio(arr: np.ndarray) -> np.ndarray:
        peak = np.max(np.abs(arr))
        return arr / peak if peak > 0 else arr

    def load_audio(self, path: str) -> AudioSample:
        arr, sr = sf.read(path, dtype="float32")
        return self._from_array(arr, sr)

    def save_audio(self, sample: AudioSample, path: str):
        sf.write(path, sample.array, sample.sampling_rate)
