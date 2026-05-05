"""
Whisper ASR implementation.
"""
import whisper
import torch
import numpy as np
from typing import Dict, List, Optional, Any
import logging
import warnings
import time
from pathlib import Path
import sys

warnings.filterwarnings("ignore")

# Fix imports - use absolute imports
try:
    # When running as part of package
    from src.asr.base import BaseASR, TranscriptionResult, ModelInfo
except ImportError:
    # When running directly
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from asr.base import BaseASR, TranscriptionResult, ModelInfo

# Get logger directly
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

try:
    from src.utils.audio_utils import AudioProcessor
except ImportError:
    from utils.audio_utils import AudioProcessor

class WhisperASR(BaseASR):
    """Whisper ASR implementation using OpenAI's Whisper models."""
    
    SUPPORTED_SIZES = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Whisper ASR.
        
        Args:
            config: Configuration dictionary with keys:
                - name: Model name
                - model_size: Whisper model size
                - language: Target language
                - device: Device to use (auto, cuda, cpu, mps)
                - compute_type: Compute type (float16, float32)
                - beam_size: Beam size for decoding
                - best_of: Number of candidates for beam search
                - temperature: Temperature for sampling
        """
        super().__init__(config)
        
        self.model_size = config.get("model_size", "base")
        if self.model_size not in self.SUPPORTED_SIZES:
            raise ValueError(f"Unsupported model size: {self.model_size}. "
                           f"Supported: {self.SUPPORTED_SIZES}")
        
        self.device = self._detect_device(config.get("device", "auto"))
        self.compute_type = config.get("compute_type", "float16" if self.device == "cuda" else "float32")
        self.beam_size = config.get("beam_size", 5)
        self.best_of = config.get("best_of", 5)
        self.temperature = config.get("temperature", 0.0)
        
        self.audio_processor = AudioProcessor(target_sr=16000)
        
        logger.info(f"Initialized WhisperASR: {self.model_size}, "
                   f"device: {self.device}, language: {self.language}")
    
    def _detect_device(self, device: str) -> str:
        """Detect and validate device."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def load_model(self):
        """Load Whisper model."""
        if self._is_loaded:
            return
        
        logger.info(f"Loading Whisper model: {self.model_size}")
        
        try:
            # Load model
            self._model = whisper.load_model(
                self.model_size,
                device=self.device,
                download_root=str(Path.home() / ".cache" / "whisper")
            )
            
            # Note: Newer versions of Whisper don't have set_beams method
            # Remove this line if it causes issues
            # if hasattr(self._model, 'decoder'):
            #     self._model.decoder.set_beams(self.beam_size)
            
            self._is_loaded = True
            
            # Test model
            test_result = self._test_model()
            logger.info(f"Model loaded successfully. Test: {test_result}")
            
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise
    
    def _test_model(self) -> str:
        """Test model with a simple audio sample."""
        test_audio = np.random.randn(16000).astype(np.float32) * 0.01  # Quiet noise
        result = self.transcribe(test_audio, 16000)
        return f"Transcription: '{result.text[:30]}...'"
    
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> TranscriptionResult:
        """
        Transcribe audio using Whisper.
        
        Args:
            audio: Audio data as numpy array
            sample_rate: Sampling rate of audio
            
        Returns:
            TranscriptionResult object
        """
        if not self._is_loaded:
            self.load_model()
        
        start_time = time.time()
        
        try:
            # Preprocess audio
            if sample_rate != 16000:
                audio = self.audio_processor.resample_audio(audio, sample_rate, 16000)
                sample_rate = 16000
            
            # Normalize
            audio = audio.astype(np.float32)
            audio_max = np.max(np.abs(audio))
            if audio_max > 0:
                audio = audio / audio_max
            
            # Temperature tuple enables Whisper's built-in fallback: if greedy
            # decoding produces repetitive or low-confidence output, it retries
            # with progressively higher temperatures instead of aborting.
            #
            # no_speech_threshold=0.99 effectively disables silence suppression —
            # medical audio clips are always speech; empty outputs from false-positive
            # silence detection are worse than a slightly noisy transcription.
            # compression_ratio_threshold=3.5 prevents whisper_small from cutting off
            # mid-utterance when it produces slightly repetitive output.
            options = {
                "language": self.language,
                "task": "transcribe",
                "fp16": self.compute_type == "float16" and self.device == "cuda",
                "beam_size": self.beam_size,
                "best_of": self.best_of,
                "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                "condition_on_previous_text": False,
                "no_speech_threshold": 0.99,
                "compression_ratio_threshold": 3.5,
                "logprob_threshold": -2.5,
            }
            
            # Transcribe
            result = self._model.transcribe(audio, **options)
            
            # Extract information
            text = result["text"].strip()
            confidence = np.exp(result.get("avg_logprob", 0.0)) if "avg_logprob" in result else None
            
            processing_time = time.time() - start_time
            
            # Prepare metadata
            metadata = {
                "language": result.get("language", self.language),
                "language_probability": result.get("language_probability", 1.0),
                "duration": len(audio) / sample_rate,
                "has_speech": result.get("no_speech_prob", 0.0) < 0.5,
                "segments": [
                    {
                        "start": seg.get("start", 0),
                        "end": seg.get("end", 0),
                        "text": seg.get("text", ""),
                        "confidence": np.exp(seg.get("avg_logprob", 0.0)) if "avg_logprob" in seg else None
                    }
                    for seg in result.get("segments", [])
                ]
            }
            
            return TranscriptionResult(
                text=text,
                confidence=confidence,
                processing_time=processing_time,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            return TranscriptionResult(
                text="",
                confidence=0.0,
                processing_time=time.time() - start_time,
                metadata={"error": str(e)}
            )
    
    def get_model_info(self) -> ModelInfo:
        """Get detailed model information."""
        params = None
        size_gb = None
        
        if self._model:
            # Estimate parameters
            params = sum(p.numel() for p in self._model.parameters())
            
            # Estimate size (rough)
            if params:
                bytes_per_param = 2 if self.compute_type == "float16" else 4
                size_gb = (params * bytes_per_param) / (1024**3)
        
        # Model size mapping
        size_map = {
            "tiny": 39,
            "base": 74,
            "small": 244,
            "medium": 769,
            "large": 1550,
            "large-v2": 1550,
            "large-v3": 1550
        }
        
        if size_gb is None and self.model_size in size_map:
            size_gb = size_map[self.model_size] / 1024  # MB to GB
        
        return ModelInfo(
            name=f"Whisper {self.model_size}",
            model_type="whisper",
            language=self.language,
            parameters=params,
            size_gb=size_gb,
            requires_gpu=self.device == "cuda",
            offline=True,
            metadata={
                "model_size": self.model_size,
                "device": self.device,
                "compute_type": self.compute_type,
                "beam_size": self.beam_size,
                "temperature": self.temperature
            }
        )
    
    def get_memory_usage(self) -> Optional[Dict[str, float]]:
        """Get memory usage of the model."""
        if not self._model or self.device != "cuda":
            return None
        
        try:
            import torch.cuda as cuda
            
            memory_allocated = cuda.memory_allocated() / 1024**3  # GB
            memory_reserved = cuda.memory_reserved() / 1024**3  # GB
            memory_free = cuda.get_device_properties(0).total_memory / 1024**3 - memory_reserved
            
            return {
                "allocated_gb": memory_allocated,
                "reserved_gb": memory_reserved,
                "free_gb": memory_free,
                "total_gb": cuda.get_device_properties(0).total_memory / 1024**3
            }
        except:
            return None
    
    def transcribe_batch(self, audios: List[np.ndarray], 
                        sample_rates: List[int]) -> List[TranscriptionResult]:
        """
        Optimized batch transcription for Whisper.
        
        Args:
            audios: List of audio arrays
            sample_rates: List of sampling rates
            
        Returns:
            List of TranscriptionResult objects
        """
        if not self._is_loaded:
            self.load_model()
        
        # Preprocess all audios
        processed_audios = []
        for audio, sr in zip(audios, sample_rates):
            if sr != 16000:
                audio = self.audio_processor.resample(audio, sr, 16000)
            audio = audio.astype(np.float32)
            audio_max = np.max(np.abs(audio))
            if audio_max > 0:
                audio = audio / audio_max
            processed_audios.append(audio)
        
        results = []
        start_time = time.time()
        
        # Process in smaller batches to avoid memory issues
        batch_size = min(8, len(processed_audios))  # Whisper is memory intensive
        
        for i in range(0, len(processed_audios), batch_size):
            batch = processed_audios[i:i + batch_size]
            batch_start = time.time()
            
            for audio in batch:
                try:
                    result = self._model.transcribe(
                        audio,
                        language=self.language,
                        fp16=self.compute_type == "float16" and self.device == "cuda",
                        task="transcribe",
                        beam_size=self.beam_size,
                        best_of=self.best_of,
                        temperature=self.temperature
                    )
                    
                    results.append(TranscriptionResult(
                        text=result["text"].strip(),
                        confidence=np.exp(result.get("avg_logprob", 0.0)) if "avg_logprob" in result else None,
                        processing_time=time.time() - batch_start,
                        metadata={
                            "language": result.get("language", self.language),
                            "batch_index": i
                        }
                    ))
                    
                except Exception as e:
                    logger.error(f"Failed to transcribe batch sample: {e}")
                    results.append(TranscriptionResult(
                        text="",
                        confidence=0.0,
                        processing_time=time.time() - batch_start,
                        metadata={"error": str(e), "batch_index": i}
                    ))
            
            logger.info(f"Processed batch {i//batch_size + 1}/{(len(processed_audios) + batch_size - 1)//batch_size}")
        
        total_time = time.time() - start_time
        logger.info(f"Batch transcription complete: {len(audios)} samples, "
                   f"total time: {total_time:.2f}s, avg: {total_time/len(audios):.2f}s per sample")
        
        return results