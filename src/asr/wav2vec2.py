"""
Wav2Vec2 ASR implementation for German medical conversations.
"""
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import numpy as np
from typing import Dict, List, Optional, Any
import logging
import time
from pathlib import Path
import warnings
import sys

warnings.filterwarnings("ignore")

try:
    from .base import BaseASR, TranscriptionResult, ModelInfo
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from asr.base import BaseASR, TranscriptionResult, ModelInfo

logger = logging.getLogger(__name__)

class Wav2Vec2ASR(BaseASR):
    """Wav2Vec2 ASR implementation using HuggingFace models."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Wav2Vec2 ASR.
        
        Args:
            config: Configuration dictionary with keys:
                - name: Model name
                - model_name: HuggingFace model name
                - language: Target language
                - device: Device to use (auto, cuda, cpu)
                - chunk_length_s: Chunk length in seconds
                - stride_length_s: Stride length in seconds
        """
        super().__init__(config)
        
        self.model_name = config.get("model_name", "facebook/wav2vec2-large-xlsr-53-german")
        self.device = self._detect_device(config.get("device", "auto"))
        self.chunk_length_s = config.get("chunk_length_s", 30)
        self.stride_length_s = config.get("stride_length_s", 5)
        
        # Handle cache directory
        if hasattr(config, 'paths') and hasattr(config.paths, 'cache_dir'):
            self.cache_dir = config.paths.cache_dir
        elif isinstance(config, dict) and 'cache_dir' in config:
            self.cache_dir = Path(config['cache_dir'])
        else:
            self.cache_dir = Path.home() / ".cache" / "huggingface"
        
        logger.info(f"Initialized Wav2Vec2ASR: {self.model_name}")
        logger.info(f"Device: {self.device}, Cache: {self.cache_dir}")
    
    def _detect_device(self, device: str) -> str:
        """Detect and validate device."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def load_model(self):
        if self._is_loaded:
            return
        
        logger.info(f"Loading Wav2Vec2 model: {self.model_name}")
        
        try:
            # Create cache directory
            model_cache_dir = self.cache_dir / "models" / "wav2vec2"
            model_cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Load processor and model
            self._processor = Wav2Vec2Processor.from_pretrained(
                self.model_name,
                cache_dir=str(model_cache_dir)
            )
            
            self._model = Wav2Vec2ForCTC.from_pretrained(
                self.model_name,
                cache_dir=str(model_cache_dir)
            )
            
            # Move to device
            self._model = self._model.to(self.device)
            self._model.eval()
            
            # Freeze model
            for param in self._model.parameters():
                param.requires_grad = False
            
            self._is_loaded = True
            
            logger.info(f"Model loaded successfully")
            
            # FIXED: Test with simple audio - add proper torch import
            try:
                # Import torch here if not already imported
                import torch
                test_audio = np.random.randn(16000).astype(np.float32) * 0.01
                
                # Use transcribe method directly
                with torch.no_grad():
                    # Preprocess
                    inputs = self._processor(
                        test_audio, 
                        sampling_rate=16000,
                        return_tensors="pt",
                        padding=True
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    # Inference
                    logits = self._model(**inputs).logits
                    predicted_ids = torch.argmax(logits, dim=-1)
                    transcription = self._processor.batch_decode(predicted_ids)[0]
                
                logger.info(f"Test transcription: '{transcription[:50]}...'")
            except Exception as test_e:
                logger.warning(f"Test transcription skipped: {test_e}")
            
        except Exception as e:
            logger.error(f"Failed to load Wav2Vec2 model: {e}")
            raise
    
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> TranscriptionResult:
        """Transcribe audio using Wav2Vec2."""
        if not self._is_loaded:
            self.load_model()
        
        start_time = time.time()
        
        try:
            # Preprocess audio
            if sample_rate != 16000:
                audio_tensor = torch.FloatTensor(audio).unsqueeze(0)
                import torchaudio
                transform = torchaudio.transforms.Resample(sample_rate, 16000)
                audio = transform(audio_tensor).squeeze().numpy()
                sample_rate = 16000
            
            # Normalize
            audio = audio.astype(np.float32)
            audio_max = np.max(np.abs(audio))
            if audio_max > 0:
                audio = audio / audio_max
            
            # Prepare input
            inputs = self._processor(
                audio, 
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Inference
            with torch.no_grad():
                logits = self._model(**inputs).logits
            
            # Get predicted ids
            predicted_ids = torch.argmax(logits, dim=-1)
            
            # Decode
            transcription = self._processor.batch_decode(predicted_ids)[0]
            
            # Clean transcription
            transcription = transcription.lower().strip()
            
            # Calculate confidence
            with torch.no_grad():
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                max_probs = torch.max(probabilities, dim=-1).values
                confidence = torch.mean(max_probs).item()
            
            processing_time = time.time() - start_time
            
            return TranscriptionResult(
                text=transcription,
                confidence=confidence,
                processing_time=processing_time,
                metadata={
                    "model_name": self.model_name,
                    "audio_length": len(audio),
                    "audio_duration": len(audio) / 16000
                }
            )
            
        except Exception as e:
            logger.error(f"Wav2Vec2 transcription failed: {e}")
            return TranscriptionResult(
                text="",
                confidence=0.0,
                processing_time=time.time() - start_time,
                metadata={"error": str(e)}
            )
    
    def transcribe_batch(self, audios: List[np.ndarray], 
                        sample_rates: List[int]) -> List[TranscriptionResult]:
        """Batch transcription for Wav2Vec2."""
        results = []
        
        for audio, sr in zip(audios, sample_rates):
            result = self.transcribe(audio, sr)
            results.append(result)
        
        return results
    
    def get_model_info(self) -> ModelInfo:
        """Get detailed model information."""
        params = None
        size_gb = None
        
        if hasattr(self, '_model') and self._model:
            params = sum(p.numel() for p in self._model.parameters())
            
            # Estimate size
            if params:
                size_bytes = params * 4  # Assuming float32
                size_gb = size_bytes / (1024**3)
        
        # Model size mapping (approximate)
        size_map = {
            "base": 95,  # MB
            "large": 315,
            "large-xlsr": 315,
            "large-xlsr-53": 315,
            "large-xlsr-53-german": 315
        }
        
        if size_gb is None:
            for key, mb_size in size_map.items():
                if key in self.model_name:
                    size_gb = mb_size / 1024
                    break
        
        return ModelInfo(
            name=f"Wav2Vec2 ({self.model_name.split('/')[-1]})",
            model_type="wav2vec2",
            language=self.language,
            parameters=params,
            size_gb=size_gb,
            requires_gpu=self.device == "cuda",
            offline=True,
            metadata={
                "model_name": self.model_name,
                "device": self.device,
                "chunk_length_s": self.chunk_length_s,
                "stride_length_s": self.stride_length_s
            }
        )
    
    def get_memory_usage(self) -> Optional[Dict[str, float]]:
        """Get memory usage of the model."""
        if not hasattr(self, '_model') or self.device != "cuda":
            return None
        
        try:
            import torch.cuda as cuda
            
            memory_allocated = cuda.memory_allocated() / 1024**3
            memory_reserved = cuda.memory_reserved() / 1024**3
            
            return {
                "allocated_gb": memory_allocated,
                "reserved_gb": memory_reserved
            }
        except:
            return None
    
    def cleanup(self):
        """Clean up model resources."""
        self._model = None
        self._processor = None
        self._is_loaded = False
        logger.info(f"Cleaned up Wav2Vec2 model")