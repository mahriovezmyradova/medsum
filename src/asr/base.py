"""
Base classes for ASR models.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass, field
import time
import logging
import sys
from pathlib import Path

# Get logger directly (no circular import)
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    
@dataclass
class TranscriptionResult:
    """Result of a transcription."""
    text: str
    confidence: Optional[float] = None
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.confidence is None:
            self.confidence = 1.0

@dataclass
class ModelInfo:
    """Information about an ASR model."""
    name: str
    model_type: str
    language: str
    parameters: Optional[int] = None
    size_gb: Optional[float] = None
    supported_formats: List[str] = field(default_factory=lambda: ["wav", "mp3", "flac"])
    requires_gpu: bool = False
    offline: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

class BaseASR(ABC):
    """Abstract base class for all ASR models."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ASR model.
        
        Args:
            config: Model configuration
        """
        self.config = config
        self.name = config.get("name", self.__class__.__name__)
        self.language = config.get("language", "de")
        self._model = None
        self._processor = None
        self._is_loaded = False
        
        logger.info(f"Initialized {self.name} for language: {self.language}")
    
    @abstractmethod
    def load_model(self):
        """Load the ASR model and processor."""
        pass
    
    @abstractmethod
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> TranscriptionResult:
        """
        Transcribe audio to text.
        
        Args:
            audio: Audio data as numpy array
            sample_rate: Sampling rate of audio
            
        Returns:
            TranscriptionResult object
        """
        pass
    
    def transcribe_batch(self, audios: List[np.ndarray], 
                        sample_rates: List[int]) -> List[TranscriptionResult]:
        """
        Transcribe a batch of audio samples.
        
        Args:
            audios: List of audio arrays
            sample_rates: List of sampling rates
            
        Returns:
            List of TranscriptionResult objects
        """
        if not self._is_loaded:
            self.load_model()
        
        results = []
        total_time = 0.0
        
        for i, (audio, sr) in enumerate(zip(audios, sample_rates)):
            start_time = time.time()
            
            try:
                result = self.transcribe(audio, sr)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to transcribe sample {i}: {e}")
                results.append(TranscriptionResult(
                    text="",
                    confidence=0.0,
                    processing_time=0.0,
                    metadata={"error": str(e)}
                ))
            
            elapsed = time.time() - start_time
            total_time += elapsed
            
            # Log progress
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(audios)} samples "
                          f"({(i + 1)/len(audios)*100:.1f}%)")
        
        avg_time = total_time / len(audios) if audios else 0
        logger.info(f"Batch transcription complete: {len(audios)} samples, "
                   f"avg {avg_time:.2f}s per sample")
        
        return results
    
    @abstractmethod
    def get_model_info(self) -> ModelInfo:
        """Get information about the model."""
        pass
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded
    
    def get_memory_usage(self) -> Optional[Dict[str, float]]:
        """
        Get memory usage of the model.
        
        Returns:
            Dictionary with memory usage in GB, or None if not available
        """
        return None
    
    def cleanup(self):
        """Clean up model resources."""
        self._model = None
        self._processor = None
        self._is_loaded = False
        logger.info(f"Cleaned up {self.name}")

class ASRFactory:
    """Factory for creating ASR models."""
    
    @staticmethod
    def create_model(config: Dict[str, Any]) -> BaseASR:
        """
        Create an ASR model from configuration.
        
        Args:
            config: Model configuration dictionary
            
        Returns:
            ASR model instance
        """
        model_type = config.get("model_type", config.get("name", ""))
        
        if "whisper" in model_type.lower():
            from .whisper import WhisperASR
            return WhisperASR(config)
        
        elif "wav2vec" in model_type.lower():
            from .wav2vec2 import Wav2Vec2ASR
            return Wav2Vec2ASR(config)
        
        elif "google" in model_type.lower():
            from .google_speech import GoogleSpeechASR
            return GoogleSpeechASR(config)
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def create_all_models(configs: List[Dict[str, Any]]) -> Dict[str, BaseASR]:
        """
        Create multiple ASR models.
        
        Args:
            configs: List of model configurations
            
        Returns:
            Dictionary of model_name: model_instance
        """
        models = {}
        
        for config in configs:
            try:
                model = ASRFactory.create_model(config)
                models[config["name"]] = model
                logger.info(f"Created model: {config['name']}")
            except Exception as e:
                logger.error(f"Failed to create model {config.get('name', 'unknown')}: {e}")
        
        return models