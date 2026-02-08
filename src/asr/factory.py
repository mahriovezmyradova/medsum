# src/asr/factory.py
"""
Factory for creating ASR models.
"""
import logging
from typing import Dict, List, Any
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from .base import BaseASR, ASRFactory as BaseASRFactory

logger = logging.getLogger(__name__)

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
        model_type = config.get("model_type", "")
        model_name = config.get("name", "").lower()
        
        # Determine model type
        if "whisper" in model_name or "whisper" in model_type:
            from .whisper import WhisperASR
            return WhisperASR(config)
        
        elif "wav2vec" in model_name or "wav2vec" in model_type:
            from .wav2vec2 import Wav2Vec2ASR
            return Wav2Vec2ASR(config)
        
        elif "google" in model_name or "google" in model_type:
            try:
                from .google_speech import GoogleSpeechASR
                return GoogleSpeechASR(config)
            except ImportError:
                logger.warning("Google Speech API not available. Skipping.")
                raise
        
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