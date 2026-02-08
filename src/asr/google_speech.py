"""
Google Cloud Speech-to-Text API implementation.
"""
import numpy as np
from typing import Dict, Any, List, Optional
import logging
import time
import os
from pathlib import Path

from .base import BaseASR, TranscriptionResult, ModelInfo

logger = logging.getLogger(__name__)

class GoogleSpeechASR(BaseASR):
    """Google Cloud Speech-to-Text API implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.client = None
        self.credentials_path = config.get("credentials_path")
        
        # Set language code for German
        if self.language == "de":
            self.language_code = "de-DE"
        else:
            self.language_code = self.language
    
    def load_model(self):
        """Initialize Google Speech client with proper authentication."""
        if self._is_loaded:
            return
        
        try:
            # Check if credentials are available
            if self.credentials_path:
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(self.credentials_path)
            elif not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
                logger.warning("Google credentials not set. Set GOOGLE_APPLICATION_CREDENTIALS environment variable.")
                raise ValueError("Google credentials not configured")
            
            # Import inside try block to catch import errors
            from google.cloud import speech
            
            # Create client
            self.client = speech.SpeechClient()
            self._is_loaded = True
            logger.info("Google Speech API client initialized")
            
        except ImportError:
            raise ImportError("google-cloud-speech not installed. Install with: pip install google-cloud-speech")
        except Exception as e:
            logger.error(f"Failed to initialize Google Speech client: {e}")
            raise RuntimeError(f"Google Speech API initialization failed: {e}")
    
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> TranscriptionResult:
        """Transcribe using Google Speech API."""
        if not self._is_loaded:
            self.load_model()
        
        start_time = time.time()
        
        try:
            from google.cloud import speech
            
            # Convert audio to proper format (16-bit PCM)
            if audio.dtype == np.float32 or audio.dtype == np.float64:
                # Convert from float to int16
                audio_int16 = (audio * 32767).astype(np.int16)
            else:
                audio_int16 = audio.astype(np.int16)
            
            audio_bytes = audio_int16.tobytes()
            
            # Create audio content
            audio_content = speech.RecognitionAudio(content=audio_bytes)
            
            # Create config
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=sample_rate,
                language_code=self.language_code,
                enable_automatic_punctuation=True,
                enable_word_confidence=True,
                model="default"
            )
            
            # Send request
            response = self.client.recognize(config=config, audio=audio_content)
            
            # Process results
            if response.results:
                result = response.results[0]
                transcript = result.alternatives[0].transcript
                confidence = result.alternatives[0].confidence
            else:
                transcript = ""
                confidence = 0.0
            
            processing_time = time.time() - start_time
            
            return TranscriptionResult(
                text=transcript.strip(),
                confidence=confidence,
                processing_time=processing_time,
                metadata={
                    "provider": "google_speech",
                    "language_code": self.language_code,
                    "requires_internet": True
                }
            )
            
        except Exception as e:
            logger.error(f"Google Speech transcription failed: {e}")
            return TranscriptionResult(
                text="",
                confidence=0.0,
                processing_time=time.time() - start_time,
                metadata={"error": str(e), "provider": "google_speech"}
            )
    
    def transcribe_batch(self, audios: List[np.ndarray], 
                        sample_rates: List[int]) -> List[TranscriptionResult]:
        """Batch transcription for Google Speech API."""
        results = []
        
        for audio, sr in zip(audios, sample_rates):
            result = self.transcribe(audio, sr)
            results.append(result)
        
        return results
    
    def get_model_info(self) -> ModelInfo:
        """Get model information."""
        return ModelInfo(
            name="Google Speech-to-Text",
            model_type="google_speech",
            language=self.language_code,
            parameters=None,
            size_gb=None,
            supported_formats=["wav", "flac", "linear16"],
            requires_gpu=False,
            offline=False,
            metadata={
                "provider": "google",
                "requires_internet": True,
                "requires_credentials": True,
                "model_type": "default",
                "features": ["punctuation", "word_confidence"]
            }
        )