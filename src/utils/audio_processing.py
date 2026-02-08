"""
Audio processing utilities.
"""
import numpy as np
import torch
import torchaudio
import soundfile as sf
import io
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

# Import logger directly
import logging
logger = logging.getLogger(__name__)

@dataclass
class AudioData:
    """Container for audio data."""
    array: np.ndarray
    sample_rate: int
    duration: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.array is not None and self.sample_rate > 0:
            self.duration = len(self.array) / self.sample_rate

class AudioProcessor:
    """Audio processing utilities."""
    
    def __init__(self, target_sr: int = 16000, normalize: bool = True, 
                 remove_silence: bool = False):
        """
        Initialize audio processor.
        
        Args:
            target_sr: Target sample rate
            normalize: Whether to normalize audio
            remove_silence: Whether to remove silence
        """
        self.target_sr = target_sr
        self.normalize = normalize
        self.remove_silence = remove_silence
        
    def process(self, audio_item: Any) -> Optional[AudioData]:
        """
        Process audio item from dataset.
        
        Args:
            audio_item: Audio data (dict with bytes, array, etc.)
            
        Returns:
            AudioData or None if processing fails
        """
        try:
            if audio_item is None:
                return None
            
            # Case 1: Dictionary with bytes (OGG format)
            if isinstance(audio_item, dict) and 'bytes' in audio_item:
                return self._process_bytes_audio(audio_item['bytes'])
            
            # Case 2: Dictionary with array
            elif isinstance(audio_item, dict) and 'array' in audio_item:
                array = np.array(audio_item['array'])
                sr = audio_item.get('sampling_rate', self.target_sr)
                return self._process_array(array, sr)
            
            # Case 3: Raw array
            elif isinstance(audio_item, np.ndarray):
                return self._process_array(audio_item, self.target_sr)
            
            # Case 4: Bytes directly
            elif isinstance(audio_item, bytes):
                return self._process_bytes_audio(audio_item)
            
            else:
                return None
                
        except Exception as e:
            return None
    
    def _process_bytes_audio(self, audio_bytes: bytes) -> Optional[AudioData]:
        """Process audio from bytes."""
        try:
            audio_file = io.BytesIO(audio_bytes)
            array, sr = sf.read(audio_file)
            return self._process_array(array, sr)
        except:
            return None
    
    def _process_array(self, array: np.ndarray, sr: int) -> AudioData:
        """Process audio array."""
        # Convert to mono if stereo
        if len(array.shape) > 1:
            array = self._to_mono(array)
        
        # Resample if needed
        if sr != self.target_sr:
            array = self.resample(array, sr, self.target_sr)
            sr = self.target_sr
        
        # Normalize if requested
        if self.normalize:
            array = self.normalize_audio(array)
        
        return AudioData(
            array=array,
            sample_rate=sr,
            duration=len(array) / sr,
            metadata={'original_sample_rate': sr, 'normalized': self.normalize}
        )
    
    def _to_mono(self, array: np.ndarray) -> np.ndarray:
        """Convert stereo to mono."""
        if len(array.shape) == 1:
            return array
        return np.mean(array, axis=0) if array.shape[0] == 2 else array[:, 0]
    
    def resample(self, array: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio using torchaudio."""
        if orig_sr == target_sr:
            return array
        
        audio_tensor = torch.FloatTensor(array)
        if len(audio_tensor.shape) == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        
        resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)
        resampled = resampler(audio_tensor)
        return resampled.squeeze().numpy()
    
    def normalize_audio(self, array: np.ndarray) -> np.ndarray:
        """Normalize audio to range [-1, 1]."""
        if len(array) == 0:
            return array
        max_val = np.max(np.abs(array))
        return array / max_val if max_val > 0 else array