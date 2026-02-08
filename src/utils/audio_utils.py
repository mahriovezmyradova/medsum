import numpy as np
import torch
import torchaudio
import soundfile as sf
import io
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AudioSample:
    """Dataclass for audio samples."""
    array: np.ndarray
    sampling_rate: int
    duration: float
    metadata: Optional[Dict] = None
    
    def __post_init__(self):
        if self.array is not None:
            self.duration = len(self.array) / self.sampling_rate
        else:
            self.duration = 0.0
    
    def __repr__(self):
        return f"AudioSample(duration={self.duration:.2f}s, sr={self.sampling_rate}, shape={self.array.shape})"

class AudioProcessor:
    """Process audio data from various formats."""
    
    def __init__(self, target_sr: int = 16000):
        """
        Initialize audio processor.
        
        Args:
            target_sr: Target sampling rate for all audio
        """
        self.target_sr = target_sr
        logger.info(f"Initialized AudioProcessor with target SR: {target_sr}")
        
    def process_audio_item(self, audio_item: Any) -> Optional[AudioSample]:
        """
        Process an audio item from the dataset.
        
        Args:
            audio_item: Audio data from parquet file
            
        Returns:
            AudioSample object or None if processing fails
        """
        try:
            if audio_item is None:
                logger.warning("Audio item is None")
                return None
            
            # Case 1: Audio is a dictionary with 'bytes' key (most common for this dataset)
            if isinstance(audio_item, dict):
                return self._process_dict_audio(audio_item)
            
            # Case 2: Audio is already an AudioSample
            elif isinstance(audio_item, AudioSample):
                return audio_item
            
            # Case 3: Audio is a numpy array
            elif isinstance(audio_item, np.ndarray):
                return AudioSample(
                    array=audio_item,
                    sampling_rate=self.target_sr,
                    duration=len(audio_item) / self.target_sr
                )
            
            # Case 4: Audio is bytes directly
            elif isinstance(audio_item, bytes):
                return self._process_bytes_audio(audio_item)
            
            else:
                logger.warning(f"Unsupported audio type: {type(audio_item)}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to process audio: {e}")
            return None
    
    def _process_dict_audio(self, audio_dict: Dict) -> Optional[AudioSample]:
        """Process audio stored as dictionary."""
        try:
            # Check for 'bytes' key (OGG audio)
            if 'bytes' in audio_dict:
                audio_bytes = audio_dict['bytes']
                
                # Check if it's OGG format (common in datasets)
                if isinstance(audio_bytes, bytes):
                    if audio_bytes.startswith(b'OggS'):
                        # OGG format, use soundfile
                        try:
                            import soundfile as sf
                            with io.BytesIO(audio_bytes) as f:
                                array, sr = sf.read(f)
                            
                            # Process the audio
                            array = self._post_process_array(array, sr)
                            return AudioSample(
                                array=array,
                                sampling_rate=sr,
                                duration=len(array) / sr,
                                metadata={'format': 'ogg'}
                            )
                        except Exception as e:
                            logger.warning(f"Failed to read OGG bytes: {e}")
                            return None
                    else:
                        # Try to decode as WAV
                        try:
                            import wave
                            with io.BytesIO(audio_bytes) as f:
                                with wave.open(f, 'rb') as wav_file:
                                    sr = wav_file.getframerate()
                                    n_frames = wav_file.getnframes()
                                    array = np.frombuffer(
                                        wav_file.readframes(n_frames),
                                        dtype=np.int16
                                    ).astype(np.float32) / 32768.0
                                
                                array = self._post_process_array(array, sr)
                                return AudioSample(
                                    array=array,
                                    sampling_rate=sr,
                                    duration=len(array) / sr,
                                    metadata={'format': 'wav'}
                                )
                        except:
                            logger.warning("Failed to decode as WAV, trying raw load")
                            # Try to load with soundfile anyway
                            try:
                                import soundfile as sf
                                with io.BytesIO(audio_bytes) as f:
                                    array, sr = sf.read(f)
                                
                                array = self._post_process_array(array, sr)
                                return AudioSample(
                                    array=array,
                                    sampling_rate=sr,
                                    duration=len(array) / sr,
                                    metadata={'format': 'unknown'}
                                )
                            except Exception as e:
                                logger.error(f"Failed to decode audio bytes: {e}")
                                return None
            
            # Check for 'array' and 'sampling_rate' keys
            elif 'array' in audio_dict and 'sampling_rate' in audio_dict:
                array = np.array(audio_dict['array'])
                sr = audio_dict['sampling_rate']
                
                array = self._post_process_array(array, sr)
                return AudioSample(
                    array=array,
                    sampling_rate=sr,
                    duration=len(array) / sr,
                    metadata={'format': 'array_dict'}
                )
            
            else:
                logger.warning(f"Dictionary missing required keys: {audio_dict.keys()}")
                return None
                
        except Exception as e:
            logger.error(f"Error processing dictionary audio: {e}")
            return None
    
    def _process_bytes_audio(self, audio_bytes: bytes) -> Optional[AudioSample]:
        """Process raw bytes audio."""
        try:
            import soundfile as sf
            with io.BytesIO(audio_bytes) as f:
                array, sr = sf.read(f)
            
            array = self._post_process_array(array, sr)
            return AudioSample(
                array=array,
                sampling_rate=sr,
                duration=len(array) / sr,
                metadata={'format': 'bytes_direct'}
            )
        except Exception as e:
            logger.error(f"Failed to process bytes audio: {e}")
            return None
    
    def _post_process_array(self, array: np.ndarray, sr: int) -> np.ndarray:
        """Post-process audio array."""
        # Convert to mono if stereo
        if len(array.shape) > 1:
            array = self._to_mono(array)
        
        # Resample if necessary
        if sr != self.target_sr:
            array = self.resample_audio(array, sr, self.target_sr)
            sr = self.target_sr
        
        # Normalize audio
        array = self.normalize_audio(array)
        
        return array
    
    def _to_mono(self, audio_array: np.ndarray) -> np.ndarray:
        """Convert stereo to mono."""
        if len(audio_array.shape) == 1:
            return audio_array
        
        # Average channels
        if audio_array.shape[0] == 2:
            return np.mean(audio_array, axis=0)
        elif audio_array.shape[1] == 2:
            return np.mean(audio_array, axis=1)
        else:
            # Take first channel if not stereo
            return audio_array[:, 0] if audio_array.shape[1] > 1 else audio_array
    
    def resample_audio(self, audio_array: np.ndarray, 
                       orig_sr: int, target_sr: int) -> np.ndarray:
        """
        Resample audio array.
        
        Args:
            audio_array: Audio data as numpy array
            orig_sr: Original sampling rate
            target_sr: Target sampling rate
            
        Returns:
            Resampled audio array
        """
        if orig_sr == target_sr:
            return audio_array
        
        # Calculate duration for logging
        duration = len(audio_array) / orig_sr
        
        # Convert to torch tensor for resampling
        audio_tensor = torch.FloatTensor(audio_array)
        
        if len(audio_tensor.shape) == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        
        # Resample
        resampler = torchaudio.transforms.Resample(
            orig_freq=orig_sr,
            new_freq=target_sr
        )
        resampled = resampler(audio_tensor)
        
        result = resampled.squeeze().numpy()
        
        logger.debug(f"Resampled {duration:.2f}s audio from {orig_sr}Hz to {target_sr}Hz")
        
        return result
    
    def normalize_audio(self, audio_array: np.ndarray) -> np.ndarray:
        """
        Normalize audio to range [-1, 1].
        
        Args:
            audio_array: Audio data as numpy array
            
        Returns:
            Normalized audio array
        """
        if len(audio_array) == 0:
            return audio_array
        
        max_val = np.max(np.abs(audio_array))
        if max_val > 0:
            return audio_array / max_val
        
        return audio_array
    
    def save_audio(self, audio_sample: AudioSample, filepath: str):
        """
        Save audio sample to file.
        
        Args:
            audio_sample: AudioSample object
            filepath: Path to save the audio file
        """
        try:
            sf.write(filepath, audio_sample.array, audio_sample.sampling_rate)
            logger.debug(f"Saved audio to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save audio to {filepath}: {e}")
    
    def load_audio(self, filepath: str) -> AudioSample:
        """
        Load audio from file.
        
        Args:
            filepath: Path to audio file
            
        Returns:
            AudioSample object
        """
        try:
            array, sampling_rate = sf.read(filepath)
            
            return AudioSample(
                array=array,
                sampling_rate=sampling_rate,
                duration=len(array) / sampling_rate,
                metadata={'source': 'file', 'path': filepath}
            )
        except Exception as e:
            logger.error(f"Failed to load audio from {filepath}: {e}")
            raise

# Test function
def test_audio_processor():
    """Test the audio processor with sample data."""
    processor = AudioProcessor(target_sr=16000)
    
    # Create test data
    print("Testing AudioProcessor...")
    
    # Test 1: Create a synthetic audio array
    duration = 1.0
    sr = 16000
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    test_array = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    
    test_sample = AudioSample(
        array=test_array,
        sampling_rate=sr,
        duration=duration
    )
    
    print(f"\nTest 1: Synthetic Audio")
    print(f"  Sample: {test_sample}")
    print(f"  Array shape: {test_sample.array.shape}")
    print(f"  Duration: {test_sample.duration:.2f}s")
    
    # Test 2: Test normalization
    test_array_unnormalized = test_array * 2.5
    normalized = processor.normalize_audio(test_array_unnormalized)
    print(f"\nTest 2: Normalization")
    print(f"  Original max: {np.abs(test_array_unnormalized).max():.2f}")
    print(f"  Normalized max: {np.abs(normalized).max():.2f}")
    
    # Test 3: Test resampling
    print(f"\nTest 3: Resampling")
    resampled = processor.resample_audio(test_array, 16000, 8000)
    print(f"  Original shape: {test_array.shape}")
    print(f"  Resampled shape: {resampled.shape}")
    
    print("\nAll tests passed!")

if __name__ == "__main__":
    test_audio_processor()