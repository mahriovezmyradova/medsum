"""
Canonical MultiMedLoader - Single consolidated implementation.
Combines features from both loader.py and data_loader.py.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from dataclasses import dataclass, field
import warnings
import yaml
from loguru import logger
from tqdm import tqdm
import sys
from sklearn.model_selection import train_test_split

# Import AudioProcessor with fallback
try:
    from src.utils.audio_processing import AudioProcessor
except ImportError:
    try:
        sys.path.append(str(Path(__file__).parent.parent))
        from utils.audio_processing import AudioProcessor
    except ImportError:
        # Define minimal AudioProcessor if needed
        class AudioProcessor:
            def __init__(self, **kwargs):
                pass
            def process(self, item):
                return item

# Setup logging
logger.remove()  # Remove default logger
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)

@dataclass
class AudioData:
    """Container for audio data with metadata."""
    array: np.ndarray
    sample_rate: int
    duration: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.array is not None and self.sample_rate > 0:
            self.duration = len(self.array) / self.sample_rate
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'sample_rate': self.sample_rate,
            'duration': self.duration,
            'array_shape': self.array.shape,
            **self.metadata
        }

@dataclass
class DatasetSample:
    """Complete dataset sample with audio and text."""
    id: str
    audio: AudioData
    text: str
    duration: float
    split: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'text': self.text,
            'duration': self.duration,
            'split': self.split,
            'text_length': len(self.text),
            'word_count': len(self.text.split()),
            **self.metadata
        }


class MultiMedLoader:
    """
    Single canonical loader for MultiMed German dataset.
    
    Supports both config object and config file initialization.
    Unified API that replaces both previous implementations.
    """
    
    def __init__(self, config=None, config_path: Optional[str] = None):
        """
        Initialize loader with flexible configuration.
        
        Args:
            config: Configuration object (preferred)
            config_path: Path to YAML config file (alternative)
            
        Examples:
            loader = MultiMedLoader(config=config_obj)
            loader = MultiMedLoader(config_path="config.yaml")
        """
        self.config = self._load_config(config, config_path)
        self._setup_audio_processor()
        
        # Caches
        self._splits_data: Dict[str, pd.DataFrame] = {}
        self._samples_cache: Dict[str, List[DatasetSample]] = {}
        self._audio_cache: Dict[str, List[AudioData]] = {}
        
        logger.info(f"Initialized consolidated MultiMedLoader")
        logger.info(f"Data directory: {self._get_config_value('data_dir')}")
    
    def _load_config(self, config, config_path):
        """Load configuration from object or file."""
        if config is not None:
            return config
        elif config_path:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            # Convert to namespace-like object for compatibility
            from types import SimpleNamespace
            config_obj = SimpleNamespace()
            
            # Paths
            paths_ns = SimpleNamespace()
            for key, value in config_dict.get('paths', {}).items():
                setattr(paths_ns, key, Path(value) if isinstance(value, str) else value)
            setattr(config_obj, 'paths', paths_ns)
            
            # Dataset
            dataset_ns = SimpleNamespace()
            for key, value in config_dict.get('dataset', {}).items():
                setattr(dataset_ns, key, value)
            setattr(config_obj, 'dataset', dataset_ns)
            
            return config_obj
        else:
            raise ValueError("Either config or config_path must be provided")
    
    def _get_config_value(self, key: str, default=None, section: str = 'dataset'):
        """Get value from config with fallback."""
        try:
            if hasattr(self.config, section):
                section_obj = getattr(self.config, section)
                if hasattr(section_obj, key):
                    return getattr(section_obj, key)
            
            # Try dict access for backward compatibility
            if isinstance(self.config, dict):
                return self.config.get(section, {}).get(key, default)
            
            return default
        except:
            return default
    
    def _setup_audio_processor(self):
        """Setup audio processor based on config."""
        audio_config = {
            'target_sr': self._get_config_value('target_sample_rate', 16000),
            'normalize': self._get_config_value('normalize_audio', True),
            'remove_silence': self._get_config_value('remove_silence', False)
        }
        
        # Handle different parameter names
        if audio_config['target_sr'] is None:
            audio_config['target_sr'] = self._get_config_value('sample_rate', 16000)
        
        self.audio_processor = AudioProcessor(**audio_config)
    
    # ==================== CORE LOADING METHODS ====================
    
    def load_split(self, split_name: str) -> pd.DataFrame:
        """
        Load a split as DataFrame (unified method name).
        
        Args:
            split_name: Name of split to load
            
        Returns:
            DataFrame with split data
        """
        # Check cache
        if split_name in self._splits_data:
            return self._splits_data[split_name]
        
        splits = self._get_config_value('splits', [])
        if splits and split_name not in splits:
            raise ValueError(f"Split must be one of {splits}")
        
        # Find data directory
        data_dir = self._get_config_value('data_dir', section='paths')
        if not data_dir:
            data_dir = self._get_config_value('raw_data_dir', section='paths')
        
        if not data_dir:
            raise ValueError("No data directory specified in config")
        
        data_dir = Path(data_dir)
        
        # Try different file patterns
        patterns = [
            f"{split_name}-*.parquet",
            f"*{split_name}*.parquet",
            f"{split_name}.parquet"
        ]
        
        file_path = None
        for pattern in patterns:
            files = list(data_dir.glob(pattern))
            if files:
                file_path = files[0]
                break
        
        if file_path is None:
            raise FileNotFoundError(f"No {split_name} files found in {data_dir}")
        
        logger.info(f"Loading {split_name} from {file_path}")
        df = pd.read_parquet(file_path)
        
        # Validate required columns
        audio_col = self._get_config_value('audio_column', 'audio')
        text_col = self._get_config_value('text_column', 'text')
        
        missing_cols = []
        if audio_col not in df.columns:
            missing_cols.append(audio_col)
        if text_col not in df.columns:
            missing_cols.append(text_col)
        
        if missing_cols:
            logger.warning(f"Missing columns in {split_name}: {missing_cols}")
            logger.warning(f"Available columns: {list(df.columns)}")
        
        # Apply max samples limit
        max_samples = self._get_config_value('max_samples_per_split')
        if max_samples and len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=42)
            logger.info(f"Limited {split_name} to {len(df)} samples")
        
        # Cache and return
        self._splits_data[split_name] = df
        logger.info(f"Loaded {split_name}: {len(df)} samples, {len(df.columns)} columns")
        
        return df
    
    # Alias for backward compatibility
    load_split_dataframe = load_split
    
    def load_all_splits(self) -> Dict[str, pd.DataFrame]:
        """
        Load all available splits.
        
        Returns:
            Dictionary with split names as keys and DataFrames as values
        """
        splits = {}
        split_names = self._get_config_value('splits', [])
        
        if not split_names:
            # Try to discover splits from files
            data_dir = Path(self._get_config_value('data_dir', section='paths'))
            if data_dir.exists():
                parquet_files = list(data_dir.glob("*.parquet"))
                split_names = list(set(f.stem.split('-')[0] for f in parquet_files))
        
        for split_name in split_names:
            try:
                splits[split_name] = self.load_split(split_name)
            except FileNotFoundError as e:
                logger.warning(f"Could not load {split_name}: {e}")
        
        return splits
    
    def get_samples(self, split_name: str, max_samples: Optional[int] = None) -> List[DatasetSample]:
        """
        Load and process samples from a split.
        
        Args:
            split_name: Name of split
            max_samples: Maximum number of samples to load
            
        Returns:
            List of DatasetSample objects
        """
        cache_key = f"{split_name}_{max_samples}"
        if cache_key in self._samples_cache:
            return self._samples_cache[cache_key]
        
        df = self.load_split(split_name)
        
        if max_samples and max_samples < len(df):
            df = df.head(max_samples)
        
        samples = []
        failed_count = 0
        
        logger.info(f"Processing {len(df)} samples from {split_name}...")
        
        audio_col = self._get_config_value('audio_column', 'audio')
        text_col = self._get_config_value('text_column', 'text')
        duration_col = self._get_config_value('duration_column')
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split_name}"):
            try:
                # Process audio
                audio_item = row[audio_col]
                audio_array, sample_rate = self._process_audio(audio_item)
                
                if audio_array is None:
                    failed_count += 1
                    continue
                
                # Get text
                text = str(row[text_col]).strip()
                
                # Get duration
                if duration_col and duration_col in row:
                    duration = float(row[duration_col])
                else:
                    duration = len(audio_array) / sample_rate
                
                # Create AudioData
                audio_data = AudioData(
                    array=audio_array,
                    sample_rate=sample_rate,
                    duration=duration,
                    metadata={'original_index': idx}
                )
                
                # Create DatasetSample
                sample = DatasetSample(
                    id=f"{split_name}_{idx}",
                    audio=audio_data,
                    text=text,
                    duration=duration,
                    split=split_name,
                    metadata={
                        'original_index': idx,
                        'text_length': len(text),
                        'word_count': len(text.split())
                    }
                )
                
                samples.append(sample)
                
            except Exception as e:
                logger.warning(f"Failed to process sample {idx}: {e}")
                failed_count += 1
        
        if failed_count > 0:
            logger.warning(f"Failed to process {failed_count}/{len(df)} samples")
        
        logger.info(f"Successfully processed {len(samples)} samples")
        self._samples_cache[cache_key] = samples
        
        return samples
    
    # Alias for backward compatibility
    load_samples = get_samples
    
    def _process_audio(self, audio_item):
        """Process audio item to array and sample rate."""
        try:
            if isinstance(audio_item, (np.ndarray, list)):
                # Already an array
                return np.array(audio_item, dtype=np.float32), 16000
            elif isinstance(audio_item, dict):
                # Dict with array and sr
                return audio_item.get('array'), audio_item.get('sample_rate', 16000)
            elif hasattr(audio_item, '__array__'):
                # Some audio object
                return np.array(audio_item, dtype=np.float32), 16000
            else:
                logger.warning(f"Unknown audio type: {type(audio_item)}")
                return None, None
        except Exception as e:
            logger.warning(f"Audio processing failed: {e}")
            return None, None
    
    # ==================== TEXT PROCESSING METHODS ====================
    
    def get_texts(self, split_name: str, clean: bool = True) -> List[str]:
        """
        Extract text transcripts from a split.
        
        Args:
            split_name: Name of split
            clean: Whether to clean the text
            
        Returns:
            List of text transcripts
        """
        df = self.load_split(split_name)
        text_col = self._get_config_value('text_column', 'text')
        
        if text_col not in df.columns:
            raise ValueError(f"Text column '{text_col}' not found")
        
        texts = df[text_col].astype(str).tolist()
        
        if clean:
            texts = [self._clean_text(text) for text in texts]
        
        return texts
    
    def _clean_text(self, text: str) -> str:
        """Clean text by removing extra whitespace."""
        import re
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    # ==================== STATISTICS METHODS ====================
    
    def get_statistics(self) -> pd.DataFrame:
        """
        Get comprehensive dataset statistics.
        
        Returns:
            DataFrame with statistics per split
        """
        stats = []
        splits = self.load_all_splits()
        
        for split_name, df in splits.items():
            try:
                # Get samples for statistics
                samples = self.get_samples(split_name, max_samples=min(100, len(df)))
                
                if not samples:
                    continue
                
                # Calculate statistics
                text_lengths = [len(s.text) for s in samples]
                word_counts = [s.metadata['word_count'] for s in samples]
                durations = [s.duration for s in samples]
                
                stats.append({
                    'split': split_name,
                    'samples': len(samples),
                    'text_length_mean': np.mean(text_lengths),
                    'text_length_std': np.std(text_lengths),
                    'text_length_min': np.min(text_lengths),
                    'text_length_max': np.max(text_lengths),
                    'word_count_mean': np.mean(word_counts),
                    'word_count_std': np.std(word_counts),
                    'word_count_min': np.min(word_counts),
                    'word_count_max': np.max(word_counts),
                    'duration_mean': np.mean(durations),
                    'duration_std': np.std(durations),
                    'duration_min': np.min(durations),
                    'duration_max': np.max(durations),
                })
                
            except Exception as e:
                logger.error(f"Failed to get stats for {split_name}: {e}")
        
        return pd.DataFrame(stats)
    
    # Alias for backward compatibility
    get_dataset_statistics = get_statistics
    get_dataset_info = get_statistics
    
    # ==================== SAVE METHODS ====================
    
    def save_processed_data(self, split_name: Optional[str] = None, 
                          output_dir: Optional[Union[str, Path]] = None):
        """
        Unified save method that handles both APIs.
        
        Args:
            split_name: Split to save (optional, saves all if None)
            output_dir: Output directory (optional, uses config if None)
            
        Examples:
            # Save specific split
            loader.save_processed_data('train', '/path/to/output')
            
            # Save all splits to config directory
            loader.save_processed_data()
            
            # Legacy API: save all splits to specific directory
            loader.save_processed_data(output_dir='/path/to/output')
        """
        # Handle different API signatures
        if isinstance(split_name, (str, Path)) and output_dir is None:
            # Legacy API: save_processed_data(output_dir)
            output_dir = split_name
            split_name = None
        
        # Determine output directory
        if output_dir is None:
            output_dir = self._get_config_value('processed_dir', section='paths')
            if not output_dir:
                output_dir = Path('data/processed')
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine which splits to save
        if split_name:
            splits_to_save = [split_name]
        else:
            splits_to_save = list(self.load_all_splits().keys())
        
        logger.info(f"Saving processed data to {output_dir}")
        
        for split_name in splits_to_save:
            try:
                self._save_split(split_name, output_dir)
            except Exception as e:
                logger.error(f"Failed to save {split_name}: {e}")
    
    def _save_split(self, split_name: str, output_dir: Path):
        """Save a single split."""
        split_dir = output_dir / split_name
        split_dir.mkdir(exist_ok=True)
        
        samples = self.get_samples(split_name)
        
        # Save audio files
        audio_dir = split_dir / "audio"
        audio_dir.mkdir(exist_ok=True)
        
        for sample in tqdm(samples, desc=f"Saving {split_name} audio"):
            audio_path = audio_dir / f"{sample.id}.wav"
            try:
                import soundfile as sf
                sf.write(audio_path, sample.audio.array, sample.audio.sample_rate)
            except Exception as e:
                logger.warning(f"Failed to save audio {sample.id}: {e}")
        
        # Save metadata
        metadata = []
        for sample in samples:
            metadata.append({
                'id': sample.id,
                'text': sample.text,
                'duration': sample.duration,
                'audio_file': f"audio/{sample.id}.wav",
                'split': sample.split,
                'text_length': len(sample.text),
                'word_count': len(sample.text.split()),
                **sample.metadata
            })
        
        metadata_df = pd.DataFrame(metadata)
        metadata_path = split_dir / "metadata.csv"
        metadata_df.to_csv(metadata_path, index=False, encoding='utf-8')
        
        # Also save as parquet for faster loading
        parquet_path = split_dir / "metadata.parquet"
        metadata_df.to_parquet(parquet_path, index=False)
        
        logger.info(f"Saved {split_name}: {len(samples)} samples")
        logger.info(f"  Audio: {audio_dir}")
        logger.info(f"  Metadata: {metadata_path}")
        logger.info(f"  Parquet: {parquet_path}")
    
    # ==================== UTILITY METHODS ====================
    
    def create_data_splits(self, samples: List[DatasetSample], 
                          train_ratio: float = 0.7,
                          val_ratio: float = 0.15,
                          test_ratio: float = 0.15,
                          random_seed: int = 42) -> Dict[str, List[DatasetSample]]:
        """
        Split samples into train/validation/test sets.
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-10:
            raise ValueError("Ratios must sum to 1.0")
        
        train_samples, temp_samples = train_test_split(
            samples, 
            train_size=train_ratio,
            random_state=random_seed
        )
        
        val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
        val_samples, test_samples = train_test_split(
            temp_samples,
            train_size=val_ratio_adjusted,
            random_state=random_seed
        )
        
        return {
            'train': train_samples,
            'val': val_samples,
            'test': test_samples
        }
    
    def get_audio_samples(self, split_name: str, 
                         max_samples: Optional[int] = None) -> List[AudioData]:
        """
        Extract only audio samples from a split.
        """
        samples = self.get_samples(split_name, max_samples)
        return [sample.audio for sample in samples]


# ==================== FACTORY FUNCTION ====================

def create_loader(config=None, config_path: Optional[str] = None) -> MultiMedLoader:
    """
    Factory function to create loader instance.
    
    Args:
        config: Configuration object
        config_path: Path to YAML config file
        
    Returns:
        MultiMedLoader instance
    """
    return MultiMedLoader(config=config, config_path=config_path)


# ==================== DEPRECATION HANDLING ====================

class LegacyDataLoader:
    """Deprecated - use MultiMedLoader instead."""
    def __init__(self, config_path: str = "../config/config.yaml"):
        warnings.warn(
            "LegacyDataLoader is deprecated. Use MultiMedLoader instead.",
            DeprecationWarning,
            stacklevel=2
        )
        self.loader = MultiMedLoader(config_path=config_path)
    
    def __getattr__(self, name):
        return getattr(self.loader, name)


# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    # Example 1: Using config object
    print("Example 1: Using config object")
    
    from types import SimpleNamespace
    config = SimpleNamespace()
    config.paths = SimpleNamespace()
    config.paths.data_dir = "data/raw"
    config.dataset = SimpleNamespace()
    config.dataset.splits = ["train", "val", "test"]
    config.dataset.audio_column = "audio"
    config.dataset.text_column = "text"
    
    loader1 = MultiMedLoader(config=config)
    stats1 = loader1.get_statistics()
    print(stats1)
    
    # Example 2: Using config file
    print("\nExample 2: Using config file")
    try:
        loader2 = MultiMedLoader(config_path="config.yaml")
        stats2 = loader2.get_statistics()
        print(stats2)
        
        # Save processed data
        loader2.save_processed_data('train', 'data/processed')
        
    except FileNotFoundError:
        print("config.yaml not found, skipping example 2")
    
    print("\nMultiMedLoader consolidated successfully!")