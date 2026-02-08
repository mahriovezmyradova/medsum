"""
Configuration settings for ASR evaluation project.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional
import yaml
import os

@dataclass
class PathConfig:
    """Path configuration."""
    # Base paths
    project_root: Path = Path(__file__).parent.parent
    data_dir: Path = Path("/Users/mahriovezmyradova/MultiMed_dataset/German")
    
    # Derived paths
    @property
    def raw_data_dir(self) -> Path:
        return self.data_dir
    
    @property
    def processed_dir(self) -> Path:
        return self.project_root / "data" / "processed"
    
    @property
    def outputs_dir(self) -> Path:
        return self.project_root / "data" / "outputs"
    
    @property
    def cache_dir(self) -> Path:
        return self.project_root / "data" / "cache"
    
    @property
    def models_dir(self) -> Path:
        return self.project_root / "models"
    
    @property
    def reports_dir(self) -> Path:
        return self.project_root / "reports"
    
    def create_directories(self):
        """Create all necessary directories."""
        directories = [
            self.processed_dir,
            self.outputs_dir / "asr_results",
            self.outputs_dir / "figures",
            self.outputs_dir / "transcripts",
            self.cache_dir / "models",
            self.cache_dir / "transcripts",
            self.models_dir,
            self.reports_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

@dataclass
class DatasetConfig:
    """Dataset configuration."""
    splits: List[str] = field(default_factory=lambda: ["train", "eval", "test"])
    audio_column: str = "audio"
    text_column: str = "text"
    duration_column: str = "duration"
    
    # Sampling
    max_samples_per_split: int = 1000  # For quick testing
    test_split: str = "eval"  # Split to use for evaluation
    random_seed: int = 42
    
    # Audio processing
    target_sample_rate: int = 16000
    normalize_audio: bool = True
    remove_silence: bool = False
    
    @property
    def split_files(self) -> Dict[str, str]:
        """Get expected parquet file names."""
        return {split: f"{split}-00000-of-00001.parquet" for split in self.splits}

@dataclass
class ASRConfig:
    """ASR model configuration."""
    
    # Whisper configurations
    whisper_models: List[Dict[str, Any]] = field(default_factory=lambda: [
        {
            "name": "whisper_tiny",
            "model_size": "tiny",
            "language": "de",
            "device": "auto"
        },
        {
            "name": "whisper_base",
            "model_size": "base", 
            "language": "de",
            "device": "auto"
        },
        {
            "name": "whisper_small",
            "model_size": "small",
            "language": "de",
            "device": "auto"
        },
        {
            "name": "whisper_medium",
            "model_size": "medium",
            "language": "de",
            "device": "auto"
        },
        {
            "name": "whisper_large_v2",
            "model_size": "large-v2",
            "language": "de",
            "device": "auto"
        }
    ])
    
    # Wav2Vec2 configurations
    wav2vec2_models: List[Dict[str, Any]] = field(default_factory=lambda: [
        {
            "name": "wav2vec2_base",
            "model_name": "facebook/wav2vec2-base-960h",
            "language": "de",
            "device": "auto"
        },
        {
            "name": "wav2vec2_large_xlsr",
            "model_name": "facebook/wav2vec2-large-xlsr-53-german",
            "language": "de",
            "device": "auto"
        },
        {
            "name": "wav2vec2_large_xlsr_300m",
            "model_name": "facebook/wav2vec2-large-xlsr-53",
            "language": "de",
            "device": "auto"
        }
    ])
    
    # Google Speech-to-Text
    google_config: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,  # Set to True if you have credentials
        "language_code": "de-DE",
        "use_enhanced": True,
        "model": "default",
        "enable_automatic_punctuation": True
    })
    
    # Common settings
    batch_size: int = 16
    num_workers: int = 4
    compute_type: str = "float16"  # For Whisper
    use_cache: bool = True
    
    @property
    def all_models(self) -> List[Dict[str, Any]]:
        """Get all model configurations."""
        models = []
        models.extend(self.whisper_models)
        models.extend(self.wav2vec2_models)
        
        if self.google_config["enabled"]:
            models.append({
                "name": "google_speech",
                "model_type": "google",
                **self.google_config
            })
        
        return models

@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    # Metrics to compute
    metrics: List[str] = field(default_factory=lambda: [
        "wer", "cer", "mer", "wil",  # Error rates
        "precision", "recall", "f1",  # Word-level metrics
        "bleu", "rouge",  # Text similarity
        "semantic_similarity"  # BERT-based similarity
    ])
    
    # Statistical tests
    statistical_tests: List[str] = field(default_factory=lambda: [
        "paired_t_test",
        "wilcoxon_signed_rank",
        "anova"
    ])
    
    # Confidence intervals
    confidence_level: float = 0.95
    bootstrap_samples: int = 1000
    
    # Error analysis
    analyze_errors: bool = True
    error_categories: List[str] = field(default_factory=lambda: [
        "medical_terms",
        "numbers",
        "dates",
        "proper_nouns",
        "acronyms"
    ])
    
    # Visualization
    save_figures: bool = True
    figure_format: str = "png"
    figure_dpi: int = 300

@dataclass
class ExperimentConfig:
    """Experiment configuration."""
    name: str = "asr_comparison_german_medical"
    version: str = "1.0.0"
    description: str = "Comparison of ASR models on German medical conversations"
    
    # Execution
    max_samples: Optional[int] = None  # None for all samples
    use_multiprocessing: bool = True
    save_intermediate: bool = True
    verbose: bool = True
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = "experiment.log"

@dataclass
class Config:
    """Main configuration class."""
    paths: PathConfig
    dataset: DatasetConfig
    asr: ASRConfig
    evaluation: EvaluationConfig
    experiment: ExperimentConfig
    
    @classmethod
    def from_yaml(cls, config_path: Optional[str] = None):
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = Path(__file__).parent / "default_config.yaml"
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(
            paths=PathConfig(**config_dict.get('paths', {})),
            dataset=DatasetConfig(**config_dict.get('dataset', {})),
            asr=ASRConfig(**config_dict.get('asr', {})),
            evaluation=EvaluationConfig(**config_dict.get('evaluation', {})),
            experiment=ExperimentConfig(**config_dict.get('experiment', {}))
        )
    
    def save(self, path: Path):
        """Save configuration to YAML file."""
        config_dict = {
            'paths': self.paths.__dict__,
            'dataset': self.dataset.__dict__,
            'asr': self.asr.__dict__,
            'evaluation': self.evaluation.__dict__,
            'experiment': self.experiment.__dict__
        }
        
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

# Default configuration
config = Config(
    paths=PathConfig(),
    dataset=DatasetConfig(),
    asr=ASRConfig(),
    evaluation=EvaluationConfig(),
    experiment=ExperimentConfig()
)

# Initialize directories
config.paths.create_directories()