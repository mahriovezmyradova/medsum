#!/usr/bin/env python3
"""
Comprehensive Medical ASR Benchmarking Suite
Tests multiple models with medical term focus and performance metrics
Specifically optimized for German medical transcription needs
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import io
import soundfile as sf
from tqdm import tqdm
import warnings
import logging
import json
import time
import psutil
import os
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
import torch
import torchaudio
import re
import gc

# Auto-detect best device (MPS on Apple Silicon, CUDA on NVIDIA, else CPU)
def _get_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
DEVICE = _get_device()

warnings.filterwarnings('ignore')
logging.getLogger('absl').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.WARNING)
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('medical_asr_benchmark.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add project root to path so 'src' imports work
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# ==================== CONFIGURATION ====================
DATASET_PATH = Path("/Users/mahriovezmyradova/MultiMed_dataset/German")
OUTPUT_DIR = Path("data/outputs/full_dataset_analysis")  # Changed to match your structure
SAMPLE_LIMIT_PER_SPLIT = None  # Process 100 samples from each split for benchmarking
# Set to None for full dataset

# ==================== MEDICAL TERMS FOR GERMAN ====================
MEDICAL_TERMS = {
    # Core medical terminology for MTER calculation
    'diagnosis': ['diagnose', 'diagnostik', 'diagnostiziert', 'befund', 'befunde'],
    'therapy': ['therapie', 'behandlung', 'therapeutisch', 'medikation', 'therapien'],
    'medication': ['medikament', 'tablette', 'arznei', 'wirkstoff', 'dosis', 'medikamente'],
    'symptoms': ['schmerz', 'schmerzen', 'fieber', 'husten', 'übelkeit', 
                 'brechreiz', 'schwindel', 'müdigkeit', 'atemnot', 'kopfschmerzen'],
    'body_parts': ['herz', 'kopf', 'bauch', 'rücken', 'brust', 'arm', 'bein', 'hals'],
    'vitals': ['blutdruck', 'puls', 'temperatur', 'sauerstoffsättigung', 'atmen', 'herzfrequenz'],
    'conditions': ['infektion', 'entzündung', 'allergie', 'diabetes', 'hypertonie',
                   'asthma', 'migräne', 'arthrose', 'grippe', 'erkältung'],
    'procedures': ['operation', 'untersuchung', 'röntgen', 'mrt', 'ct', 'ultraschall', 'endoskopie'],
    'professionals': ['arzt', 'ärztin', 'pfleger', 'schwester', 'therapeut', 'chirurg'],
    # Medical phrases for context
    'phrases': ['wie geht es ihnen', 'haben sie schmerzen', 'nehmen sie die tablette',
                'blutdruck messen', 'krankmeldung', 'arbeitsunfähig', 'praxis', 'krankenhaus']
}

# Flatten for easy lookup
ALL_MEDICAL_TERMS = []
for category, terms in MEDICAL_TERMS.items():
    ALL_MEDICAL_TERMS.extend(terms)

# Create regex pattern for medical term detection
MEDICAL_PATTERN = re.compile(r'\b(' + '|'.join(ALL_MEDICAL_TERMS) + r')\b', re.IGNORECASE)

# ==================== MODEL CONFIGURATIONS ====================
MODEL_CONFIGS = {
    # Whisper variants (all sizes for comparison)
    'whisper_tiny': {
        'type': 'whisper',
        'model_size': 'tiny',
        'description': 'Whisper Tiny - Fastest, smallest (39M params)',
        'size_mb': 75,
        'params_m': 39,
        'relative_speed': 10  # Baseline for comparison
    },
    'whisper_base': {
        'type': 'whisper',
        'model_size': 'base',
        'description': 'Whisper Base - Balanced (74M params)',
        'size_mb': 142,
        'params_m': 74,
        'relative_speed': 5
    },
    'whisper_small': {
        'type': 'whisper',
        'model_size': 'small',
        'description': 'Whisper Small - Good accuracy (244M params)',
        'size_mb': 466,
        'params_m': 244,
        'relative_speed': 2
    },
    
    # Wav2Vec2 German variants
    'wav2vec2_base': {
        'type': 'wav2vec2',
        'model_name': 'facebook/wav2vec2-base-10k-voxpopuli',
        'description': 'Wav2Vec2 Base (VoxPopuli) - 95M params',
        'size_mb': 360,
        'params_m': 95,
        'relative_speed': 8
    },
    'wav2vec2_large_xlsr': {
        'type': 'wav2vec2',
        'model_name': 'jonatasgrosman/wav2vec2-large-xlsr-53-german',
        'description': 'Wav2Vec2 Large XLSR-53 German - Fine-tuned for German',
        'size_mb': 1260,
        'params_m': 315,
        'relative_speed': 3
    },
    'wav2vec2_oliver': {
        'type': 'wav2vec2',
        'model_name': 'oliverguhr/wav2vec2-large-xlsr-53-german-cv8',
        'description': 'Wav2Vec2 Large (Common Voice 8) - Optimized for German',
        'size_mb': 1260,
        'params_m': 315,
        'relative_speed': 3
    }
}

# Models to test - exactly as you specified
MODELS_TO_TEST = [
    'whisper_tiny',
    'whisper_base',
    'whisper_small',
    'wav2vec2_base',
    'wav2vec2_large_xlsr',
    'wav2vec2_oliver'
]

# ==================== DATA CLASSES ====================
@dataclass
class BenchmarkResult:
    """Comprehensive benchmark result for a single sample."""
    model_name: str
    split: str
    sample_id: int
    reference_text: str
    asr_text: str
    wer: float
    medical_wer: float  # WER on medical terms only
    processing_time: float
    confidence: float
    audio_duration: float
    word_count: int
    medical_term_count: int
    medical_terms_found: List[str]
    peak_memory_mb: float
    cpu_percent: float
    error: Optional[str] = None
    
    def to_dict(self):
        """Convert to dictionary, handling list fields."""
        d = asdict(self)
        d['medical_terms_found'] = ','.join(self.medical_terms_found)
        return d

# ==================== HELPER FUNCTIONS ====================

def load_audio_from_item(audio_item):
    """Load audio from parquet item."""
    try:
        if isinstance(audio_item, dict) and 'bytes' in audio_item:
            audio_bytes = audio_item['bytes']
            audio_file = io.BytesIO(audio_bytes)
            array, sr = sf.read(audio_file)
            
            # Convert to mono if stereo
            if len(array.shape) > 1:
                array = np.mean(array, axis=1)
            
            # Convert to float32 and normalize
            array = array.astype(np.float32)
            array_max = np.max(np.abs(array))
            if array_max > 0:
                array = array / array_max
            
            return array, sr
        return None, None
    except Exception as e:
        logger.error(f"Failed to load audio: {e}")
        return None, None

def calculate_wer(reference, hypothesis):
    """Calculate Word Error Rate."""
    try:
        import jiwer
        
        def clean_text(text):
            if not isinstance(text, str):
                text = str(text)
            text = text.lower()
            import string
            text = text.translate(str.maketrans('', '', string.punctuation))
            text = ' '.join(text.split())
            return text
        
        ref_clean = clean_text(reference)
        hyp_clean = clean_text(hypothesis)
        
        return jiwer.wer(ref_clean, hyp_clean)
        
    except ImportError:
        # Simple WER calculation
        if not isinstance(reference, str):
            reference = str(reference)
        if not isinstance(hypothesis, str):
            hypothesis = str(hypothesis)
            
        ref_words = reference.lower().split()
        hyp_words = hypothesis.lower().split()
        
        n, m = len(ref_words), len(hyp_words)
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        
        for i in range(n + 1):
            dp[i][0] = i
        for j in range(m + 1):
            dp[0][j] = j
            
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if ref_words[i-1] == hyp_words[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(
                        dp[i-1][j] + 1,
                        dp[i][j-1] + 1,
                        dp[i-1][j-1] + 1
                    )
        
        errors = dp[n][m]
        return errors / max(n, 1)

def calculate_medical_wer(reference, hypothesis, medical_terms):
    """Calculate WER only on medical terms."""
    if not isinstance(reference, str):
        reference = str(reference)
    if not isinstance(hypothesis, str):
        hypothesis = str(hypothesis)
        
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    
    # Extract only medical terms
    ref_medical = [w for w in ref_words if any(term in w for term in medical_terms)]
    hyp_medical = [w for w in hyp_words if any(term in w for term in medical_terms)]
    
    if not ref_medical:
        return 0.0  # No medical terms in reference
    
    # Calculate WER on medical terms only
    return calculate_wer(' '.join(ref_medical), ' '.join(hyp_medical))

def extract_medical_terms(text):
    """Extract medical terms from text."""
    if not isinstance(text, str):
        text = str(text)
    matches = MEDICAL_PATTERN.findall(text.lower())
    return list(set(matches))  # Unique terms

def get_memory_usage():
    """Get current memory usage."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

def estimate_model_size_mb(model_id):
    """Estimate model size on disk."""
    config = MODEL_CONFIGS.get(model_id, {})
    return config.get('size_mb', 0)

# ==================== MODEL INITIALIZATION ====================

def initialize_model(model_id, config):
    """Initialize ASR model based on configuration."""
    try:
        if config['type'] == 'whisper':
            from src.asr.whisper import WhisperASR

            model_config = {
                "name": model_id,
                "model_size": config['model_size'],
                "language": "de",
                "device": DEVICE
            }
            model = WhisperASR(model_config)

        elif config['type'] == 'wav2vec2':
            from src.asr.wav2vec2 import Wav2Vec2ASR

            model_config = {
                "name": model_id,
                "model_name": config['model_name'],
                "language": "de",
                "device": DEVICE,
                "cache_dir": str(Path.home() / ".cache" / "huggingface")
            }
            model = Wav2Vec2ASR(model_config)
            
        else:
            logger.error(f"Unknown model type: {config['type']}")
            return None
        
        # Measure load time and memory
        start_mem = get_memory_usage()
        start_time = time.time()
        
        model.load_model()
        
        load_time = time.time() - start_time
        end_mem = get_memory_usage()
        memory_increase = end_mem - start_mem
        
        logger.info(f"✓ Loaded {model_id} in {load_time:.2f}s, Memory: +{memory_increase:.1f}MB")
        
        return model
        
    except Exception as e:
        logger.error(f"Failed to initialize {model_id}: {e}")
        return None

# ==================== BENCHMARKING ====================

def benchmark_model(model_id, model, audio_samples, split_name):
    """Run comprehensive benchmark for a single model."""
    results = []
    
    logger.info(f"\nBenchmarking {model_id} on {split_name}...")
    
    for idx, sample in enumerate(tqdm(audio_samples, desc=f"{model_id}")):
        audio_array, sr, ref_text = sample
        
        try:
            # Pre-benchmark metrics
            start_time = time.time()
            start_mem = get_memory_usage()
            cpu_start = psutil.cpu_percent(interval=None)
            
            # Transcribe
            result = model.transcribe(audio_array, sr)
            
            # Post-benchmark metrics
            end_time = time.time()
            end_mem = get_memory_usage()
            cpu_end = psutil.cpu_percent(interval=None)
            
            # Calculate metrics
            wer = calculate_wer(ref_text, result.text)
            medical_wer = calculate_medical_wer(ref_text, result.text, ALL_MEDICAL_TERMS)
            medical_terms_found = extract_medical_terms(ref_text)
            
            # Audio duration
            audio_duration = len(audio_array) / sr if sr > 0 else 0
            
            # Create result
            benchmark = BenchmarkResult(
                model_name=model_id,
                split=split_name,
                sample_id=idx,
                reference_text=ref_text,
                asr_text=result.text,
                wer=wer,
                medical_wer=medical_wer,
                processing_time=end_time - start_time,
                confidence=result.confidence if hasattr(result, 'confidence') else 0.0,
                audio_duration=audio_duration,
                word_count=len(ref_text.split()),
                medical_term_count=len(medical_terms_found),
                medical_terms_found=medical_terms_found,
                peak_memory_mb=end_mem - start_mem,
                cpu_percent=(cpu_start + cpu_end) / 2
            )
            
            results.append(benchmark)
            
        except Exception as e:
            logger.error(f"Error processing sample {idx}: {e}")
            results.append(BenchmarkResult(
                model_name=model_id,
                split=split_name,
                sample_id=idx,
                reference_text=ref_text,
                asr_text="",
                wer=1.0,
                medical_wer=1.0,
                processing_time=0,
                confidence=0,
                audio_duration=0,
                word_count=len(ref_text.split()),
                medical_term_count=0,
                medical_terms_found=[],
                peak_memory_mb=0,
                cpu_percent=0,
                error=str(e)
            ))
    
    return results

# ==================== ANALYSIS FUNCTIONS ====================

def analyze_results(all_results):
    """Comprehensive analysis of benchmark results."""
    # Convert to DataFrame
    records = []
    for r in all_results:
        d = r.to_dict()
        records.append(d)
    
    df = pd.DataFrame(records)
    
    analysis = {
        'overall': {},
        'per_model': {},
        'per_split': {},
        'medical_analysis': {},
        'performance': {},
        'recommendations': []
    }
    
    # Overall statistics
    analysis['overall'] = {
        'total_samples': len(df),
        'total_medical_terms': int(df['medical_term_count'].sum()) if 'medical_term_count' in df else 0,
        'models_tested': df['model_name'].unique().tolist(),
        'splits': df['split'].unique().tolist()
    }
    
    # Per-model analysis
    for model in df['model_name'].unique():
        model_df = df[df['model_name'] == model]
        
        # Calculate inference time for different audio durations
        short_audio = model_df[model_df['audio_duration'] < 300]  # <5 minutes
        long_audio = model_df[model_df['audio_duration'] >= 300]  # >=5 minutes
        
        analysis['per_model'][model] = {
            'samples': len(model_df),
            'avg_wer': float(model_df['wer'].mean()),
            'std_wer': float(model_df['wer'].std()),
            'avg_medical_wer': float(model_df['medical_wer'].mean()),
            'std_medical_wer': float(model_df['medical_wer'].std()),
            'avg_processing_time': float(model_df['processing_time'].mean()),
            'avg_confidence': float(model_df['confidence'].mean()) if 'confidence' in model_df else 0,
            'avg_memory_mb': float(model_df['peak_memory_mb'].mean()),
            'avg_cpu_percent': float(model_df['cpu_percent'].mean()),
            'inference_time_5min': float(short_audio['processing_time'].mean()) if len(short_audio) > 0 else 0,
            'inference_time_15min': float(long_audio['processing_time'].mean()) if len(long_audio) > 0 else 0,
            'model_size_mb': estimate_model_size_mb(model),
            'config': MODEL_CONFIGS.get(model, {})
        }
    
    # Medical term accuracy analysis
    if 'medical_term_count' in df.columns:
        medical_df = df[df['medical_term_count'] > 0]
        if len(medical_df) > 0:
            for model in df['model_name'].unique():
                model_medical_df = medical_df[medical_df['model_name'] == model]
                if len(model_medical_df) > 0:
                    analysis['medical_analysis'][model] = {
                        'samples_with_medical_terms': int(len(model_medical_df)),
                        'avg_medical_wer': float(model_medical_df['medical_wer'].mean()),
                        'total_medical_terms': int(model_medical_df['medical_term_count'].sum())
                    }
    
    # Performance analysis (speed/accuracy trade-off)
    for model in df['model_name'].unique():
        model_df = df[df['model_name'] == model]
        avg_wer = model_df['wer'].mean()
        avg_time = model_df['processing_time'].mean()
        
        analysis['performance'][model] = {
            'wer_per_second': float(avg_wer / avg_time if avg_time > 0 else 0),
            'words_per_second': float(model_df['word_count'].mean() / avg_time if avg_time > 0 else 0),
            'medical_wer_per_second': float(model_df['medical_wer'].mean() / avg_time if avg_time > 0 and model_df['medical_wer'].mean() > 0 else 0),
            'accuracy_speed_tradeoff': float((1 - avg_wer) / avg_time if avg_time > 0 else 0)  # Higher is better
        }
    
    # Generate recommendations
    # Best overall accuracy
    best_wer_model = df.groupby('model_name')['wer'].mean().idxmin()
    best_wer = df.groupby('model_name')['wer'].mean().min()
    
    # Best medical term accuracy
    if 'medical_wer' in df.columns:
        medical_means = df.groupby('model_name')['medical_wer'].mean()
        if len(medical_means) > 0:
            best_medical_model = medical_means.idxmin()
            best_medical_wer = medical_means.min()
        else:
            best_medical_model = best_wer_model
            best_medical_wer = best_wer
    else:
        best_medical_model = best_wer_model
        best_medical_wer = best_wer
    
    # Fastest
    fastest_model = df.groupby('model_name')['processing_time'].mean().idxmin()
    fastest_time = df.groupby('model_name')['processing_time'].mean().min()
    
    # Smallest memory footprint
    smallest_memory = df.groupby('model_name')['peak_memory_mb'].mean().idxmin()
    smallest_memory_val = df.groupby('model_name')['peak_memory_mb'].mean().min()
    
    # Best for medical terms per second (efficiency)
    perf_df = pd.DataFrame(analysis['performance']).T
    if 'medical_wer_per_second' in perf_df.columns:
        best_medical_efficiency = perf_df['medical_wer_per_second'].idxmax()
    else:
        best_medical_efficiency = fastest_model
    
    analysis['recommendations'] = [
        f"🏆 Best Overall Accuracy: {best_wer_model} (WER: {best_wer:.3f})",
        f"🩺 Best Medical Term Accuracy: {best_medical_model} (Medical WER: {best_medical_wer:.3f})",
        f"⚡ Fastest Inference: {fastest_model} ({fastest_time:.2f}s per sample)",
        f"💾 Smallest Memory: {smallest_memory} ({smallest_memory_val:.1f}MB peak)",
        f"🎯 Best Medical Term Efficiency: {best_medical_efficiency} (Medical WER per second)"
    ]
    
    return analysis, df

def generate_report(analysis, df, output_dir):
    """Generate comprehensive HTML report."""
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Medical ASR Benchmark Report</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; background-color: #f5f5f5; }}
            .container {{ max-width: 1400px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }}
            h1, h2 {{ color: #2c3e50; }}
            h1 {{ border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; font-size: 14px; }}
            th {{ background-color: #3498db; color: white; padding: 12px; text-align: left; }}
            td {{ padding: 10px; border-bottom: 1px solid #ddd; }}
            tr:hover {{ background-color: #f5f5f5; }}
            .best {{ background-color: #d4edda; font-weight: bold; }}
            .metric {{ font-size: 24px; font-weight: bold; color: #3498db; }}
            .recommendation {{ background-color: #fff3cd; padding: 15px; border-radius: 5px; margin: 10px 0; border-left: 5px solid #ffc107; }}
            .badge {{ display: inline-block; padding: 5px 10px; border-radius: 3px; color: white; font-size: 12px; margin-right: 5px; }}
            .badge-accuracy {{ background-color: #28a745; }}
            .badge-speed {{ background-color: #007bff; }}
            .badge-medical {{ background-color: #dc3545; }}
            .badge-memory {{ background-color: #6c757d; }}
            .summary-box {{ display: inline-block; padding: 15px; margin: 10px; background-color: #e9ecef; border-radius: 5px; min-width: 200px; }}
            .summary-box .value {{ font-size: 28px; font-weight: bold; color: #007bff; }}
            .summary-box .label {{ font-size: 14px; color: #6c757d; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🏥 Medical ASR Benchmark Report</h1>
            <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            
            <div style="display: flex; flex-wrap: wrap;">
                <div class="summary-box">
                    <div class="value">{analysis['overall']['total_samples']}</div>
                    <div class="label">Total Samples</div>
                </div>
                <div class="summary-box">
                    <div class="value">{analysis['overall']['total_medical_terms']}</div>
                    <div class="label">Medical Terms Found</div>
                </div>
                <div class="summary-box">
                    <div class="value">{len(analysis['overall']['models_tested'])}</div>
                    <div class="label">Models Tested</div>
                </div>
            </div>
            
            <h2>📊 Model Performance Comparison</h2>
            <table>
                <tr>
                    <th>Model</th>
                    <th>Type</th>
                    <th>Overall WER</th>
                    <th>Medical WER</th>
                    <th>Time (s)</th>
                    <th>Memory (MB)</th>
                    <th>Size (MB)</th>
                    <th>Confidence</th>
                </tr>
    """
    
    # Find best values for highlighting
    best_wer = min([stats['avg_wer'] for stats in analysis['per_model'].values()])
    best_medical_wer = min([stats['avg_medical_wer'] for stats in analysis['per_model'].values()])
    best_time = min([stats['avg_processing_time'] for stats in analysis['per_model'].values()])
    best_memory = min([stats['avg_memory_mb'] for stats in analysis['per_model'].values()])
    
    # Add model rows
    for model, stats in analysis['per_model'].items():
        config = stats.get('config', {})
        
        html_content += f"""
                <tr>
                    <td><strong>{model}</strong><br><small>{config.get('description', '')}</small></td>
                    <td>{config.get('type', 'N/A')}</td>
                    <td{' class="best"' if stats['avg_wer'] == best_wer else ''}>{stats['avg_wer']:.3f} (±{stats['std_wer']:.3f})</td>
                    <td{' class="best"' if stats['avg_medical_wer'] == best_medical_wer else ''}>{stats['avg_medical_wer']:.3f} (±{stats['std_medical_wer']:.3f})</td>
                    <td{' class="best"' if stats['avg_processing_time'] == best_time else ''}>{stats['avg_processing_time']:.2f}</td>
                    <td{' class="best"' if stats['avg_memory_mb'] == best_memory else ''}>{stats['avg_memory_mb']:.1f}</td>
                    <td>{stats['model_size_mb']}</td>
                    <td>{stats['avg_confidence']:.2f}</td>
                </tr>
        """
    
    # Inference time by audio duration
    html_content += """
            </table>
            
            <h2>⏱️ Inference Time by Audio Duration</h2>
            <table>
                <tr>
                    <th>Model</th>
                    <th>5-min Audio (s)</th>
                    <th>15-min Audio (s)</th>
                    <th>Real-time Factor (5min)</th>
                    <th>Real-time Factor (15min)</th>
                </tr>
    """
    
    for model, stats in analysis['per_model'].items():
        rtf_5min = stats['inference_time_5min'] / 300 if stats['inference_time_5min'] > 0 else 0
        rtf_15min = stats['inference_time_15min'] / 900 if stats['inference_time_15min'] > 0 else 0
        
        html_content += f"""
                <tr>
                    <td>{model}</td>
                    <td>{stats['inference_time_5min']:.2f}</td>
                    <td>{stats['inference_time_15min']:.2f}</td>
                    <td>{rtf_5min:.2f}x</td>
                    <td>{rtf_15min:.2f}x</td>
                </tr>
        """
    
    # Medical term analysis
    if analysis.get('medical_analysis'):
        html_content += """
            </table>
            
            <h2>🩺 Medical Term Accuracy</h2>
            <table>
                <tr>
                    <th>Model</th>
                    <th>Samples with Medical Terms</th>
                    <th>Medical WER</th>
                    <th>Total Medical Terms</th>
                    <th>Medical Term Accuracy</th>
                </tr>
        """
        
        for model, stats in analysis.get('medical_analysis', {}).items():
            medical_accuracy = 1 - stats['avg_medical_wer']
            html_content += f"""
                <tr>
                    <td>{model}</td>
                    <td>{stats['samples_with_medical_terms']}</td>
                    <td>{stats['avg_medical_wer']:.3f}</td>
                    <td>{stats['total_medical_terms']}</td>
                    <td>{medical_accuracy:.1%}</td>
                </tr>
            """
    
    # Performance metrics
    html_content += """
            </table>
            
            <h2>⚡ Performance Efficiency</h2>
            <table>
                <tr>
                    <th>Model</th>
                    <th>WER/Second</th>
                    <th>Words/Second</th>
                    <th>Medical WER/Second</th>
                    <th>Accuracy/Speed Trade-off</th>
                </tr>
    """
    
    for model, stats in analysis['performance'].items():
        html_content += f"""
                <tr>
                    <td>{model}</td>
                    <td>{stats['wer_per_second']:.4f}</td>
                    <td>{stats['words_per_second']:.1f}</td>
                    <td>{stats['medical_wer_per_second']:.4f}</td>
                    <td>{stats['accuracy_speed_tradeoff']:.4f}</td>
                </tr>
        """
    
    # Recommendations
    html_content += """
            </table>
            
            <h2>🎯 Recommendations for Your Use Case</h2>
    """
    
    for i, rec in enumerate(analysis['recommendations']):
        badge_class = ['badge-accuracy', 'badge-medical', 'badge-speed', 'badge-memory', 'badge-accuracy'][i % 5]
        html_content += f"""
            <div class="recommendation">
                <span class="badge {badge_class}">{rec.split()[0]}</span> {rec}
            </div>
        """
    
    # Top errors for manual review
    html_content += """
            <h2>🔍 Top 50 Errors for Manual Review</h2>
            <p>These samples show the highest WER - useful for error analysis</p>
            <table>
                <tr>
                    <th>Model</th>
                    <th>Split</th>
                    <th>WER</th>
                    <th>Medical Terms</th>
                    <th>Reference (first 100 chars)</th>
                    <th>ASR Output (first 100 chars)</th>
                </tr>
    """
    
    # Get top 50 errors
    if len(df) > 0:
        top_errors = df.nlargest(50, 'wer')[['model_name', 'split', 'wer', 'medical_term_count', 
                                             'reference_text', 'asr_text']].to_dict('records')
        
        for error in top_errors[:20]:  # Show first 20 in report
            ref_preview = error['reference_text'][:100] + "..." if len(error['reference_text']) > 100 else error['reference_text']
            asr_preview = error['asr_text'][:100] + "..." if len(error['asr_text']) > 100 else error['asr_text']
            html_content += f"""
                <tr>
                    <td>{error['model_name']}</td>
                    <td>{error['split']}</td>
                    <td>{error['wer']:.3f}</td>
                    <td>{int(error['medical_term_count'])}</td>
                    <td>{ref_preview}</td>
                    <td>{asr_preview}</td>
                </tr>
            """
    
    # Model details
    html_content += """
            </table>
            
            <h2>📦 Model Specifications</h2>
            <table>
                <tr>
                    <th>Model</th>
                    <th>Type</th>
                    <th>Size (MB)</th>
                    <th>Parameters (M)</th>
                    <th>Relative Speed</th>
                    <th>Description</th>
                </tr>
    """
    
    for model_id in MODELS_TO_TEST:
        config = MODEL_CONFIGS.get(model_id, {})
        html_content += f"""
                <tr>
                    <td>{model_id}</td>
                    <td>{config.get('type', 'N/A')}</td>
                    <td>{config.get('size_mb', 'N/A')}</td>
                    <td>{config.get('params_m', 'N/A')}</td>
                    <td>{config.get('relative_speed', 'N/A')}x</td>
                    <td>{config.get('description', 'N/A')}</td>
                </tr>
        """
    
    html_content += """
            </table>
            
            <h2>📁 Output Files</h2>
            <ul>
                <li><strong>all_transcriptions.csv</strong> - Complete results with all transcriptions</li>
                <li><strong>dataset_analysis.json</strong> - Comprehensive analysis in JSON format</li>
                <li><strong>analysis_summary.csv</strong> - Summary statistics by model</li>
                <li><strong>manual_review_data.csv</strong> - Data formatted for manual error analysis</li>
                <li><strong>top_errors_for_review.csv</strong> - Top 50 errors for focused review</li>
                <li><strong>train/</strong> - Split-specific results for training data</li>
                <li><strong>eval/</strong> - Split-specific results for evaluation data</li>
                <li><strong>test/</strong> - Split-specific results for test data</li>
            </ul>
            
            <p><i>Generated for Medical ASR Thesis Research - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</i></p>
        </div>
    </body>
    </html>
    """
    
    report_path = output_dir / "medical_asr_benchmark_report.html"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return report_path

def create_manual_review_files(df, output_dir):
    """Create CSV files formatted for manual error analysis."""
    
    # All results with medical term focus
    manual_review = df[['model_name', 'split', 'sample_id', 'wer', 'medical_wer', 
                        'medical_term_count', 'reference_text', 'asr_text', 
                        'processing_time', 'audio_duration']].copy()
    
    # Add columns for manual review
    manual_review['correct'] = ''  # To be filled manually
    manual_review['notes'] = ''     # For reviewer comments
    manual_review['medical_term_accuracy'] = manual_review['medical_wer'].apply(lambda x: 1 - x)
    
    # Sort by medical WER (highest errors first) for focused review
    manual_review = manual_review.sort_values('medical_wer', ascending=False)
    
    # Save for manual review
    manual_review.to_csv(output_dir / "manual_review_data.csv", index=False)
    
    # Top 50 errors for quick review
    top_errors = manual_review.nlargest(50, 'wer')[['model_name', 'split', 'wer', 'medical_wer', 
                                                    'medical_term_count', 'reference_text', 'asr_text']]
    top_errors.to_csv(output_dir / "top_errors_for_review.csv", index=False)
    
    # Summary by model and split
    summary = df.groupby(['model_name', 'split']).agg({
        'wer': ['mean', 'std', 'min', 'max'],
        'medical_wer': ['mean', 'std'],
        'processing_time': 'mean',
        'peak_memory_mb': 'mean',
        'sample_id': 'count'
    }).round(4)
    
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.rename(columns={'sample_id_count': 'samples'})
    summary.to_csv(output_dir / "analysis_summary.csv")
    
    return manual_review, top_errors, summary

# ==================== MEDICAL TERM CATEGORY ANALYSIS ====================

def analyze_medical_terms_by_category(df):
    """Analyze how each model performs on different medical term categories."""
    
    results = []
    
    for model in df['model_name'].unique():
        model_df = df[df['model_name'] == model]
        
        model_results = {'model': model}
        
        for category, terms in MEDICAL_TERMS.items():
            # Filter samples containing these terms
            mask = model_df['reference_text'].str.lower().str.contains('|'.join(terms), na=False)
            samples_with_terms = model_df[mask]
            
            if len(samples_with_terms) > 0:
                avg_wer = samples_with_terms['wer'].mean()
                avg_medical_wer = samples_with_terms['medical_wer'].mean()
                
                model_results[f'{category}_samples'] = len(samples_with_terms)
                model_results[f'{category}_wer'] = avg_wer
                model_results[f'{category}_medical_wer'] = avg_medical_wer
                model_results[f'{category}_accuracy'] = 1 - avg_medical_wer
            else:
                model_results[f'{category}_samples'] = 0
                model_results[f'{category}_wer'] = 0
                model_results[f'{category}_medical_wer'] = 0
                model_results[f'{category}_accuracy'] = 0
        
        results.append(model_results)
    
    category_df = pd.DataFrame(results)
    category_df.to_csv(OUTPUT_DIR / "medical_term_categories.csv", index=False)
    
    return category_df

# ==================== MAIN EXECUTION ====================

def main():
    print("=" * 80)
    print("🏥 MEDICAL ASR BENCHMARKING SUITE")
    print("=" * 80)
    print(f"Dataset: {DATASET_PATH}")
    print(f"Models to test: {len(MODELS_TO_TEST)}")
    for model_id in MODELS_TO_TEST:
        config = MODEL_CONFIGS.get(model_id, {})
        print(f"  • {model_id}: {config.get('description', '')}")
    print(f"Samples per split: {SAMPLE_LIMIT_PER_SPLIT}")
    print(f"Medical terms tracked: {len(ALL_MEDICAL_TERMS)}")
    print("=" * 80)
    
    # Create output directory structure
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for split in ['train', 'eval', 'test']:
        (OUTPUT_DIR / split).mkdir(parents=True, exist_ok=True)
    
    # Load dataset splits
    splits = ['train', 'eval', 'test']
    audio_samples_by_split = {}
    
    for split in splits:
        file_path = DATASET_PATH / f"{split}-00000-of-00001.parquet"
        
        if not file_path.exists():
            logger.warning(f"Split not found: {split}")
            continue
        
        logger.info(f"\nLoading {split} split...")
        df = pd.read_parquet(file_path)
        
        # Limit samples
        if SAMPLE_LIMIT_PER_SPLIT and len(df) > SAMPLE_LIMIT_PER_SPLIT:
            df = df.sample(n=SAMPLE_LIMIT_PER_SPLIT, random_state=42)
        
        # Prepare audio samples
        samples = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Loading {split}"):
            audio_item = row['audio']
            text = str(row['text'])
            
            audio_array, sr = load_audio_from_item(audio_item)
            
            if audio_array is not None:
                samples.append((audio_array, sr, text))
            else:
                logger.warning(f"Failed to load audio for sample {idx}")
        
        audio_samples_by_split[split] = samples
        logger.info(f"✓ Loaded {len(samples)} samples from {split}")
    
    if not audio_samples_by_split:
        logger.error("No audio samples loaded!")
        return
    
    # Initialize and benchmark each model
    all_results = []
    
    for model_id in MODELS_TO_TEST:
        config = MODEL_CONFIGS.get(model_id)
        if not config:
            logger.warning(f"Unknown model: {model_id}")
            continue
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing: {model_id}")
        logger.info(f"{'='*60}")
        
        # Initialize model
        model = initialize_model(model_id, config)
        if not model:
            continue
        
        # Benchmark on each split
        for split_name, samples in audio_samples_by_split.items():
            results = benchmark_model(model_id, model, samples, split_name)
            all_results.extend(results)
            
            # Save intermediate results
            split_df = pd.DataFrame([r.to_dict() for r in results])
            split_output = OUTPUT_DIR / split_name / f"{model_id}_results.csv"
            split_output.parent.mkdir(parents=True, exist_ok=True)
            split_df.to_csv(split_output, index=False)
            
            # Print summary
            avg_wer = split_df['wer'].mean()
            avg_medical_wer = split_df['medical_wer'].mean()
            avg_time = split_df['processing_time'].mean()
            
            logger.info(f"\n{model_id} on {split_name}:")
            logger.info(f"  WER: {avg_wer:.3f}")
            logger.info(f"  Medical WER: {avg_medical_wer:.3f}")
            logger.info(f"  Time: {avg_time:.2f}s")
        
        # Clean up model to free memory
        if hasattr(model, 'cleanup'):
            model.cleanup()
        del model
        gc.collect()
    
    # Generate comprehensive analysis
    logger.info(f"\n{'='*60}")
    logger.info("Generating final analysis...")
    logger.info(f"{'='*60}")
    
    analysis, df = analyze_results(all_results)
    
    # Save all results (all_transcriptions.csv)
    all_results_df = pd.DataFrame([r.to_dict() for r in all_results])
    all_results_path = OUTPUT_DIR / "all_transcriptions.csv"
    all_results_df.to_csv(all_results_path, index=False)
    
    # Save analysis as JSON (dataset_analysis.json)
    analysis_path = OUTPUT_DIR / "dataset_analysis.json"
    with open(analysis_path, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    
    # Create manual review files
    manual_review, top_errors, summary = create_manual_review_files(all_results_df, OUTPUT_DIR)
    
    # Generate HTML report
    report_path = generate_report(analysis, all_results_df, OUTPUT_DIR)
    
    # Medical term category analysis
    category_df = analyze_medical_terms_by_category(all_results_df)
    
    # Print final summary
    print("\n" + "=" * 80)
    print("🏁 BENCHMARKING COMPLETE")
    print("=" * 80)
    
    print(f"\n📊 Results Summary:")
    print(f"  • Total samples processed: {len(all_results_df)}")
    print(f"  • Models tested: {len(MODELS_TO_TEST)}")
    print(f"  • Medical terms found: {all_results_df['medical_term_count'].sum()}")
    
    print(f"\n🏆 Winners by Category:")
    for rec in analysis['recommendations']:
        print(f"  {rec}")
    
    print(f"\n📁 Output Files (all in {OUTPUT_DIR}):")
    print(f"  • All transcriptions: all_transcriptions.csv")
    print(f"  • Analysis JSON: dataset_analysis.json")
    print(f"  • Summary stats: analysis_summary.csv")
    print(f"  • Manual review: manual_review_data.csv")
    print(f"  • Top 50 errors: top_errors_for_review.csv")
    print(f"  • Medical categories: medical_term_categories.csv")
    print(f"  • HTML Report: {report_path.name}")
    print(f"  • Split results: train/, eval/, test/ directories")
    
    print(f"\n🔍 Next Steps:")
    print(f"  1. Open {report_path.name} in browser for interactive analysis")
    print(f"  2. Review top_errors_for_review.csv for error patterns")
    print(f"  3. Use manual_review_data.csv for detailed annotation")
    print(f"  4. Check medical_term_categories.csv for domain-specific performance")
    
    print("\n" + "=" * 80)

# ==================== AUDIO CONCATENATION FUNCTIONS ====================

def combine_audio_segments(audio_samples, target_duration=300):  # 300 seconds = 5 minutes
    """
    Combine multiple audio samples into segments of target duration.
    
    Args:
        audio_samples: List of (audio_array, sample_rate, text) tuples
        target_duration: Target duration in seconds (default 5 minutes = 300s)
    
    Returns:
        List of combined audio segments with concatenated texts
    """
    combined_segments = []
    
    if not audio_samples:
        return combined_segments
    
    current_segment = []
    current_duration = 0
    current_texts = []
    current_sample_rate = audio_samples[0][1]  # Use first sample's sample rate
    
    for audio_array, sr, text in tqdm(audio_samples, desc="Combining audio"):
        # Ensure same sample rate
        if sr != current_sample_rate:
            # Resample if needed
            audio_array = resample_audio(audio_array, sr, current_sample_rate)
        
        duration = len(audio_array) / current_sample_rate
        
        # If adding this sample would exceed target, save current segment and start new
        if current_duration + duration > target_duration and current_segment:
            # Save current segment
            combined = np.concatenate(current_segment)
            combined_text = ' '.join(current_texts)
            
            combined_segments.append({
                'audio': combined,
                'sample_rate': current_sample_rate,
                'text': combined_text,
                'duration': current_duration,
                'num_utterances': len(current_segment),
                'original_texts': current_texts.copy()
            })
            
            # Start new segment
            current_segment = [audio_array]
            current_texts = [text]
            current_duration = duration
        else:
            # Add to current segment
            current_segment.append(audio_array)
            current_texts.append(text)
            current_duration += duration
    
    # Add the last segment if it has content
    if current_segment:
        combined = np.concatenate(current_segment)
        combined_text = ' '.join(current_texts)
        
        combined_segments.append({
            'audio': combined,
            'sample_rate': current_sample_rate,
            'text': combined_text,
            'duration': current_duration,
            'num_utterances': len(current_segment),
            'original_texts': current_texts.copy()
        })
    
    logger.info(f"Created {len(combined_segments)} segments of ~{target_duration}s each")
    return combined_segments

def resample_audio(audio_array, orig_sr, target_sr):
    """Resample audio to target sample rate."""
    if orig_sr == target_sr:
        return audio_array
    
    audio_tensor = torch.FloatTensor(audio_array).unsqueeze(0)
    resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
    resampled = resampler(audio_tensor)
    return resampled.squeeze().numpy()

def create_varied_duration_segments(audio_samples, durations=[60, 300, 900]):
    """
    Create segments of different durations (1min, 5min, 15min) for benchmarking.
    
    Args:
        audio_samples: List of (audio_array, sample_rate, text) tuples
        durations: List of target durations in seconds
    
    Returns:
        Dictionary with duration as key and list of segments as value
    """
    segments_by_duration = {}
    
    for target_duration in durations:
        logger.info(f"\nCreating {target_duration//60}-minute segments...")
        segments = combine_audio_segments(audio_samples, target_duration)
        segments_by_duration[target_duration] = segments
    
    return segments_by_duration

def benchmark_with_varied_durations(model_id, model, segments_by_duration, split_name):
    """
    Benchmark model on segments of different durations.
    """
    all_results = []
    
    for duration, segments in segments_by_duration.items():
        logger.info(f"\nBenchmarking {model_id} on {duration//60}-minute segments ({len(segments)} segments)...")
        
        for idx, segment in enumerate(tqdm(segments, desc=f"{model_id} ({duration//60}min)")):
            try:
                # Pre-benchmark metrics
                start_time = time.time()
                start_mem = get_memory_usage()
                cpu_start = psutil.cpu_percent(interval=None)
                
                # Transcribe the combined audio
                result = model.transcribe(segment['audio'], segment['sample_rate'])
                
                # Post-benchmark metrics
                end_time = time.time()
                end_mem = get_memory_usage()
                cpu_end = psutil.cpu_percent(interval=None)
                
                # Calculate metrics against the combined reference text
                wer = calculate_wer(segment['text'], result.text)
                medical_wer = calculate_medical_wer(segment['text'], result.text, ALL_MEDICAL_TERMS)
                medical_terms_found = extract_medical_terms(segment['text'])
                
                # Create result
                benchmark = BenchmarkResult(
                    model_name=model_id,
                    split=f"{split_name}_{duration//60}min",
                    sample_id=idx,
                    reference_text=segment['text'][:1000],
                    asr_text=result.text[:1000],
                    wer=wer,
                    medical_wer=medical_wer,
                    processing_time=end_time - start_time,
                    confidence=result.confidence if hasattr(result, 'confidence') else 0.0,
                    audio_duration=segment['duration'],
                    word_count=len(segment['text'].split()),
                    medical_term_count=len(medical_terms_found),
                    medical_terms_found=medical_terms_found,
                    peak_memory_mb=end_mem - start_mem,
                    cpu_percent=(cpu_start + cpu_end) / 2
                )
                
                all_results.append(benchmark)
                
                # Log real-time factor
                rtf = (end_time - start_time) / segment['duration']
                if idx < 3:  # Log first few
                    logger.info(f"    Segment {idx}: RTF = {rtf:.2f}x ({end_time-start_time:.1f}s for {segment['duration']:.1f}s audio)")
                
            except Exception as e:
                logger.error(f"Error processing segment {idx}: {e}")
                all_results.append(BenchmarkResult(
                    model_name=model_id,
                    split=f"{split_name}_{duration//60}min",
                    sample_id=idx,
                    reference_text=segment['text'][:1000],
                    asr_text="",
                    wer=1.0,
                    medical_wer=1.0,
                    processing_time=0,
                    confidence=0,
                    audio_duration=segment['duration'],
                    word_count=len(segment['text'].split()),
                    medical_term_count=0,
                    medical_terms_found=[],
                    peak_memory_mb=0,
                    cpu_percent=0,
                    error=str(e)
                ))
    
    return all_results

# ==================== MODIFIED MAIN EXECUTION SECTION ====================

def main_with_concatenation():
    """
    Modified main function that combines audio into 5-minute segments.
    """
    print("=" * 80)
    print("🏥 MEDICAL ASR BENCHMARKING WITH AUDIO CONCATENATION")
    print("=" * 80)
    print(f"Dataset: {DATASET_PATH}")
    print(f"Creating segments of: 1min, 5min, 15min")
    print(f"Models to test: {len(MODELS_TO_TEST)}")
    print("=" * 80)
    
    # Create output directory structure
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load dataset splits
    splits = ['train', 'eval', 'test']
    segments_by_split = {}
    
    for split in splits:
        file_path = DATASET_PATH / f"{split}-00000-of-00001.parquet"
        
        if not file_path.exists():
            logger.warning(f"Split not found: {split}")
            continue
        
        logger.info(f"\nLoading {split} split...")
        df = pd.read_parquet(file_path)
        
        # Limit samples if needed (but we'll combine them anyway)
        if SAMPLE_LIMIT_PER_SPLIT:
            df = df.head(SAMPLE_LIMIT_PER_SPLIT * 10)  # Load more for combination
        
        # Prepare audio samples
        samples = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Loading {split}"):
            audio_item = row['audio']
            text = str(row['text'])
            
            audio_array, sr = load_audio_from_item(audio_item)
            
            if audio_array is not None:
                samples.append((audio_array, sr, text))
        
        # Create segments of different durations
        segments_by_duration = create_varied_duration_segments(
            samples, 
            durations=[60, 300, 900]  # 1min, 5min, 15min
        )
        
        segments_by_split[split] = segments_by_duration
        logger.info(f"✓ Created segments for {split}:")
        for duration, segs in segments_by_duration.items():
            logger.info(f"  • {duration//60}min: {len(segs)} segments")
    
    if not segments_by_split:
        logger.error("No audio segments created!")
        return
    
    # Initialize and benchmark each model
    all_results = []
    
    for model_id in MODELS_TO_TEST:
        config = MODEL_CONFIGS.get(model_id)
        if not config:
            logger.warning(f"Unknown model: {model_id}")
            continue
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing: {model_id}")
        logger.info(f"{'='*60}")
        
        # Initialize model
        model = initialize_model(model_id, config)
        if not model:
            continue
        
        # Benchmark on each split's segments
        for split_name, segments_by_duration in segments_by_split.items():
            results = benchmark_with_varied_durations(
                model_id, 
                model, 
                segments_by_duration, 
                split_name
            )
            all_results.extend(results)
            
            # Save intermediate results
            for duration in [60, 300, 900]:
                duration_results = [r for r in results if f"{duration//60}min" in r.split]
                if duration_results:
                    split_df = pd.DataFrame([r.to_dict() for r in duration_results])
                    split_output = OUTPUT_DIR / split_name / f"{duration//60}min" / f"{model_id}_results.csv"
                    split_output.parent.mkdir(parents=True, exist_ok=True)
                    split_df.to_csv(split_output, index=False)
        
        # Clean up model
        if hasattr(model, 'cleanup'):
            model.cleanup()
        del model
        gc.collect()
    
    # Generate analysis (similar to original but with duration info)
    analysis, df = analyze_results(all_results)
    
    # Save all results
    all_results_df = pd.DataFrame([r.to_dict() for r in all_results])
    all_results_path = OUTPUT_DIR / "all_transcriptions_duration.csv"
    all_results_df.to_csv(all_results_path, index=False)
    
    # Create duration-specific summary
    duration_summary = all_results_df.groupby(['model_name', 'split']).agg({
        'wer': 'mean',
        'medical_wer': 'mean',
        'processing_time': 'mean',
        'audio_duration': 'mean',
        'sample_id': 'count'
    }).round(3)
    
    # Calculate real-time factor
    duration_summary['rtf'] = duration_summary['processing_time'] / duration_summary['audio_duration']
    
    duration_summary.to_csv(OUTPUT_DIR / "duration_benchmark_summary.csv")
    
    print("\n" + "=" * 80)
    print("🏁 BENCHMARKING COMPLETE")
    print("=" * 80)
    
    print("\n📊 Real-Time Factor by Duration:")
    for model in all_results_df['model_name'].unique():
        model_df = all_results_df[all_results_df['model_name'] == model]
        print(f"\n{model}:")
        for duration in [60, 300, 900]:
            dur_df = model_df[model_df['audio_duration'].between(duration-10, duration+10)]
            if len(dur_df) > 0:
                avg_rtf = (dur_df['processing_time'] / dur_df['audio_duration']).mean()
                print(f"  {duration//60}min: RTF = {avg_rtf:.2f}x")
    
    return all_results_df

# Replace the original main() call with:
if __name__ == "__main__":
    # Choose which version to run:
    use_concatenation = True  # Set to False for original per-sample benchmarking
    
    if use_concatenation:
        main_with_concatenation()
    else:
        main()