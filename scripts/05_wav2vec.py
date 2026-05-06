#!/usr/bin/env python3
"""
MEDICAL WAV2VEC2 EVALUATION - COMPREHENSIVE ANALYSIS
Tests multiple Wav2Vec2 variants with medical-specific metrics.
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
from collections import defaultdict
import re

warnings.filterwarnings('ignore')
logging.getLogger('absl').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.WARNING)
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('wav2vec2_medical_evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import torch as _torch
def _get_device():
    if _torch.cuda.is_available(): return "cuda"
    if hasattr(_torch.backends, 'mps') and _torch.backends.mps.is_available(): return "mps"
    return "cpu"
DEVICE = _get_device()

# ==================== CONFIGURATION ====================
DATASET_PATH = Path("/Users/mahriovezmyradova/MultiMed_dataset/German")
OUTPUT_DIR = Path("data/outputs/wav2vec2_medical_evaluation")

# Wav2Vec2 models to test
WAV2VEC2_MODELS = [
    {
        "name": "wav2vec2_facebook_original",
        "model_name": "facebook/wav2vec2-large-xlsr-53-german",
        "description": "Facebook Original German",
        "type": "base"
    },
    {
        "name": "wav2vec2_jonatasgrosman",
        "model_name": "jonatasgrosman/wav2vec2-large-xlsr-53-german",
        "description": "Jonatas Grosman Fine-tuned",
        "type": "fine_tuned_general"
    },
    {
        "name": "wav2vec2_oliverguhr",
        "model_name": "oliverguhr/wav2vec2-large-xlsr-53-german-cv8",
        "description": "Oliver Guhr Fine-tuned (Common Voice 8)",
        "type": "fine_tuned_cv"
    }
]

# Test splits to evaluate
TEST_SPLITS = ["test", "eval"]  # Use both test and eval

# Sample size for testing (None = all samples)
SAMPLE_LIMIT = None  # Set to e.g., 50 for quick testing

# ==================== MEDICAL TERM LISTS ====================

# Comprehensive medical German vocabulary for MTER calculation
MEDICAL_TERMS = {
    # Diagnoses and conditions
    "diagnosis": [
        "diagnose", "diagnosen", "diagnostik", "diagnostische",
        "krankheit", "erkrankung", "erkrankungen", "leiden",
        "syndrom", "symptom", "symptome", "beschwerden",
        "infektion", "entzündung", "entzuendung", "tumor",
        "karzinom", "metastase", "diabetes", "hypertonie",
        "hypotonie", "asthma", "bronchitis", "pneumonie",
        "grippe", "migräne", "migraene", "epilepsie",
        "depression", "demenz", "alzheimer", "parkinson",
        "arthrose", "arthritis", "osteoporose", "sklerose"
    ],
    
    # Symptoms
    "symptoms": [
        "schmerz", "schmerzen", "kopfschmerzen", "rueckenschmerzen",
        "bauchschmerzen", "gliederschmerzen", "gelenkschmerzen",
        "muskelschmerzen", "fieber", "husten", "schnupfen",
        "übelkeit", "uebelkeit", "erbrechen", "durchfall",
        "verstopfung", "schwindel", "müdigkeit", "muedigkeit",
        "schwäche", "schwaeche", "appetitlosigkeit", "gewichtsverlust",
        "blutdruck", "herzrasen", "atemnot", "kurzatmigkeit"
    ],
    
    # Treatments and medications
    "treatment": [
        "therapie", "behandlung", "medikament", "medikamente",
        "tablette", "tabletten", "kapsel", "kapseln", "salbe",
        "spritze", "infusion", "operation", "eingriff",
        "bestrahlung", "chemotherapie", "reha", "rehabilitation",
        "physiotherapie", "krankengymnastik", "massage",
        "impfung", "vorsorge", "nachsorge"
    ],
    
    # Medical procedures and examinations
    "procedures": [
        "untersuchung", "untersuchungen", "röntgen", "roentgen",
        "mrt", "ct", "ultraschall", "sonographie", "ekg",
        "blutabnahme", "bluttest", "blutdruckmessung",
        "körperliche", "koerperliche", "abhören", "abhoeren",
        "abtasten", "reflexe", "blutprobe", "urinprobe"
    ],
    
    # Medical professionals and facilities
    "professionals": [
        "arzt", "ärztin", "aerztin", "doktor", "mediziner",
        "chirurg", "chirurgin", "internist", "internistin",
        "kardiologe", "kardiologin", "neurologe", "neurologin",
        "onkologe", "onkologin", "radiologe", "radiologin",
        "apotheker", "apothekerin", "pfleger", "schwester",
        "krankenhaus", "klinik", "praxis", "ambulanz"
    ],
    
    # Body parts and anatomy
    "anatomy": [
        "kopf", "hals", "nacken", "schulter", "rücken", "ruecken",
        "wirbelsäule", "wirbelsaule", "brust", "bauch", "becken",
        "arm", "arme", "hand", "hände", "haende", "finger",
        "bein", "beine", "fuß", "fuss", "füße", "fuesse", "zeh",
        "herz", "lunge", "leber", "niere", "magen", "darm",
        "gehirn", "hirn", "nerven", "muskel", "muskeln", "gelenk",
        "gelenke", "knochen", "haut", "blut", "gefäße", "gefaesse"
    ]
}

# Flatten the medical terms for easier use
ALL_MEDICAL_TERMS = []
for category in MEDICAL_TERMS.values():
    ALL_MEDICAL_TERMS.extend(category)
ALL_MEDICAL_TERMS = list(set(ALL_MEDICAL_TERMS))  # Remove duplicates

# ==================== UTILITY FUNCTIONS ====================

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
            text = text.lower()
            import string
            text = text.translate(str.maketrans('', '', string.punctuation))
            text = ' '.join(text.split())
            return text
        
        ref_clean = clean_text(reference)
        hyp_clean = clean_text(hypothesis)
        
        return jiwer.wer(ref_clean, hyp_clean)
        
    except ImportError:
        # Simple WER calculation fallback
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
                        dp[i-1][j] + 1,  # deletion
                        dp[i][j-1] + 1,  # insertion
                        dp[i-1][j-1] + 1 # substitution
                    )
        
        errors = dp[n][m]
        return errors / max(n, 1)

def calculate_medical_term_error(reference, hypothesis, medical_terms=None):
    """
    Calculate Medical Term Error Rate (MTER) - WER specifically on medical terms.
    Returns both overall MTER and per-category MTER.
    """
    if medical_terms is None:
        medical_terms = ALL_MEDICAL_TERMS
    
    # Lowercase and split
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    
    # Extract only medical terms from reference
    ref_medical_words = [w for w in ref_words if w in medical_terms]
    
    if not ref_medical_words:  # No medical terms in this sample
        return 0.0, {}
    
    # Create alignment (simplified - find corresponding words in hypothesis)
    # This is a simplified approach - for production, use proper alignment
    hyp_medical_words = [w for w in hyp_words if w in medical_terms]
    
    # Calculate WER on these sequences
    n = len(ref_medical_words)
    m = len(hyp_medical_words)
    
    if n == 0:
        return 0.0, {}
    
    # Simple Levenshtein distance
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
        
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_medical_words[i-1] == hyp_medical_words[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(
                    dp[i-1][j] + 1,  # deletion
                    dp[i][j-1] + 1,  # insertion
                    dp[i-1][j-1] + 1 # substitution
                )
    
    errors = dp[n][m]
    mter = errors / n
    
    # Calculate per-category MTER
    category_errors = {}
    for category, terms in MEDICAL_TERMS.items():
        category_ref = [w for w in ref_words if w in terms]
        if category_ref:
            category_hyp = [w for w in hyp_words if w in terms]
            cat_n = len(category_ref)
            cat_m = len(category_hyp)
            
            # Simple error count (can be refined)
            cat_errors = abs(cat_n - cat_m)  # Simplified
            category_errors[category] = cat_errors / cat_n
    
    return mter, category_errors

def get_model_size(model_name):
    """Get approximate model size in MB."""
    try:
        from huggingface_hub import model_info
        
        info = model_info(model_name)
        # Estimate size from safetensors or pytorch files
        siblings = info.siblings
        total_size = sum(s.size for s in siblings if s.rfilename.endswith(('.bin', '.safetensors')))
        return total_size / (1024 * 1024)  # Convert to MB
    except:
        # Default sizes based on model type
        size_map = {
            "facebook/wav2vec2-large-xlsr-53-german": 1260,  # ~1.26GB
            "jonatasgrosman/wav2vec2-large-xlsr-53-german": 1260,
            "oliverguhr/wav2vec2-large-xlsr-53-german-cv8": 1260
        }
        return size_map.get(model_name, 1260)

def measure_inference_time(model, audio_path, durations=[60, 300, 900]):
    """
    Measure inference time for different audio durations.
    durations: list of seconds to test (60s=1min, 300s=5min, 900s=15min)
    """
    results = {}
    
    for duration in durations:
        try:
            # Load or generate test audio
            if audio_path and Path(audio_path).exists():
                # Load actual audio file
                audio, sr = sf.read(audio_path)
                # Trim/pad to desired duration
                target_samples = int(sr * duration)
                if len(audio) > target_samples:
                    audio = audio[:target_samples]
                elif len(audio) < target_samples:
                    # Pad with zeros
                    audio = np.pad(audio, (0, target_samples - len(audio)))
            else:
                # Generate synthetic audio
                sr = 16000
                audio = np.random.randn(sr * duration).astype(np.float32) * 0.01
            
            # Warm-up
            _ = model.transcribe(audio[:sr], sr)
            
            # Measure
            start_time = time.time()
            result = model.transcribe(audio, sr)
            end_time = time.time()
            
            # Get memory usage
            memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            results[duration] = {
                'time_seconds': end_time - start_time,
                'real_time_factor': (end_time - start_time) / duration,
                'memory_mb': memory,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Failed to measure inference for {duration}s: {e}")
            results[duration] = {
                'time_seconds': None,
                'real_time_factor': None,
                'memory_mb': None,
                'success': False,
                'error': str(e)
            }
    
    return results

# ==================== MODEL INITIALIZATION ====================

def initialize_wav2vec2_model(model_config):
    """Initialize Wav2Vec2 model."""
    try:
        from src.asr.wav2vec2 import Wav2Vec2ASR

        config = {
            "name": model_config["name"],
            "model_name": model_config["model_name"],
            "language": "de",
            "device": DEVICE,
            "cache_dir": str(Path.home() / ".cache" / "huggingface")
        }
        
        model = Wav2Vec2ASR(config)
        
        # Load model and measure time
        start_time = time.time()
        model.load_model()
        load_time = time.time() - start_time
        
        # Get model size
        model_size = get_model_size(model_config["model_name"])
        
        logger.info(f"✓ Initialized {model_config['name']} in {load_time:.1f}s")
        
        return {
            'instance': model,
            'config': model_config,
            'load_time': load_time,
            'size_mb': model_size
        }
        
    except Exception as e:
        logger.error(f"Failed to initialize {model_config['name']}: {e}")
        return None

# ==================== EVALUATION FUNCTIONS ====================

def evaluate_on_split(model_info, split_name, audio_arrays, sample_rates, reference_texts):
    """Evaluate a single model on a split."""
    model = model_info['instance']
    model_name = model_info['config']['name']
    
    results = []
    
    logger.info(f"Evaluating {model_name} on {split_name} ({len(audio_arrays)} samples)...")
    
    for idx, (audio, sr, ref) in enumerate(tqdm(zip(audio_arrays, sample_rates, reference_texts), 
                                                total=len(audio_arrays), 
                                                desc=f"{model_name}")):
        try:
            # Measure time
            start_time = time.time()
            result = model.transcribe(audio, sr)
            processing_time = time.time() - start_time
            
            # Calculate WER
            wer = calculate_wer(ref, result.text)
            
            # Calculate Medical Term Error Rate (MTER)
            mter, category_errors = calculate_medical_term_error(ref, result.text)
            
            # Store result
            sample_result = {
                'model_name': model_name,
                'model_description': model_info['config']['description'],
                'split': split_name,
                'sample_id': idx,
                'reference_text': ref,
                'asr_text': result.text,
                'wer': wer,
                'mter': mter,
                'confidence': result.confidence,
                'processing_time': processing_time,
                'audio_duration': len(audio) / sr,
                'audio_length': len(audio)
            }
            
            # Add category errors
            for category, error in category_errors.items():
                sample_result[f'mter_{category}'] = error
            
            results.append(sample_result)
            
        except Exception as e:
            logger.error(f"Error processing sample {idx} with {model_name}: {e}")
            results.append({
                'model_name': model_name,
                'model_description': model_info['config']['description'],
                'split': split_name,
                'sample_id': idx,
                'reference_text': ref,
                'asr_text': '',
                'wer': 1.0,
                'mter': 1.0,
                'confidence': 0,
                'processing_time': 0,
                'audio_duration': len(audio) / sr if sr > 0 else 0,
                'error': str(e)
            })
    
    return results

def generate_comprehensive_report(all_results, model_infos, performance_metrics):
    """Generate a comprehensive evaluation report."""
    
    df = pd.DataFrame(all_results)
    
    # Overall statistics
    report = {
        'evaluation_date': datetime.now().isoformat(),
        'dataset_info': {
            'total_samples': len(df),
            'splits': df['split'].unique().tolist(),
            'models_evaluated': df['model_name'].unique().tolist()
        },
        'models': {},
        'comparisons': {},
        'recommendations': []
    }
    
    # Per-model statistics
    for model_name in df['model_name'].unique():
        model_df = df[df['model_name'] == model_name]
        model_info = next((m for m in model_infos if m['config']['name'] == model_name), {})
        
        # Calculate metrics
        avg_wer = model_df['wer'].mean()
        avg_mter = model_df['mter'].mean()
        avg_confidence = model_df['confidence'].mean()
        avg_time = model_df['processing_time'].mean()
        
        # Medical term categories
        category_metrics = {}
        for category in MEDICAL_TERMS.keys():
            col = f'mter_{category}'
            if col in model_df.columns:
                category_metrics[category] = {
                    'error_rate': model_df[col].mean(),
                    'std': model_df[col].std()
                }
        
        # Find best/worst samples
        best_sample = model_df.loc[model_df['wer'].idxmin()] if len(model_df) > 0 else None
        worst_sample = model_df.loc[model_df['wer'].idxmax()] if len(model_df) > 0 else None
        
        # Store model stats
        report['models'][model_name] = {
            'description': model_info.get('config', {}).get('description', ''),
            'model_name': model_info.get('config', {}).get('model_name', ''),
            'size_mb': model_info.get('size_mb', 0),
            'load_time_seconds': model_info.get('load_time', 0),
            'performance': {
                'wer': {
                    'mean': avg_wer,
                    'std': model_df['wer'].std(),
                    'min': model_df['wer'].min(),
                    'max': model_df['wer'].max(),
                    'median': model_df['wer'].median()
                },
                'mter': {
                    'mean': avg_mter,
                    'std': model_df['mter'].std(),
                    'min': model_df['mter'].min(),
                    'max': model_df['mter'].max()
                },
                'confidence': {
                    'mean': avg_confidence,
                    'std': model_df['confidence'].std()
                },
                'speed': {
                    'mean_processing_time': avg_time,
                    'real_time_factor': avg_time / model_df['audio_duration'].mean() if model_df['audio_duration'].mean() > 0 else 0
                },
                'medical_categories': category_metrics
            },
            'samples_processed': len(model_df),
            'best_sample': {
                'wer': best_sample['wer'],
                'reference': best_sample['reference_text'][:100] + '...' if best_sample is not None else '',
                'asr': best_sample['asr_text'][:100] + '...' if best_sample is not None else ''
            } if best_sample is not None else None,
            'worst_sample': {
                'wer': worst_sample['wer'],
                'reference': worst_sample['reference_text'][:100] + '...' if worst_sample is not None else '',
                'asr': worst_sample['asr_text'][:100] + '...' if worst_sample is not None else ''
            } if worst_sample is not None else None
        }
        
        # Add performance metrics if available
        if model_name in performance_metrics:
            report['models'][model_name]['inference_benchmark'] = performance_metrics[model_name]
    
    # Generate comparisons
    models_list = list(df['model_name'].unique())
    if len(models_list) >= 2:
        comparisons = {}
        for i, model1 in enumerate(models_list):
            for model2 in models_list[i+1:]:
                df1 = df[df['model_name'] == model1]
                df2 = df[df['model_name'] == model2]
                
                # Compare WER
                wer_diff = df1['wer'].mean() - df2['wer'].mean()
                
                # Compare MTER
                mter_diff = df1['mter'].mean() - df2['mter'].mean()
                
                comparisons[f"{model1}_vs_{model2}"] = {
                    'wer_difference': wer_diff,
                    'mter_difference': mter_diff,
                    'better_for_wer': model1 if wer_diff < 0 else model2,
                    'better_for_mter': model1 if mter_diff < 0 else model2,
                    'wer_improvement_percent': abs(wer_diff) / max(df1['wer'].mean(), df2['wer'].mean()) * 100
                }
        
        report['comparisons'] = comparisons
    
    # Generate recommendations
    recommendations = []
    
    # Find best overall
    if report['models']:
        best_wer_model = min(report['models'].items(), key=lambda x: x[1]['performance']['wer']['mean'])
        best_mter_model = min(report['models'].items(), key=lambda x: x[1]['performance']['mter']['mean'])
        fastest_model = min(report['models'].items(), key=lambda x: x[1]['performance']['speed']['mean_processing_time'])
        smallest_model = min(report['models'].items(), key=lambda x: x[1]['size_mb'])
        
        recommendations.append({
            'category': 'Best Overall Accuracy',
            'model': best_wer_model[0],
            'metric': f"WER: {best_wer_model[1]['performance']['wer']['mean']:.3f}",
            'reason': "Lowest Word Error Rate across all samples"
        })
        
        recommendations.append({
            'category': 'Best Medical Term Recognition',
            'model': best_mter_model[0],
            'metric': f"MTER: {best_mter_model[1]['performance']['mter']['mean']:.3f}",
            'reason': "Lowest error rate specifically on medical terminology"
        })
        
        recommendations.append({
            'category': 'Fastest Processing',
            'model': fastest_model[0],
            'metric': f"Avg: {fastest_model[1]['performance']['speed']['mean_processing_time']:.2f}s",
            'reason': "Quickest transcription time per sample"
        })
        
        recommendations.append({
            'category': 'Smallest Model Size',
            'model': smallest_model[0],
            'metric': f"Size: {smallest_model[1]['size_mb']:.0f}MB",
            'reason': "Most lightweight for deployment"
        })
    
    report['recommendations'] = recommendations
    
    return report

# ==================== MAIN EXECUTION ====================

def main():
    print("=" * 80)
    print("MEDICAL WAV2VEC2 EVALUATION - COMPREHENSIVE ANALYSIS")
    print("=" * 80)
    print(f"Dataset: {DATASET_PATH}")
    print(f"Models to evaluate: {len(WAV2VEC2_MODELS)}")
    print(f"Splits: {TEST_SPLITS}")
    print(f"Sample limit per split: {SAMPLE_LIMIT or 'All'}")
    print(f"Medical terms tracked: {len(ALL_MEDICAL_TERMS)}")
    print("=" * 80)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize all models
    model_infos = []
    for model_config in WAV2VEC2_MODELS:
        logger.info(f"\nInitializing {model_config['name']}...")
        model_info = initialize_wav2vec2_model(model_config)
        if model_info:
            model_infos.append(model_info)
    
    if not model_infos:
        logger.error("No models initialized successfully!")
        return
    
    # Load dataset
    logger.info("\n" + "=" * 60)
    logger.info("Loading dataset...")
    logger.info("=" * 60)
    
    dataset_audio = {}
    dataset_texts = {}
    dataset_rates = {}
    
    for split in TEST_SPLITS:
        split_file = DATASET_PATH / f"{split}-00000-of-00001.parquet"
        
        if not split_file.exists():
            logger.warning(f"Split file not found: {split_file}")
            continue
        
        logger.info(f"Loading {split} split...")
        df = pd.read_parquet(split_file)
        
        # Apply sample limit
        if SAMPLE_LIMIT and len(df) > SAMPLE_LIMIT:
            df = df.head(SAMPLE_LIMIT)
            logger.info(f"  Limited to {SAMPLE_LIMIT} samples")
        
        # Load audio
        audio_arrays = []
        sample_rates = []
        reference_texts = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Loading {split}"):
            audio_item = row['audio']
            text = str(row['text'])
            
            audio_array, sr = load_audio_from_item(audio_item)
            
            if audio_array is not None:
                audio_arrays.append(audio_array)
                sample_rates.append(sr)
                reference_texts.append(text)
        
        dataset_audio[split] = audio_arrays
        dataset_rates[split] = sample_rates
        dataset_texts[split] = reference_texts
        
        logger.info(f"  ✓ Loaded {len(audio_arrays)}/{len(df)} samples for {split}")
    
    # Evaluate each model on each split
    all_results = []
    
    logger.info("\n" + "=" * 60)
    logger.info("Starting evaluation...")
    logger.info("=" * 60)
    
    for model_info in model_infos:
        model_name = model_info['config']['name']
        logger.info(f"\n📊 Evaluating {model_name}...")
        
        for split in TEST_SPLITS:
            if split not in dataset_audio:
                continue
                
            logger.info(f"  Processing {split} split...")
            
            results = evaluate_on_split(
                model_info,
                split,
                dataset_audio[split],
                dataset_rates[split],
                dataset_texts[split]
            )
            
            all_results.extend(results)
            
            # Save intermediate results
            split_df = pd.DataFrame(results)
            split_output = OUTPUT_DIR / split / model_name
            split_output.mkdir(parents=True, exist_ok=True)
            split_df.to_csv(split_output / f"{split}_{model_name}_results.csv", index=False, encoding='utf-8')
            
            # Log quick stats
            avg_wer = split_df['wer'].mean()
            avg_mter = split_df['mter'].mean()
            logger.info(f"    WER: {avg_wer:.3f}, MTER: {avg_mter:.3f}")
    
    # Measure inference performance
    logger.info("\n" + "=" * 60)
    logger.info("Measuring inference performance...")
    logger.info("=" * 60)
    
    performance_metrics = {}
    test_audio_path = None  # Use synthetic audio
    
    for model_info in model_infos:
        model_name = model_info['config']['name']
        logger.info(f"Benchmarking {model_name}...")
        
        metrics = measure_inference_time(
            model_info['instance'],
            test_audio_path,
            durations=[60, 300, 900]  # 1min, 5min, 15min
        )
        
        performance_metrics[model_name] = metrics
        
        # Log results
        for duration, result in metrics.items():
            if result.get('success'):
                logger.info(f"  {duration}s audio: {result['time_seconds']:.2f}s (RTF: {result['real_time_factor']:.2f})")
    
    # Generate comprehensive report
    logger.info("\n" + "=" * 60)
    logger.info("Generating comprehensive report...")
    logger.info("=" * 60)
    
    report = generate_comprehensive_report(all_results, model_infos, performance_metrics)
    
    # Save report
    report_path = OUTPUT_DIR / "comprehensive_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # Save CSV summary
    df_all = pd.DataFrame(all_results)
    summary_path = OUTPUT_DIR / "all_results.csv"
    df_all.to_csv(summary_path, index=False, encoding='utf-8')
    
    # Create summary tables
    summary_tables = []
    
    for model_name in df_all['model_name'].unique():
        model_df = df_all[df_all['model_name'] == model_name]
        
        for split in model_df['split'].unique():
            split_df = model_df[model_df['split'] == split]
            
            row = {
                'model': model_name,
                'split': split,
                'samples': len(split_df),
                'wer_mean': split_df['wer'].mean(),
                'wer_std': split_df['wer'].std(),
                'mter_mean': split_df['mter'].mean(),
                'mter_std': split_df['mter'].std(),
                'confidence_mean': split_df['confidence'].mean(),
                'time_mean': split_df['processing_time'].mean()
            }
            
            # Add category MTERs
            for category in MEDICAL_TERMS.keys():
                col = f'mter_{category}'
                if col in split_df.columns:
                    row[f'mter_{category}'] = split_df[col].mean()
            
            summary_tables.append(row)
    
    summary_df = pd.DataFrame(summary_tables)
    summary_df.to_csv(OUTPUT_DIR / "summary_table.csv", index=False, encoding='utf-8')
    
    # Create markdown report for thesis
    markdown_path = OUTPUT_DIR / "thesis_results.md"
    with open(markdown_path, 'w', encoding='utf-8') as f:
        f.write("# Medical ASR Evaluation Results\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Models Evaluated\n\n")
        for model_info in model_infos:
            f.write(f"- **{model_info['config']['name']}**: {model_info['config']['description']}\n")
            f.write(f"  - Model: {model_info['config']['model_name']}\n")
            f.write(f"  - Size: {model_info['size_mb']:.0f} MB\n")
            f.write(f"  - Load time: {model_info['load_time']:.1f}s\n\n")
        
        f.write("## Performance Summary\n\n")
        
        # Overall table
        f.write("### Overall WER and MTER\n\n")
        f.write("| Model | Split | Samples | WER (↓) | MTER (↓) | Confidence (↑) | Time (s) |\n")
        f.write("|-------|-------|---------|---------|----------|----------------|----------|\n")
        
        for _, row in summary_df.iterrows():
            f.write(f"| {row['model']} | {row['split']} | {row['samples']} | ")
            f.write(f"{row['wer_mean']:.3f}±{row['wer_std']:.3f} | ")
            f.write(f"{row['mter_mean']:.3f}±{row['mter_std']:.3f} | ")
            f.write(f"{row['confidence_mean']:.3f} | {row['time_mean']:.2f}s |\n")
        
        f.write("\n### Medical Term Category Performance\n\n")
        for model_name in df_all['model_name'].unique():
            f.write(f"#### {model_name}\n\n")
            f.write("| Category | MTER |\n")
            f.write("|----------|------|\n")
            
            model_df = df_all[df_all['model_name'] == model_name]
            for category in MEDICAL_TERMS.keys():
                col = f'mter_{category}'
                if col in model_df.columns:
                    f.write(f"| {category} | {model_df[col].mean():.3f} |\n")
            f.write("\n")
        
        f.write("## Inference Speed Benchmark\n\n")
        f.write("| Model | 1-min audio | 5-min audio | 15-min audio |\n")
        f.write("|-------|-------------|-------------|--------------|\n")
        
        for model_name, metrics in performance_metrics.items():
            f.write(f"| {model_name} | ")
            for duration in [60, 300, 900]:
                if duration in metrics and metrics[duration].get('success'):
                    time_val = metrics[duration]['time_seconds']
                    rtf = metrics[duration]['real_time_factor']
                    f.write(f"{time_val:.1f}s (RTF: {rtf:.2f}) | ")
                else:
                    f.write("Failed | ")
            f.write("\n")
        
        f.write("\n## Recommendations\n\n")
        for rec in report.get('recommendations', []):
            f.write(f"- **{rec['category']}**: {rec['model']}\n")
            f.write(f"  - {rec['metric']}\n")
            f.write(f"  - {rec['reason']}\n\n")
    
    # Print final summary
    print("\n" + "=" * 80)
    print("📊 EVALUATION COMPLETE - SUMMARY")
    print("=" * 80)
    
    print(f"\n✅ Results saved to: {OUTPUT_DIR}")
    print("\n📁 Output files:")
    print(f"  • Comprehensive report: {OUTPUT_DIR}/comprehensive_report.json")
    print(f"  • All results: {OUTPUT_DIR}/all_results.csv")
    print(f"  • Summary table: {OUTPUT_DIR}/summary_table.csv")
    print(f"  • Thesis markdown: {OUTPUT_DIR}/thesis_results.md")
    
    print("\n🏆 RECOMMENDATIONS:")
    for rec in report.get('recommendations', []):
        print(f"\n  {rec['category']}:")
        print(f"    • Model: {rec['model']}")
        print(f"    • Metric: {rec['metric']}")
        print(f"    • Why: {rec['reason']}")
    
    # Show comparison
    if 'comparisons' in report:
        print("\n📊 MODEL COMPARISONS:")
        for comp_name, comp_data in report['comparisons'].items():
            models = comp_name.split('_vs_')
            print(f"\n  {models[0]} vs {models[1]}:")
            print(f"    • WER difference: {comp_data['wer_difference']:.3f}")
            print(f"    • MTER difference: {comp_data['mter_difference']:.3f}")
            print(f"    • Better for medical terms: {comp_data['better_for_mter']}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()