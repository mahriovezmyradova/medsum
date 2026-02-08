#!/usr/bin/env python3
"""
COMPLETE DATASET ANALYSIS FOR MEDICAL ASR
Processes entire MultiMed dataset (train, eval, test) and provides detailed analysis.
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
from datetime import datetime
import os

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dataset_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add src to path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root / "src"))

# ==================== CONFIGURATION ====================
DATASET_PATH = Path("/Users/mahriovezmyradova/MultiMed_dataset/German")
SPLITS = ["train", "eval", "test"]
OUTPUT_DIR = Path("data/outputs/full_dataset_analysis")
MODELS_TO_TEST = ["whisper_base", "wav2vec2_german"]  # Skip Google for now
SAMPLE_LIMIT_PER_SPLIT = None  # Set to None to process all, or e.g., 100 for testing

# ==================== HELPER FUNCTIONS ====================

def load_entire_dataset():
    """Load entire MultiMed dataset."""
    dataset_info = {}
    
    for split in SPLITS:
        split_file = DATASET_PATH / f"{split}-00000-of-00001.parquet"
        if not split_file.exists():
            logger.warning(f"Split file not found: {split_file}")
            continue
            
        logger.info(f"Loading {split} split...")
        df = pd.read_parquet(split_file)
        
        # Apply sample limit if specified
        if SAMPLE_LIMIT_PER_SPLIT and len(df) > SAMPLE_LIMIT_PER_SPLIT:
            df = df.sample(n=SAMPLE_LIMIT_PER_SPLIT, random_state=42)
            logger.info(f"  Limited to {SAMPLE_LIMIT_PER_SPLIT} samples")
        
        dataset_info[split] = {
            'dataframe': df,
            'samples': len(df),
            'file': split_file
        }
        
        # Basic statistics
        texts = df['text'].astype(str).tolist()
        text_lengths = [len(text.split()) for text in texts]
        
        logger.info(f"  ✓ Loaded {len(df)} samples")
        logger.info(f"  ✓ Average words: {np.mean(text_lengths):.1f}")
        logger.info(f"  ✓ Min words: {np.min(text_lengths)}")
        logger.info(f"  ✓ Max words: {np.max(text_lengths)}")
    
    return dataset_info

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
                        dp[i-1][j] + 1,
                        dp[i][j-1] + 1,
                        dp[i-1][j-1] + 1
                    )
        
        errors = dp[n][m]
        return errors / max(n, 1)

# ==================== ASR MODEL FUNCTIONS ====================

def initialize_model(model_name):
    """Initialize ASR model."""
    try:
        if model_name.startswith("whisper"):
            from asr.whisper import WhisperASR
            model_size = model_name.split("_")[1] if "_" in model_name else "base"
            config = {
                "name": model_name,
                "model_size": model_size,
                "language": "de",
                "device": "cpu"
            }
            model = WhisperASR(config)
            
        elif model_name.startswith("wav2vec2"):
            from asr.wav2vec2 import Wav2Vec2ASR
            config = {
                "name": model_name,
                "model_name": "facebook/wav2vec2-large-xlsr-53-german",
                "language": "de",
                "device": "cpu",
                "cache_dir": str(Path.home() / ".cache" / "huggingface")
            }
            model = Wav2Vec2ASR(config)
            
        else:
            logger.error(f"Unknown model: {model_name}")
            return None
        
        model.load_model()
        logger.info(f"✓ Initialized {model_name}")
        return model
        
    except Exception as e:
        logger.error(f"Failed to initialize {model_name}: {e}")
        return None

def process_batch_with_model(model, audio_arrays, sample_rates, reference_texts, split_name, model_name):
    """Process a batch of audio with a specific model."""
    results = []
    
    logger.info(f"Processing {len(audio_arrays)} samples with {model_name}...")
    
    for idx, (audio, sr, ref) in enumerate(tqdm(zip(audio_arrays, sample_rates, reference_texts), 
                                                total=len(audio_arrays), 
                                                desc=f"{model_name}")):
        try:
            # Transcribe
            result = model.transcribe(audio, sr)
            
            # Calculate WER
            wer = calculate_wer(ref, result.text)
            
            # Store detailed result
            sample_result = {
                'split': split_name,
                'sample_id': idx,
                'model': model_name,
                'reference_text': ref,
                'asr_text': result.text,
                'wer': wer,
                'processing_time': result.processing_time,
                'confidence': result.confidence,
                'audio_duration': len(audio) / sr if sr > 0 else 0,
                'word_count': len(ref.split()),
                'audio_length': len(audio)
            }
            
            results.append(sample_result)
            
        except Exception as e:
            logger.error(f"Error processing sample {idx} with {model_name}: {e}")
            # Add error result
            results.append({
                'split': split_name,
                'sample_id': idx,
                'model': model_name,
                'reference_text': ref,
                'asr_text': '',
                'wer': 1.0,  # Max error
                'processing_time': 0,
                'confidence': 0,
                'audio_duration': 0,
                'word_count': len(ref.split()),
                'error': str(e)
            })
    
    return results

# ==================== ANALYSIS FUNCTIONS ====================

def generate_detailed_analysis(all_results):
    """Generate comprehensive analysis from all results."""
    analysis = {
        'overall_summary': {},
        'per_split_summary': {},
        'per_model_summary': {},
        'top_errors': {},
        'sample_comparisons': []
    }
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(all_results)
    
    # Overall statistics
    analysis['overall_summary'] = {
        'total_samples': len(df),
        'unique_splits': df['split'].unique().tolist(),
        'models_tested': df['model'].unique().tolist()
    }
    
    # Per-split analysis
    for split in df['split'].unique():
        split_df = df[df['split'] == split]
        analysis['per_split_summary'][split] = {
            'samples': len(split_df),
            'avg_wer': split_df['wer'].mean(),
            'std_wer': split_df['wer'].std(),
            'avg_processing_time': split_df['processing_time'].mean(),
            'avg_confidence': split_df['confidence'].mean()
        }
    
    # Per-model analysis
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        analysis['per_model_summary'][model] = {
            'samples': len(model_df),
            'avg_wer': model_df['wer'].mean(),
            'std_wer': model_df['wer'].std(),
            'avg_processing_time': model_df['processing_time'].mean(),
            'avg_confidence': model_df['confidence'].mean(),
            'per_split_performance': {}
        }
        
        # Per split for each model
        for split in df['split'].unique():
            split_model_df = model_df[model_df['split'] == split]
            if len(split_model_df) > 0:
                analysis['per_model_summary'][model]['per_split_performance'][split] = {
                    'samples': len(split_model_df),
                    'avg_wer': split_model_df['wer'].mean(),
                    'std_wer': split_model_df['wer'].std()
                }
    
    # Find top errors for manual review
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        top_errors = model_df.nlargest(10, 'wer')
        analysis['top_errors'][model] = []
        
        for _, row in top_errors.iterrows():
            analysis['top_errors'][model].append({
                'split': row['split'],
                'sample_id': row['sample_id'],
                'wer': row['wer'],
                'reference_text': row['reference_text'],
                'asr_text': row['asr_text'],
                'confidence': row['confidence']
            })
    
    # Sample comparisons (random samples for manual review)
    random_samples = df.sample(min(20, len(df)), random_state=42)
    analysis['sample_comparisons'] = []
    
    for _, row in random_samples.iterrows():
        analysis['sample_comparisons'].append({
            'split': row['split'],
            'sample_id': row['sample_id'],
            'model': row['model'],
            'wer': row['wer'],
            'reference_text': row['reference_text'],
            'asr_text': row['asr_text'],
            'confidence': row['confidence'],
            'audio_duration': row['audio_duration'],
            'processing_time': row['processing_time']
        })
    
    return analysis, df

def save_interactive_html(df, output_dir):
    """Create interactive HTML report for manual review."""
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Create visualizations
        
        # 1. WER distribution by model
        fig1 = px.box(df, x='model', y='wer', 
                     title='WER Distribution by Model',
                     color='model')
        
        # 2. Processing time by model
        fig2 = px.box(df, x='model', y='processing_time',
                     title='Processing Time Distribution by Model',
                     color='model')
        
        # 3. WER by split for each model
        fig3 = px.box(df, x='split', y='wer', color='model',
                     title='WER by Dataset Split',
                     facet_col='model')
        
        # 4. Confidence vs WER scatter plot
        fig4 = px.scatter(df, x='confidence', y='wer', color='model',
                         title='Confidence vs WER',
                         hover_data=['reference_text', 'asr_text'])
        
        # 5. Audio duration distribution
        fig5 = px.histogram(df, x='audio_duration', 
                           title='Audio Duration Distribution',
                           nbins=30)
        
        # Create HTML report
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Medical ASR Analysis Report</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .section { margin-bottom: 40px; border-bottom: 1px solid #eee; padding-bottom: 20px; }
                .plot { margin: 20px 0; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .good { color: green; }
                .medium { color: orange; }
                .poor { color: red; }
            </style>
        </head>
        <body>
            <h1>📊 Medical ASR Analysis Report</h1>
            <p>Generated on: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
            
            <div class="section">
                <h2>📈 Overview Statistics</h2>
                <div id="plots"></div>
            </div>
            
            <div class="section">
                <h2>🔍 Top Errors for Manual Review</h2>
                <div id="error-table"></div>
            </div>
            
            <div class="section">
                <h2>📋 Random Samples for Verification</h2>
                <div id="sample-table"></div>
            </div>
            
            <script>
                // Plotly charts
                const plotsDiv = document.getElementById('plots');
                
                // You would need to embed the Plotly figures here
                // For now, we'll create a simple summary
                
                // Error table
                const errorTable = document.getElementById('error-table');
                errorTable.innerHTML = `
                    <table>
                        <tr>
                            <th>Model</th>
                            <th>Split</th>
                            <th>Sample ID</th>
                            <th>WER</th>
                            <th>Reference Text</th>
                            <th>ASR Text</th>
                        </tr>
                        <!-- Data will be populated by JavaScript -->
                    </table>
                `;
                
                // Sample table
                const sampleTable = document.getElementById('sample-table');
                sampleTable.innerHTML = `
                    <table>
                        <tr>
                            <th>Model</th>
                            <th>Split</th>
                            <th>Sample ID</th>
                            <th>WER</th>
                            <th>Audio Duration</th>
                            <th>Reference Text (first 50 chars)</th>
                            <th>ASR Text (first 50 chars)</th>
                        </tr>
                        <!-- Data will be populated by JavaScript -->
                    </table>
                `;
            </script>
        </body>
        </html>
        """
        
        html_path = output_dir / "interactive_report.html"
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"✓ Interactive HTML report saved: {html_path}")
        
    except ImportError:
        logger.warning("Plotly not installed. Skipping interactive HTML report.")
        logger.info("Install with: pip install plotly")

def create_manual_review_csv(df, output_dir):
    """Create CSV files for manual review of transcriptions."""
    
    # Create a comprehensive CSV for manual review
    review_df = df[['split', 'sample_id', 'model', 'wer', 'confidence', 
                    'processing_time', 'audio_duration', 'word_count',
                    'reference_text', 'asr_text']].copy()
    
    # Add quality indicators
    review_df['wer_category'] = pd.cut(review_df['wer'], 
                                       bins=[0, 0.1, 0.3, 1.0],
                                       labels=['Good (≤0.1)', 'Medium (0.1-0.3)', 'Poor (>0.3)'])
    
    review_df['confidence_category'] = pd.cut(review_df['confidence'],
                                              bins=[0, 0.7, 0.9, 1.0],
                                              labels=['Low (≤0.7)', 'Medium (0.7-0.9)', 'High (>0.9)'])
    
    # Sort by WER (highest errors first for review)
    review_df = review_df.sort_values('wer', ascending=False)
    
    # Save
    review_path = output_dir / "manual_review_data.csv"
    review_df.to_csv(review_path, index=False, encoding='utf-8')
    logger.info(f"✓ Manual review data saved: {review_path}")
    
    # Also create a simplified version
    simple_df = review_df[['split', 'sample_id', 'model', 'wer', 'wer_category',
                           'reference_text', 'asr_text']].head(50)  # Top 50 errors
    simple_path = output_dir / "top_errors_for_review.csv"
    simple_df.to_csv(simple_path, index=False, encoding='utf-8')
    logger.info(f"✓ Top 50 errors saved: {simple_path}")
    
    return review_df

# ==================== MAIN EXECUTION ====================

def main():
    print("=" * 80)
    print("COMPLETE MEDICAL ASR DATASET ANALYSIS")
    print("=" * 80)
    print(f"Dataset: {DATASET_PATH}")
    print(f"Splits to analyze: {SPLITS}")
    print(f"Models: {MODELS_TO_TEST}")
    print(f"Sample limit per split: {SAMPLE_LIMIT_PER_SPLIT or 'All'}")
    print("=" * 80)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    dataset = load_entire_dataset()
    
    if not dataset:
        logger.error("No dataset loaded!")
        return
    
    # Initialize models
    models = {}
    for model_name in MODELS_TO_TEST:
        model = initialize_model(model_name)
        if model:
            models[model_name] = model
        else:
            logger.warning(f"Skipping {model_name} - failed to initialize")
    
    if not models:
        logger.error("No models initialized!")
        return
    
    # Process each split with each model
    all_results = []
    
    for split_name, split_info in dataset.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {split_name.upper()} split")
        logger.info(f"{'='*60}")
        
        df = split_info['dataframe']
        
        # Prepare audio data
        audio_arrays = []
        sample_rates = []
        reference_texts = []
        
        logger.info("Loading audio data...")
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Loading audio"):
            audio_item = row['audio']
            text = str(row['text'])
            
            audio_array, sr = load_audio_from_item(audio_item)
            
            if audio_array is not None:
                audio_arrays.append(audio_array)
                sample_rates.append(sr)
                reference_texts.append(text)
            else:
                logger.warning(f"Failed to load audio for sample {idx}")
        
        logger.info(f"✓ Loaded {len(audio_arrays)}/{len(df)} audio samples")
        
        # Process with each model
        for model_name, model in models.items():
            logger.info(f"\nProcessing with {model_name}...")
            
            results = process_batch_with_model(
                model, audio_arrays, sample_rates, reference_texts,
                split_name, model_name
            )
            
            all_results.extend(results)
            
            # Save intermediate results
            split_results_df = pd.DataFrame(results)
            split_output_dir = OUTPUT_DIR / split_name / model_name
            split_output_dir.mkdir(parents=True, exist_ok=True)
            
            split_results_df.to_csv(
                split_output_dir / f"{split_name}_{model_name}_results.csv",
                index=False, encoding='utf-8'
            )
            
            # Calculate and log summary
            avg_wer = split_results_df['wer'].mean()
            avg_time = split_results_df['processing_time'].mean()
            logger.info(f"  ✓ {model_name} on {split_name}:")
            logger.info(f"    Average WER: {avg_wer:.3f}")
            logger.info(f"    Average time: {avg_time:.2f}s")
            logger.info(f"    Samples processed: {len(results)}")
    
    # Generate comprehensive analysis
    logger.info(f"\n{'='*60}")
    logger.info("Generating comprehensive analysis...")
    logger.info(f"{'='*60}")
    
    # Convert all results to DataFrame
    all_results_df = pd.DataFrame(all_results)
    
    # Save all results
    all_results_path = OUTPUT_DIR / "all_transcriptions.csv"
    all_results_df.to_csv(all_results_path, index=False, encoding='utf-8')
    logger.info(f"✓ All transcriptions saved: {all_results_path}")
    
    # Generate analysis
    analysis, analysis_df = generate_detailed_analysis(all_results)
    
    # Save analysis as JSON
    analysis_path = OUTPUT_DIR / "dataset_analysis.json"
    with open(analysis_path, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    logger.info(f"✓ Analysis saved: {analysis_path}")
    
    # Save analysis as CSV
    analysis_csv_path = OUTPUT_DIR / "analysis_summary.csv"
    analysis_df.to_csv(analysis_csv_path, index=False, encoding='utf-8')
    
    # Create manual review CSV
    review_df = create_manual_review_csv(all_results_df, OUTPUT_DIR)
    
    # Try to create interactive HTML report
    try:
        save_interactive_html(all_results_df, OUTPUT_DIR)
    except Exception as e:
        logger.warning(f"Could not create interactive report: {e}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 ANALYSIS COMPLETE")
    print("=" * 80)
    
    # Overall statistics
    total_samples = len(all_results_df)
    unique_samples = all_results_df[['split', 'sample_id']].drop_duplicates().shape[0]
    
    print(f"\n📁 Dataset Processed:")
    for split in SPLITS:
        if split in dataset:
            split_df = all_results_df[all_results_df['split'] == split]
            split_samples = split_df[['split', 'sample_id']].drop_duplicates().shape[0]
            print(f"  • {split.upper()}: {split_samples} samples")
    
    print(f"\n🤖 Models Tested:")
    for model_name in MODELS_TO_TEST:
        model_df = all_results_df[all_results_df['model'] == model_name]
        if len(model_df) > 0:
            avg_wer = model_df['wer'].mean()
            avg_time = model_df['processing_time'].mean()
            print(f"  • {model_name}:")
            print(f"    - Average WER: {avg_wer:.3f}")
            print(f"    - Average time: {avg_time:.2f}s")
    
    print(f"\n📊 Performance Summary:")
    best_model = all_results_df.groupby('model')['wer'].mean().idxmin()
    best_wer = all_results_df.groupby('model')['wer'].mean().min()
    fastest_model = all_results_df.groupby('model')['processing_time'].mean().idxmin()
    fastest_time = all_results_df.groupby('model')['processing_time'].mean().min()
    
    print(f"  • Best accuracy: {best_model} (WER: {best_wer:.3f})")
    print(f"  • Fastest: {fastest_model} ({fastest_time:.2f}s per sample)")
    
    print(f"\n📂 Output Files:")
    print(f"  • All transcriptions: {all_results_path}")
    print(f"  • Analysis summary: {analysis_csv_path}")
    print(f"  • Manual review data: {OUTPUT_DIR}/manual_review_data.csv")
    print(f"  • Top 50 errors: {OUTPUT_DIR}/top_errors_for_review.csv")
    print(f"  • Per-split results: {OUTPUT_DIR}/[split]/[model]/")
    
    print(f"\n🔍 For Manual Review:")
    print(f"  1. Open {OUTPUT_DIR}/top_errors_for_review.csv")
    print(f"  2. Check high-WER samples")
    print(f"  3. Compare reference vs ASR text")
    print(f"  4. Listen to original audio if needed")
    
    print(f"\n🎯 Recommendations:")
    print(f"  1. Focus on samples with WER > 0.3")
    print(f"  2. Check if errors are medical terms")
    print(f"  3. Consider domain-specific fine-tuning")
    print(f"  4. For production: {best_model} (accuracy) or {fastest_model} (speed)")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()