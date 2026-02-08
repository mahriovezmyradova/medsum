#!/usr/bin/env python3
"""
Small-scale evaluation of ASR models.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
from tqdm import tqdm

# Add src to path
sys.path.append('src')

print("=" * 70)
print("SMALL-SCALE ASR EVALUATION")
print("=" * 70)

# Setup
class SimpleConfig:
    class Paths:
        raw_data_dir = Path("/Users/mahriovezmyradova/MultiMed_dataset/German")
        outputs_dir = Path("data/outputs")
        
        def create_directories(self):
            self.outputs_dir.mkdir(parents=True, exist_ok=True)
    
    class Dataset:
        splits = ['eval']
        audio_column = 'audio'
        text_column = 'text'
        duration_column = 'duration'
        max_samples_per_split = 20  # Small for testing
        test_split = 'eval'
        random_seed = 42
        target_sample_rate = 16000
        normalize_audio = True
        remove_silence = False
        
        @property
        def split_files(self):
            return {split: f'{split}-00000-of-00001.parquet' for split in self.splits}
    
    paths = Paths()
    dataset = Dataset()
    
    # Create output directory
    paths.create_directories()

config = SimpleConfig()

# Load data
print("\n1. Loading data...")
from data.loader import MultiMedLoader
loader = MultiMedLoader(config)
samples = loader.load_samples('eval', max_samples=10)  # Only 10 samples for testing

print(f"✓ Loaded {len(samples)} samples for evaluation")

# Prepare data
reference_texts = [s.text for s in samples]
audio_arrays = [s.audio.array for s in samples]
sample_rates = [s.audio.sample_rate for s in samples]

print(f"✓ Average duration: {np.mean([s.duration for s in samples]):.1f}s")
print(f"✓ Total audio time: {np.sum([s.duration for s in samples]):.1f}s")

# Initialize models
print("\n2. Initializing ASR models...")
models = {}

# Whisper models
whisper_configs = [
    {
        'name': 'whisper_tiny',
        'model_size': 'tiny',
        'language': 'de',
        'device': 'cpu'
    },
    {
        'name': 'whisper_base', 
        'model_size': 'base',
        'language': 'de',
        'device': 'cpu'
    }
]

for whisper_config in whisper_configs:
    try:
        from asr.whisper import WhisperASR
        model = WhisperASR(whisper_config)
        models[whisper_config['name']] = model
        print(f"✓ Initialized {whisper_config['name']}")
    except Exception as e:
        print(f"✗ Failed to initialize {whisper_config['name']}: {e}")

# Wav2Vec2 model
try:
    from asr.wav2vec2 import Wav2Vec2ASR
    wav2vec_config = {
        'name': 'wav2vec2_german',
        'model_name': 'facebook/wav2vec2-large-xlsr-53-german',
        'language': 'de',
        'device': 'cpu'
    }
    model = Wav2Vec2ASR(wav2vec_config)
    models[wav2vec_config['name']] = model
    print(f"✓ Initialized {wav2vec_config['name']}")
except Exception as e:
    print(f"✗ Failed to initialize Wav2Vec2: {e}")

if not models:
    print("❌ No models initialized. Exiting.")
    sys.exit(1)

print(f"✓ Total models: {len(models)}")

# Run evaluations
print("\n3. Running evaluations...")
results = []

for model_name, model in models.items():
    print(f"\nEvaluating {model_name}...")
    
    try:
        # Load model
        print(f"  Loading {model_name}...")
        model.load_model()
        
        # Transcribe
        transcripts = []
        processing_times = []
        
        for i, (audio, sr) in enumerate(tqdm(zip(audio_arrays, sample_rates), 
                                            total=len(audio_arrays), 
                                            desc=f"  Transcribing")):
            start_time = pd.Timestamp.now()
            result = model.transcribe(audio, sr)
            end_time = pd.Timestamp.now()
            
            transcripts.append(result.text)
            processing_times.append((end_time - start_time).total_seconds())
        
        # Calculate metrics
        from evaluation.metrics import ASREvaluator
        evaluator = ASREvaluator(language="de")
        metrics = evaluator.evaluate(reference_texts, transcripts, compute_ci=False)
        
        # Store results
        results.append({
            'model': model_name,
            'wer': metrics.wer,
            'cer': metrics.cer,
            'mer': metrics.mer,
            'wil': metrics.wil,
            'f1': metrics.f1,
            'precision': metrics.precision,
            'recall': metrics.recall,
            'bleu': metrics.bleu if metrics.bleu else 0,
            'rouge1': metrics.rouge['rouge1'] if metrics.rouge else 0,
            'avg_processing_time': np.mean(processing_times),
            'total_processing_time': np.sum(processing_times),
            'samples_evaluated': len(samples)
        })
        
        print(f"  ✓ WER: {metrics.wer:.3f}")
        print(f"  ✓ CER: {metrics.cer:.3f}")
        print(f"  ✓ F1: {metrics.f1:.3f}")
        print(f"  ✓ Avg processing time: {np.mean(processing_times):.2f}s")
        
        # Save sample transcripts
        output_dir = config.paths.outputs_dir / "asr_evaluation"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        transcripts_file = output_dir / f"{model_name}_transcripts.json"
        with open(transcripts_file, 'w', encoding='utf-8') as f:
            json.dump({
                'model': model_name,
                'samples': [
                    {
                        'reference': ref,
                        'transcription': trans,
                        'processing_time': time
                    }
                    for ref, trans, time in zip(reference_texts, transcripts, processing_times)
                ]
            }, f, indent=2, ensure_ascii=False)
        
        print(f"  ✓ Saved transcripts to {transcripts_file}")
        
    except Exception as e:
        print(f"  ✗ Evaluation failed for {model_name}: {e}")
        import traceback
        traceback.print_exc()

# Save results
if results:
    print("\n4. Saving results...")
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('wer')  # Sort by WER (lower is better)
    
    results_file = config.paths.outputs_dir / "asr_evaluation" / "results_summary.csv"
    results_df.to_csv(results_file, index=False)
    
    print(f"✓ Results saved to {results_file}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    
    print(f"\nModels evaluated: {len(results)}")
    print(f"Samples per model: {results[0]['samples_evaluated']}")
    print(f"Total audio time: {np.sum([s.duration for s in samples]):.1f}s")
    
    print("\nRanked by WER (lower is better):")
    print("-" * 80)
    print(f"{'Model':20} {'WER':>8} {'CER':>8} {'F1':>8} {'Time/sample':>12}")
    print("-" * 80)
    
    for _, row in results_df.iterrows():
        print(f"{row['model']:20} {row['wer']:8.3f} {row['cer']:8.3f} "
              f"{row['f1']:8.3f} {row['avg_processing_time']:12.2f}s")
    
    best_model = results_df.iloc[0]
    print("\n" + "=" * 70)
    print(f"🏆 BEST MODEL: {best_model['model']}")
    print(f"   WER: {best_model['wer']:.3f}")
    print(f"   CER: {best_model['cer']:.3f}")
    print(f"   F1: {best_model['f1']:.3f}")
    print(f"   Processing time: {best_model['avg_processing_time']:.2f}s per sample")
    print("=" * 70)
    
    # Save detailed report
    report = f"""# ASR Evaluation Report

## Summary
- **Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Models evaluated**: {len(results)}
- **Samples per model**: {results[0]['samples_evaluated']}
- **Total audio time**: {np.sum([s.duration for s in samples]):.1f}s

## Results
{results_df.to_markdown(index=False)}

## Best Model
**{best_model['model']}** performed best with:
- Word Error Rate (WER): {best_model['wer']:.3f}
- Character Error Rate (CER): {best_model['cer']:.3f}
- F1 Score: {best_model['f1']:.3f}
- Processing time: {best_model['avg_processing_time']:.2f}s per sample

## Notes
- WER/CER: Lower is better (0.0 = perfect transcription)
- F1: Higher is better (1.0 = perfect)
- Evaluation performed on German medical conversation dataset
"""
    
    report_file = config.paths.outputs_dir / "asr_evaluation" / "evaluation_report.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n📄 Detailed report saved to: {report_file}")

print("\n✅ Evaluation complete!")