#!/usr/bin/env python3
"""
Quick comparison script with simplified audio loading.
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

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add src to path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root / "src"))

def load_audio_direct(audio_item):
    """Load audio directly from parquet format."""
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
    """Calculate Word Error Rate with updated jiwer API."""
    try:
        import jiwer
        
        # New jiwer API
        transformation = jiwer.Compose([
            jiwer.ToLowerCase(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.RemovePunctuation()
        ])
        
        # Apply transformation
        reference_transformed = transformation(reference)
        hypothesis_transformed = transformation(hypothesis)
        
        # Calculate WER
        return jiwer.wer(reference_transformed, hypothesis_transformed)
        
    except Exception as e:
        print(f"WER calculation error: {e}")
        # Fallback simple WER calculation
        ref_words = reference.lower().split()
        hyp_words = hypothesis.lower().split()
        
        # Simple alignment
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
                        dp[i-1][j] + 1,    # deletion
                        dp[i][j-1] + 1,    # insertion
                        dp[i-1][j-1] + 1   # substitution
                    )
        
        errors = dp[n][m]
        return errors / max(n, 1)

def main():
    print("=" * 80)
    print("QUICK ASR COMPARISON - DIRECT AUDIO LOADING")
    print("=" * 80)
    
    # Load data directly
    data_dir = Path("/Users/mahriovezmyradova/MultiMed_dataset/German")
    eval_file = data_dir / "eval-00000-of-00001.parquet"
    
    df = pd.read_parquet(eval_file)
    df = df.head(5)  # Start with just 5 samples for testing
    
    print(f"Loaded {len(df)} samples")
    
    # Prepare data
    reference_texts = []
    audio_arrays = []
    sample_rates = []
    
    for i, row in df.iterrows():
        text = str(row['text'])
        audio_item = row['audio']
        
        audio_array, sr = load_audio_direct(audio_item)
        
        if audio_array is not None:
            reference_texts.append(text)
            audio_arrays.append(audio_array)
            sample_rates.append(sr)
            print(f"✓ Sample {i+1}: {len(audio_array)} samples, {sr}Hz, text: {text[:50]}...")
        else:
            print(f"✗ Sample {i+1}: Failed to load audio")
    
    if not reference_texts:
        print("No audio loaded successfully!")
        return
    
    print(f"\nSuccessfully loaded {len(reference_texts)} audio samples")
    
    # Test Whisper
    print("\n🔊 Testing Whisper...")
    try:
        from asr.whisper import WhisperASR
        
        config = {
            "name": "whisper_base",
            "model_size": "base",
            "language": "de",
            "device": "cpu"
        }
        
        whisper = WhisperASR(config)
        whisper.load_model()
        
        results = []
        
        for i, (audio, sr, ref) in enumerate(zip(audio_arrays, sample_rates, reference_texts)):
            print(f"  Processing sample {i+1}...")
            result = whisper.transcribe(audio, sr)
            wer = calculate_wer(ref, result.text)
            
            print(f"    WER: {wer:.3f}")
            print(f"    Time: {result.processing_time:.2f}s")
            print(f"    ASR: {result.text[:80]}...")
            
            results.append({
                "sample": i+1,
                "wer": wer,
                "time": result.processing_time,
                "asr_text": result.text
            })
        
        if results:
            avg_wer = np.mean([r["wer"] for r in results])
            avg_time = np.mean([r["time"] for r in results])
            print(f"\n✓ Whisper completed:")
            print(f"  Average WER: {avg_wer:.3f}")
            print(f"  Average time: {avg_time:.2f}s")
            
    except Exception as e:
        print(f"❌ Whisper failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("QUICK TEST COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()