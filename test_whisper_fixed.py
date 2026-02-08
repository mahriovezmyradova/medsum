#!/usr/bin/env python3
"""
Test Whisper with the fix.
"""
import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.append('src')

print("=" * 60)
print("TESTING FIXED WHISPER")
print("=" * 60)

# Test Whisper
try:
    from asr.whisper import WhisperASR
    
    whisper_config = {
        'name': 'whisper_tiny',
        'model_size': 'tiny',
        'language': 'de',
        'device': 'cpu',
        'beam_size': 5,
        'best_of': 5,
        'temperature': 0.0
    }
    
    print("Creating WhisperASR instance...")
    asr = WhisperASR(whisper_config)
    
    print("Loading model...")
    asr.load_model()
    
    print("Testing transcription with synthetic audio...")
    # Create a simple German phrase audio (synthetic)
    sample_rate = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Simple sine wave
    test_audio = 0.1 * np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
    
    result = asr.transcribe(test_audio, sample_rate)
    
    print(f"\n✓ Whisper loaded successfully")
    print(f"✓ Test transcription: '{result.text}'")
    print(f"✓ Confidence: {result.confidence:.3f}")
    print(f"✓ Processing time: {result.processing_time:.2f}s")
    
    # Test with real audio from dataset
    print("\n" + "=" * 40)
    print("TESTING WITH REAL DATASET AUDIO")
    print("=" * 40)
    
    # Load one sample from dataset
    class SimpleConfig:
        class Paths:
            raw_data_dir = Path("/Users/mahriovezmyradova/MultiMed_dataset/German")
        class Dataset:
            splits = ['eval']
            audio_column = 'audio'
            text_column = 'text'
            target_sample_rate = 16000
        paths = Paths()
        dataset = Dataset()
    
    from data.loader import MultiMedLoader
    config = SimpleConfig()
    loader = MultiMedLoader(config)
    samples = loader.load_samples('eval', max_samples=1)
    
    if samples:
        sample = samples[0]
        print(f"\nDataset sample text: {sample.text[:80]}...")
        print(f"Audio duration: {sample.duration}s")
        
        # Transcribe
        result = asr.transcribe(sample.audio.array, sample.audio.sample_rate)
        
        print(f"\n✓ Transcription: '{result.text}'")
        
        # Calculate word accuracy
        ref_words = set(sample.text.lower().split())
        trans_words = set(result.text.lower().split())
        common_words = ref_words.intersection(trans_words)
        accuracy = len(common_words) / len(ref_words) if ref_words else 0
        
        print(f"✓ Word accuracy: {accuracy:.1%}")
        
        # Show some common errors
        missing_words = ref_words - trans_words
        extra_words = trans_words - ref_words
        
        if missing_words:
            print(f"✓ Missing words: {list(missing_words)[:5]}")
        if extra_words:
            print(f"✓ Extra words: {list(extra_words)[:5]}")
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)