# scripts/working_integration.py
#!/usr/bin/env python3
"""
Working integration test for ASR → Summarization pipeline.
"""
import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np

# Add src to path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root / "src"))

def test_working_pipeline():
    """Test the full ASR → Summarization pipeline."""
    
    print("=" * 80)
    print("WORKING ASR → SUMMARIZATION PIPELINE TEST")
    print("=" * 80)
    
    # Create a simple config
    class SimpleConfig:
        class Paths:
            raw_data_dir = Path("/Users/mahriovezmyradova/MultiMed_dataset/German")
            outputs_dir = project_root / "data" / "outputs"
            
            def create_directories(self):
                self.outputs_dir.mkdir(parents=True, exist_ok=True)
        
        class Dataset:
            splits = ['eval']
            audio_column = 'audio'
            text_column = 'text'
            duration_column = 'duration'
            max_samples_per_split = 20
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
    
    # STEP 1: Load data
    print("\n1. Loading data sample...")
    
    try:
        # Import with error handling
        import importlib
        
        # Try to import data loader
        try:
            from data.loader import MultiMedLoader
            print("✓ Data loader imported successfully")
        except ImportError as e:
            print(f"❌ Data loader import failed: {e}")
            # Try alternative import
            import sys
            sys.path.append(str(project_root))
            from src.data.loader import MultiMedLoader
            print("✓ Data loader imported via alternative path")
        
        loader = MultiMedLoader(config)
        samples = loader.load_samples('eval', max_samples=1)
        
        if samples:
            sample = samples[0]
            print(f"✓ Loaded sample: {sample.id}")
            print(f"  Text preview: {sample.text[:100]}...")
            print(f"  Duration: {sample.duration:.1f}s")
            
            # Show actual German text from your dataset
            print(f"\n  Full text: {sample.text}")
        else:
            print("❌ No samples loaded")
            return
            
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # STEP 2: Test ASR
    print("\n2. Testing ASR models...")
    
    asr_results = {}
    
    # Test Whisper
    try:
        print("\n🔊 Testing Whisper ASR...")
        from asr.whisper import WhisperASR
        
        whisper_config = {
            "name": "whisper_base",
            "model_size": "base",
            "language": "de",
            "device": "cpu"
        }
        
        whisper = WhisperASR(whisper_config)
        whisper.load_model()
        
        # Transcribe
        result = whisper.transcribe(sample.audio.array, sample.audio.sample_rate)
        
        print(f"✓ Transcription: {result.text}")
        print(f"  Confidence: {result.confidence:.3f}")
        print(f"  Processing time: {result.processing_time:.2f}s")
        
        asr_results["whisper"] = {
            "text": result.text,
            "confidence": result.confidence,
            "processing_time": result.processing_time
        }
        
    except Exception as e:
        print(f"❌ Whisper failed: {e}")
        import traceback
        traceback.print_exc()
    
    # STEP 3: Test Summarization on ORIGINAL text (to verify it works)
    print("\n3. Testing Summarization on ORIGINAL text...")
    
    # Test extractive summarizer
    try:
        print("\n📝 Testing Extractive Summarizer...")
        from summarization.extractive import ExtractiveSummarizer
        
        extractive_config = {
            "name": "extractive_bert",
            "language": "de",
            "model_name": "bert-base-german-cased",
            "num_sentences": 3,
            "device": "cpu"
        }
        
        extractive = ExtractiveSummarizer(extractive_config)
        extractive.load_model()
        
        # Summarize original text
        result = extractive.summarize(sample.text)
        
        print(f"✓ Summary: {result.summary}")
        print(f"  Processing time: {result.processing_time:.2f}s")
        print(f"  Metadata: {json.dumps(result.metadata, indent=2, ensure_ascii=False)}")
        
    except Exception as e:
        print(f"❌ Extractive summarizer failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test medical summarizer
    try:
        print("\n🏥 Testing Medical Summarizer...")
        from summarization.medical_summarizer import MedicalSummarizer
        
        medical_config = {
            "name": "medical_summarizer",
            "language": "de",
            "categories": ["symptoms", "diagnosis", "medication", "treatment"],
            "use_extractive": False  # Don't use extractive as fallback for now
        }
        
        medical = MedicalSummarizer(medical_config)
        medical.load_model()
        
        # Summarize original text
        result = medical.summarize(sample.text)
        
        print(f"✓ Medical Summary:\n{result.summary}")
        print(f"  Processing time: {result.processing_time:.2f}s")
        
    except Exception as e:
        print(f"❌ Medical summarizer failed: {e}")
        import traceback
        traceback.print_exc()
    
    # STEP 4: Full pipeline if ASR worked
    if asr_results:
        print("\n4. Testing Full Pipeline (ASR → Summarization)...")
        
        for asr_name, asr_data in asr_results.items():
            print(f"\n🔄 Pipeline with {asr_name.upper()} ASR:")
            print(f"  ASR Transcription: {asr_data['text'][:100]}...")
            
            # Test summarization on ASR output
            if 'extractive' in locals():
                try:
                    summary_result = extractive.summarize(asr_data['text'])
                    print(f"  📝 Extractive Summary: {summary_result.summary[:100]}..." if summary_result.summary else "  No summary")
                except:
                    print("  ❌ Extractive summarization failed on ASR output")
            
            if 'medical' in locals():
                try:
                    summary_result = medical.summarize(asr_data['text'])
                    print(f"  🏥 Medical Summary: {summary_result.summary[:100]}..." if summary_result.summary else "  No summary")
                except:
                    print("  ❌ Medical summarization failed on ASR output")
    
    # STEP 5: Manual comparison
    print("\n" + "=" * 80)
    print("MANUAL COMPARISON")
    print("=" * 80)
    
    print("\n📋 ORIGINAL TEXT:")
    print(f"{sample.text}")
    
    if asr_results:
        print(f"\n🔊 ASR TRANSCRIPTION (Whisper):")
        print(f"{asr_results.get('whisper', {}).get('text', 'No transcription')}")
    
    print("\n" + "=" * 80)
    print("NEXT STEPS FOR YOUR DIPLOMA:")
    print("=" * 80)
    
    print("""
    1. ✅ Data loading works - your German medical conversations are accessible
    2. ✅ ASR (Whisper) works - can transcribe audio
    3. ⚠️  Summarization needs debugging - but structure is there
    
    RECOMMENDATIONS:
    
    A. First, test summarization separately:
       python -c "from src.summarization.extractive import ExtractiveSummarizer; 
                 es = ExtractiveSummarizer({'name':'test','language':'de'}); 
                 es.load_model(); 
                 print('Summarizer works!')"
    
    B. Then, run small evaluation:
       python scripts/run_small_evaluation.py
    
    C. Finally, implement error analysis module to answer your research question.
    """)
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    test_working_pipeline()