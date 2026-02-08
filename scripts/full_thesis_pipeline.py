# scripts/minimal_thesis.py
#!/usr/bin/env python3
"""
MINIMAL thesis pipeline - No complex imports, just works.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import jiwer
import re
import json

# Add src to path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root / "src"))

def calculate_simple_wer(reference, hypothesis):
    """Simple WER calculation."""
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

def find_medical_terms(text):
    """Find medical terms in German text."""
    medical_keywords = {
        'schmerzen', 'schmerz', 'schlucken', 'mund', 'rachen', 'trocken',
        'gurgeln', 'salzwasser', 'salz', 'wasser', 'neugeborenen', 'säuglingen',
        'kleinkindern', 'jugendlichen', 'ibuprofen', 'medikament', 'tablette',
        'behandlung', 'therapie', 'untersuchung', 'diagnose', 'symptom',
        'fieber', 'temperatur', 'kopfschmerzen', 'übelkeit', 'erbrechen',
        'durchfall', 'verstopfung', 'müdigkeit', 'atemnot', 'husten'
    }
    
    words = text.lower().split()
    found = [w for w in words if w in medical_keywords]
    return set(found)

def simple_summary_similarity(summary1, summary2):
    """Calculate simple similarity between summaries."""
    if not summary1 or not summary2:
        return 0.0
    
    words1 = set(summary1.lower().split())
    words2 = set(summary2.lower().split())
    
    if not words1:
        return 0.0
    
    intersection = words1.intersection(words2)
    return len(intersection) / len(words1)

def run_minimal_pipeline():
    """Run minimal thesis pipeline."""
    
    print("=" * 100)
    print("MINIMAL THESIS PIPELINE - GERMAN MEDICAL ASR ANALYSIS")
    print("=" * 100)
    
    # STEP 1: Load data directly
    print("\n1️⃣  Loading data...")
    
    data_dir = Path("/Users/mahriovezmyradova/MultiMed_dataset/German")
    eval_file = data_dir / "eval-00000-of-00001.parquet"
    
    try:
        import pandas as pd
        df = pd.read_parquet(eval_file)
        print(f"✓ Loaded {len(df)} samples from {eval_file.name}")
        
        # Take first 5 samples
        samples = df.head(5).to_dict('records')
        
        # Show samples
        for i, sample in enumerate(samples):
            print(f"  Sample {i+1}: {sample.get('text', '')[:80]}...")
            
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return
    
    # STEP 2: Import ASR (Whisper)
    print("\n2️⃣  Loading Whisper ASR...")
    
    try:
        # Import whisper directly
        import whisper
        import torch
        import numpy as np
        
        # Load model
        model = whisper.load_model("base")
        print("✓ Whisper model loaded")
        
    except Exception as e:
        print(f"❌ Failed to load Whisper: {e}")
        return
    
    # STEP 3: Process each sample
    print(f"\n3️⃣  Processing {len(samples)} conversations...")
    print("-" * 60)
    
    results = []
    
    for i, sample in enumerate(samples):
        print(f"\n📋 Sample {i+1}/{len(samples)}:")
        
        # Get text and audio
        original_text = str(sample.get('text', ''))
        print(f"   Original: {original_text[:80]}...")
        
        # Check for audio data
        audio_data = sample.get('audio')
        if not audio_data:
            print("   ⚠️  No audio data, skipping")
            continue
        
        # Process audio
        try:
            # Extract audio bytes
            if isinstance(audio_data, dict) and 'bytes' in audio_data:
                audio_bytes = audio_data['bytes']
                
                # Load audio
                import io
                import soundfile as sf
                
                audio_file = io.BytesIO(audio_bytes)
                audio_array, sample_rate = sf.read(audio_file)
                
                # Transcribe
                audio_float = audio_array.astype(np.float32)
                if len(audio_float.shape) > 1:
                    audio_float = audio_float.mean(axis=1)  # Convert to mono
                
                # Normalize
                if audio_float.max() > 0:
                    audio_float = audio_float / audio_float.max()
                
                # Transcribe with Whisper
                result = model.transcribe(audio_float, language="de", fp16=False)
                transcription = result["text"]
                
                print(f"   ASR: {transcription[:80]}...")
                
                # Calculate WER
                wer = calculate_simple_wer(original_text, transcription)
                
                # Find medical terms
                original_terms = find_medical_terms(original_text)
                asr_terms = find_medical_terms(transcription)
                
                medical_error_rate = 0
                if original_terms:
                    missing_terms = original_terms - asr_terms
                    medical_error_rate = len(missing_terms) / len(original_terms)
                
                # Simple summarization (first 2 sentences)
                sentences = re.split(r'[.!?]+', original_text)
                original_summary = ' '.join([s.strip() for s in sentences[:2] if s.strip()])
                
                asr_sentences = re.split(r'[.!?]+', transcription)
                asr_summary = ' '.join([s.strip() for s in asr_sentences[:2] if s.strip()])
                
                # Calculate summary similarity
                summary_similarity = simple_summary_similarity(original_summary, asr_summary)
                
                # Store results
                result = {
                    'sample_id': i + 1,
                    'original_text': original_text,
                    'asr_transcription': transcription,
                    'wer': wer,
                    'medical_terms_original': len(original_terms),
                    'medical_terms_asr': len(asr_terms),
                    'medical_error_rate': medical_error_rate,
                    'original_summary': original_summary,
                    'asr_summary': asr_summary,
                    'summary_similarity': summary_similarity,
                    'summary_impact': wer * (1 - summary_similarity)
                }
                
                results.append(result)
                
                print(f"   📊 WER: {wer:.3f}, Medical errors: {medical_error_rate:.3f}")
                print(f"   📝 Summary similarity: {summary_similarity:.3f}")
                
                # Show example differences
                if wer > 0:
                    ref_words = original_text.lower().split()[:10]
                    hyp_words = transcription.lower().split()[:10]
                    
                    for j in range(min(len(ref_words), len(hyp_words))):
                        if ref_words[j] != hyp_words[j]:
                            print(f"   ❌ '{ref_words[j]}' → '{hyp_words[j]}'")
                            break
                
            else:
                print("   ⚠️  Audio data in unexpected format")
                
        except Exception as e:
            print(f"   ❌ Processing error: {e}")
            import traceback
            traceback.print_exc()
    
    # STEP 4: Analyze results
    print("\n4️⃣  Analyzing results...")
    print("-" * 60)
    
    if results:
        df_results = pd.DataFrame(results)
        
        # Save results
        output_dir = project_root / "data" / "outputs" / "minimal_thesis"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / "results.csv"
        df_results.to_csv(output_file, index=False, encoding='utf-8')
        
        # Calculate statistics
        stats = {
            'Average WER': df_results['wer'].mean(),
            'Average Medical Error Rate': df_results['medical_error_rate'].mean(),
            'Average Summary Similarity': df_results['summary_similarity'].mean(),
            'Average Summary Impact': df_results['summary_impact'].mean(),
            'Correlation (WER vs Similarity)': df_results['wer'].corr(df_results['summary_similarity'])
        }
        
        print("\n📊 KEY STATISTICS:")
        print("-" * 40)
        for key, value in stats.items():
            print(f"{key:35}: {value:.3f}")
        
        # Save statistics
        stats_file = output_dir / "statistics.txt"
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("THESIS STATISTICS - MINIMAL ANALYSIS\n")
            f.write("=" * 50 + "\n\n")
            for key, value in stats.items():
                f.write(f"{key}: {value:.4f}\n")
        
        # Generate insights
        print("\n💡 THESIS INSIGHTS:")
        print("-" * 40)
        
        avg_wer = stats['Average WER']
        avg_sim = stats['Average Summary Similarity']
        correlation = stats['Correlation (WER vs Similarity)']
        
        if avg_wer < 0.1:
            wer_insight = "✅ Low WER: ASR performs well"
        elif avg_wer < 0.3:
            wer_insight = "⚠️  Moderate WER: Room for improvement"
        else:
            wer_insight = "❌ High WER: Significant transcription errors"
        
        if avg_sim > 0.8:
            sim_insight = "✅ High similarity: Summarization robust to errors"
        elif avg_sim > 0.6:
            sim_insight = "⚠️  Moderate similarity: Summarization affected"
        else:
            sim_insight = "❌ Low similarity: Summarization significantly degraded"
        
        if correlation < -0.5:
            corr_insight = "📈 Strong negative correlation: Higher WER → Worse summaries"
        elif correlation < -0.3:
            corr_insight = "📊 Moderate correlation: ASR errors affect summaries"
        else:
            corr_insight = "🔍 Weak correlation: Other factors may dominate"
        
        print(f"1. ASR Performance: {wer_insight}")
        print(f"2. Summary Quality: {sim_insight}")
        print(f"3. Correlation: {corr_insight}")
        
        # Show detailed examples
        print("\n🔍 EXAMPLE ANALYSIS:")
        print("-" * 40)
        
        if len(results) > 0:
            sample = results[0]
            print(f"Sample 1 - Medical Conversation:")
            print(f"  Original: {sample['original_text'][:100]}...")
            print(f"  ASR: {sample['asr_transcription'][:100]}...")
            print(f"  WER: {sample['wer']:.3f}")
            print(f"  Medical terms: {sample['medical_terms_original']} original, {sample['medical_terms_asr']} in ASR")
            print(f"  Summary similarity: {sample['summary_similarity']:.3f}")
        
        print(f"\n💾 Results saved to: {output_file}")
        print(f"📈 Statistics saved to: {stats_file}")
        
        # Create thesis-ready table
        print("\n📋 THESIS-READY RESULTS TABLE:")
        print("-" * 80)
        print(f"{'Sample':10} {'WER':10} {'Med Error':10} {'Summary Sim':15} {'Impact':10}")
        print("-" * 80)
        
        for i, result in enumerate(results):
            print(f"{i+1:10} {result['wer']:10.3f} {result['medical_error_rate']:10.3f} "
                  f"{result['summary_similarity']:15.3f} {result['summary_impact']:10.3f}")
        
    else:
        print("❌ No results generated")
    
    print("\n" + "=" * 100)
    print("🎓 MINIMAL ANALYSIS COMPLETE - READY FOR THESIS!")
    print("=" * 100)
    print("\nYou now have empirical evidence for your research question:")
    print('"How does transcription quality affect medical summary accuracy in German doctor-patient conversations?"')
    print("\nKey metrics for your thesis:")
    print("1. Word Error Rate (WER): Measures ASR accuracy")
    print("2. Medical term error rate: Specialized metric for medical domain")
    print("3. Summary similarity: How much summaries differ due to ASR errors")
    print("4. Correlation: Statistical relationship between WER and summary quality")

if __name__ == "__main__":
    run_minimal_pipeline()