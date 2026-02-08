# scripts/test_summarization.py
#!/usr/bin/env python3
"""
Test script for summarization module.
"""
import sys
from pathlib import Path
import json

# Add src to path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root / "src"))

def test_summarization():
    """Test all summarization models."""
    
    print("=" * 70)
    print("SUMMARIZATION MODULE TEST")
    print("=" * 70)
    
    # Test text (German medical conversation)
    test_text = """
    Patient: Guten Tag, Herr Doktor. Ich habe seit drei Tagen starke Kopfschmerzen und Fieber.
    Arzt: Guten Tag. Haben Sie noch andere Symptome?
    Patient: Ja, mir ist oft übel und ich habe Halsschmerzen.
    Arzt: Lassen Sie mich Ihren Hals anschauen. Sie haben eine Rötung im Rachen. 
    Ich vermute eine bakterielle Infektion. Ich verschreibe Ihnen ein Antibiotikum.
    Patient: Wie oft soll ich das einnehmen?
    Arzt: Nehmen Sie 500 mg Amoxicillin dreimal täglich für sieben Tage. 
    Trinken Sie viel Wasser und ruhen Sie sich aus.
    Patient: Vielen Dank, Herr Doktor.
    Arzt: Gern geschehen. Kommen Sie in einer Woche wieder zur Kontrolle.
    """
    
    print(f"Test text length: {len(test_text)} characters")
    print(f"Test text: {test_text[:200]}...\n")
    
    # Test Extractive Summarizer
    print("1. Testing Extractive Summarizer")
    print("-" * 40)
    
    from summarization.extractive import ExtractiveSummarizer
    
    extractive_config = {
        "name": "extractive_bert",
        "language": "de",
        "model_name": "bert-base-german-cased",
        "num_sentences": 3
    }
    
    extractive = ExtractiveSummarizer(extractive_config)
    extractive.load_model()
    
    result = extractive.summarize(test_text)
    print(f"Summary: {result.summary}")
    print(f"Processing time: {result.processing_time:.2f}s")
    print(f"Metadata: {json.dumps(result.metadata, indent=2, ensure_ascii=False)}\n")
    
    # Test Medical Summarizer
    print("2. Testing Medical Summarizer")
    print("-" * 40)
    
    from summarization.medical_summarizer import MedicalSummarizer
    
    medical_config = {
        "name": "medical_summarizer",
        "language": "de",
        "categories": ["symptoms", "diagnosis", "medication", "treatment"]
    }
    
    medical = MedicalSummarizer(medical_config)
    medical.load_model()
    
    result = medical.summarize(test_text)
    print(f"Summary:\n{result.summary}")
    print(f"Processing time: {result.processing_time:.2f}s")
    print(f"Metadata: {json.dumps(result.metadata, indent=2, ensure_ascii=False)}\n")
    
    # Test Evaluation
    print("3. Testing Summarization Evaluation")
    print("-" * 40)
    
    from summarization.evaluation import SummarizationEvaluator
    
    evaluator = SummarizationEvaluator(language="de")
    
    # Create a reference summary
    reference_summary = """
    Der Patient klagt über Kopfschmerzen, Fieber, Übelkeit und Halsschmerzen seit drei Tagen.
    Diagnose: Bakterielle Infektion im Rachen.
    Behandlung: Amoxicillin 500 mg dreimal täglich für 7 Tage.
    Empfehlungen: Viel trinken, sich ausruhen.
    Folgetermin: Kontrolle in einer Woche.
    """
    
    metrics = evaluator.evaluate(reference_summary, result.summary)
    
    print(f"ROUGE-1: {metrics.rouge1:.3f}")
    print(f"ROUGE-2: {metrics.rouge2:.3f}")
    print(f"ROUGE-L: {metrics.rougeL:.3f}")
    print(f"BLEU: {metrics.bleu:.3f}")
    print(f"Compression ratio: {metrics.compression_ratio:.2f}")
    print(f"Medical keyword coverage: {metrics.medical_keyword_coverage:.2f}")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    test_summarization()