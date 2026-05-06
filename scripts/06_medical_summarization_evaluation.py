#!/usr/bin/env python3
"""
CORRECTED MEDICAL SUMMARIZATION ANALYSIS
Summarize ASR text, compare to reference text
With proper prompting to avoid hallucinations
"""

import os
import logging
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import warnings
import time
from tqdm import tqdm
import json
from rouge_score import rouge_scorer
import nltk
from nltk.tokenize import sent_tokenize

warnings.filterwarnings('ignore')
logging.getLogger('absl').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.WARNING)
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

# Download NLTK data
for _pkg in ('punkt', 'punkt_tab'):
    try:
        nltk.download(_pkg, quiet=True)
    except Exception:
        pass

# Device
def _get_device():
    if torch.cuda.is_available(): return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): return 'mps'
    return 'cpu'
DEVICE = _get_device()

# ================================
# CONFIGURATION
# ================================
INPUT_PATH = Path("data/outputs/full_dataset_analysis/all_transcriptions.csv")
WAV2VEC_PATH = Path("data/outputs/full_dataset_analysis/test/wav2vec2_german/test_wav2vec2_german_results.csv")
OUTPUT_DIR = Path("data/outputs/summarization_final")

# MODELS
SUMMARIZATION_MODELS = [
    {
        "name": "bert_extractive",
        "type": "extractive",
        "description": "BERT-based extractive summarization"
    },
    {
        "name": "bart_german",
        "model_name": "philschmid/bart-large-german-samsum",
        "type": "bart",
        "description": "BART German dialogue-tuned"
    },
    {
        "name": "mt5_base",
        "model_name": "google/mt5-base",
        "type": "mt5",
        "prompt": "zusammenfassen: "  # German prefix — do NOT use English "summarize:"
    }
]

# ASR MODELS (must match model_name column in the CSV)
ASR_MODELS = [
    "whisper_tiny",
    "whisper_base",
    "whisper_small",
    "wav2vec2_german",
]

# SAMPLES per ASR model (None = all)
SAMPLES_PER_MODEL = None  # set to e.g. 100 for a quick test

# Medical terms
MEDICAL_TERMS = [
    "diagnose", "krankheit", "erkrankung", "syndrom", "infektion",
    "krebs", "tumor", "diabetes", "hypertonie", "arthrose",
    "schmerz", "fieber", "husten", "übelkeit", "schwindel",
    "therapie", "behandlung", "medikament", "tablette", "dosis",
    "operation", "rezept", "arzt", "patient", "klinik"
]

# ================================
# BERT EXTRACTIVE SUMMARIZER
# ================================

class BertExtractiveSummarizer:
    def __init__(self):
        print(f"\n🔄 Loading BERT extractive...")
        self.sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=DEVICE)
        print(f"✓ Loaded BERT extractive on {DEVICE}")
    
    def summarize(self, text, num_sentences=3):
        """Extract most important sentences"""
        try:
            sentences = sent_tokenize(text)
            
            if len(sentences) <= num_sentences:
                return text
            
            # Get embeddings
            embeddings = self.sentence_model.encode(sentences)
            
            # Calculate centrality
            sim_matrix = cosine_similarity(embeddings)
            scores = sim_matrix.sum(axis=1) - np.diag(sim_matrix)
            
            # Get top sentences
            top_indices = scores.argsort()[-num_sentences:][::-1]
            top_indices.sort()
            
            summary = ' '.join([sentences[i] for i in top_indices])
            return summary
            
        except Exception as e:
            print(f"BERT error: {e}")
            return text[:200]  # Fallback

# ================================
# BART GERMAN SUMMARIZER
# ================================

class BartSummarizer:
    def __init__(self, model_name):
        self.model_name = model_name
        print(f"\n🔄 Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(DEVICE)
        self.model.eval()
        print(f"✓ Loaded {model_name} on {DEVICE}")

    def summarize(self, text, max_length=150, min_length=25):
        try:
            text = ' '.join(text.split())
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)
            with torch.no_grad():
                ids = self.model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    num_beams=4,
                    max_length=max_length,
                    min_length=min_length,
                    length_penalty=2.0,
                    no_repeat_ngram_size=3,
                    early_stopping=True
                )
            return self.tokenizer.decode(ids[0], skip_special_tokens=True).strip()
        except Exception as e:
            print(f"BART error: {e}")
            return ""


# ================================
# mT5 SUMMARIZER - FIXED PROMPT
# ================================

class MT5Summarizer:
    def __init__(self, model_name, prompt="zusammenfassen: "):
        self.model_name = model_name
        self.prompt = prompt
        print(f"\n🔄 Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(DEVICE)
        self.model.eval()
        print(f"✓ Loaded {model_name} on {DEVICE}")
    
    def summarize(self, text, max_length=100, min_length=20):
        """Generate summary with proper prompting"""
        try:
            # Clean and truncate text
            text = ' '.join(text.split())  # Normalize whitespace
            if len(text) > 512:
                text = text[:512]
            
            # CRITICAL: Proper prompt format for mT5
            input_text = f"{self.prompt}{text}"
            
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            ).to(DEVICE)
            
            with torch.no_grad():
                summary_ids = self.model.generate(
                    inputs['input_ids'],
                    num_beams=4,
                    max_length=max_length,
                    min_length=min_length,
                    length_penalty=2.0,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                    temperature=1.0,
                    do_sample=False
                )
            
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            
            # Clean up summary
            summary = summary.strip()
            if not summary or len(summary) < 5:
                return "[Summary too short]"
            
            return summary
            
        except Exception as e:
            print(f"mT5 error: {e}")
            return ""

# ================================
# METRICS
# ================================

class Metrics:
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True
        )
    
    def calculate(self, reference, hypothesis):
        metrics = {}
        
        # ROUGE scores
        try:
            rouge = self.rouge_scorer.score(reference, hypothesis)
            metrics['rouge1_f1'] = rouge['rouge1'].fmeasure
            metrics['rouge2_f1'] = rouge['rouge2'].fmeasure
            metrics['rougeL_f1'] = rouge['rougeL'].fmeasure
            metrics['rouge_avg'] = np.mean([
                metrics['rouge1_f1'],
                metrics['rouge2_f1'],
                metrics['rougeL_f1']
            ])
        except:
            metrics['rouge1_f1'] = 0
            metrics['rouge2_f1'] = 0
            metrics['rougeL_f1'] = 0
            metrics['rouge_avg'] = 0
        
        # Medical terms preservation
        ref_lower = reference.lower()
        hyp_lower = hypothesis.lower()
        
        ref_terms = [t for t in MEDICAL_TERMS if t in ref_lower]
        hyp_terms = [t for t in MEDICAL_TERMS if t in hyp_lower]
        
        if len(ref_terms) > 0:
            overlap = len(set(ref_terms) & set(hyp_terms))
            metrics['medical_preservation'] = overlap / len(ref_terms)
            metrics['medical_precision'] = overlap / max(len(hyp_terms), 1)
            metrics['medical_recall'] = overlap / len(ref_terms)
        else:
            metrics['medical_preservation'] = 1.0 if len(hyp_terms) == 0 else 0.0
            metrics['medical_precision'] = 1.0 if len(hyp_terms) == 0 else 0.0
            metrics['medical_recall'] = 1.0 if len(hyp_terms) == 0 else 0.0
        
        # Length metrics
        metrics['ref_words'] = len(reference.split())
        metrics['hyp_words'] = len(hypothesis.split())
        metrics['compression_ratio'] = metrics['hyp_words'] / max(metrics['ref_words'], 1)
        
        return metrics

# ================================
# MAIN
# ================================

def main():
    print("=" * 80)
    print("MEDICAL SUMMARIZATION ANALYSIS - FINAL VERSION")
    print("=" * 80)
    print(f"ASR Models: {len(ASR_MODELS)}")
    print(f"Summarizers: {len(SUMMARIZATION_MODELS)}")
    print(f"Samples per ASR: {SAMPLES_PER_MODEL}")
    print(f"Total summaries: {len(ASR_MODELS) * len(SUMMARIZATION_MODELS) * SAMPLES_PER_MODEL}")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"\n📂 Loading data from {INPUT_PATH}...")
    df = pd.read_csv(INPUT_PATH, encoding='utf-8')
    df = df.rename(columns={"model_name": "model_name"})  # already correct name
    print(f"Loaded {len(df)} rows from whisper results")

    # Also load wav2vec2 results if available
    if WAV2VEC_PATH.exists():
        wv_df = pd.read_csv(WAV2VEC_PATH, encoding='utf-8')
        wv_df = wv_df.rename(columns={"model": "model_name"})
        wv_df["model_name"] = "wav2vec2_german"
        wv_df["split"] = "test"
        df = pd.concat([df, wv_df], ignore_index=True)
        print(f"Added {len(wv_df)} wav2vec2_german rows — total: {len(df)}")

    print(f"Models in data: {df['model_name'].unique().tolist()}")
    
    # Initialize
    metrics_calc = Metrics()
    
    # Load summarizers
    print("\n🔄 Loading summarizers...")
    summarizers = []
    
    # BERT extractive
    try:
        summarizers.append({
            'instance': BertExtractiveSummarizer(),
            'name': 'bert_extractive',
            'type': 'extractive'
        })
        print("✓ Added bert_extractive")
    except Exception as e:
        print(f"✗ bert_extractive failed: {e}")

    # BART German
    try:
        summarizers.append({
            'instance': BartSummarizer('philschmid/bart-large-german-samsum'),
            'name': 'bart_german',
            'type': 'abstractive'
        })
        print("✓ Added bart_german")
    except Exception as e:
        print(f"✗ bart_german failed: {e}")

    # mT5 base
    try:
        summarizers.append({
            'instance': MT5Summarizer('google/mt5-base', 'zusammenfassen: '),
            'name': 'mt5_base',
            'type': 'abstractive'
        })
        print("✓ Added mt5_base")
    except Exception as e:
        print(f"✗ mt5_base failed: {e}")
    
    print(f"\n✓ Loaded {len(summarizers)} summarizers")
    
    # Results storage
    all_results = []
    detailed_results = []
    
    # Process each ASR model
    for asr_model in ASR_MODELS:
        if asr_model not in df['model_name'].unique():
            print(f"\n⚠️ {asr_model} not found in data — skipping")
            continue

        print(f"\n{'='*60}")
        print(f"📊 PROCESSING: {asr_model}")
        print(f"{'='*60}")

        # Get test samples (prefer test split)
        model_df = df[df['model_name'] == asr_model]
        test_df = model_df[model_df['split'] == 'test']
        if len(test_df) == 0:
            test_df = model_df[model_df['split'] == 'eval']
        if len(test_df) == 0:
            test_df = model_df

        # Apply sample limit
        if SAMPLES_PER_MODEL is not None:
            n_samples = min(SAMPLES_PER_MODEL, len(test_df))
            sampled = test_df.sample(n=n_samples, random_state=42)
        else:
            sampled = test_df
            n_samples = len(sampled)
        
        print(f"Testing on {n_samples} samples")
        
        # Process each sample
        for idx, row in tqdm(sampled.iterrows(), total=len(sampled), desc=asr_model[:15]):
            asr_text = str(row['asr_text']) if not pd.isna(row['asr_text']) else ""
            reference = str(row['reference_text']) if not pd.isna(row['reference_text']) else ""
            
            if len(asr_text.strip()) < 10 or len(reference.strip()) < 10:
                continue
            
            for summ in summarizers:
                try:
                    start = time.time()
                    
                    # Generate summary from ASR text
                    if summ['type'] == 'extractive':
                        summary = summ['instance'].summarize(asr_text, num_sentences=3)
                    elif summ['type'] == 'bart':
                        summary = summ['instance'].summarize(asr_text)
                    else:  # mt5
                        summary = summ['instance'].summarize(asr_text)
                    
                    proc_time = time.time() - start
                    
                    # Calculate metrics (compare summary to reference)
                    metrics = metrics_calc.calculate(reference, summary)
                    
                    # Store result
                    result = {
                        'asr_model': asr_model,
                        'split': row.get('split', 'test'),
                        'sample_id': row.get('sample_id', idx),
                        'summarizer': summ['name'],
                        'summarizer_type': summ['type'],
                        
                        # ROUGE scores
                        'rouge1_f1': metrics['rouge1_f1'],
                        'rouge2_f1': metrics['rouge2_f1'],
                        'rougeL_f1': metrics['rougeL_f1'],
                        'rouge_avg': metrics['rouge_avg'],
                        
                        # Medical metrics
                        'medical_preservation': metrics['medical_preservation'],
                        'medical_precision': metrics['medical_precision'],
                        'medical_recall': metrics['medical_recall'],
                        
                        # Length metrics
                        'compression_ratio': metrics['compression_ratio'],
                        'ref_words': metrics['ref_words'],
                        'hyp_words': metrics['hyp_words'],
                        
                        # Performance
                        'processing_time': proc_time,
                        
                        # ASR quality
                        'asr_wer': row.get('wer', None),
                        'asr_medical_wer': row.get('medical_wer', None),
                        
                        # Full texts for verification
                        'asr_text': asr_text[:500],  # Store first 500 chars
                        'reference_text': reference[:500],
                        'summary': summary[:500]
                    }
                    
                    all_results.append(result)
                    
                    # Store detailed for first few
                    if len(detailed_results) < 20:
                        detailed_results.append({
                            'asr_model': asr_model,
                            'summarizer': summ['name'],
                            'asr_text_preview': asr_text[:200],
                            'reference_preview': reference[:200],
                            'summary_preview': summary[:200],
                            'rouge_avg': metrics['rouge_avg'],
                            'medical_preservation': metrics['medical_preservation']
                        })
                    
                except Exception as e:
                    print(f"\nError with {summ['name']}: {e}")
                    continue
    
    # Save and display results
    if all_results:
        results_df = pd.DataFrame(all_results)
        detailed_df = pd.DataFrame(detailed_results)
        
        # Save everything
        results_df.to_csv(OUTPUT_DIR / "all_summaries_final.csv", index=False, encoding='utf-8')
        detailed_df.to_csv(OUTPUT_DIR / "example_summaries_final.csv", index=False, encoding='utf-8')
        
        print("\n" + "=" * 80)
        print("📊 FINAL RESULTS")
        print("=" * 80)
        
        # Overall by summarizer
        print("\n🏆 OVERALL PERFORMANCE:")
        print("-" * 60)
        
        overall = results_df.groupby('summarizer').agg({
            'rouge1_f1': 'mean',
            'rouge2_f1': 'mean',
            'rougeL_f1': 'mean',
            'rouge_avg': 'mean',
            'medical_preservation': 'mean',
            'compression_ratio': 'mean',
            'processing_time': 'mean'
        }).round(4)
        
        print(overall)
        
        # Detailed table by ASR model
        print("\n\n📋 RESULTS BY ASR MODEL:")
        print("-" * 100)
        
        # Create pivot tables
        rouge_pivot = results_df.pivot_table(
            values='rouge_avg',
            index='asr_model',
            columns='summarizer',
            aggfunc='mean'
        ).round(4)
        
        med_pivot = results_df.pivot_table(
            values='medical_preservation',
            index='asr_model',
            columns='summarizer',
            aggfunc='mean'
        ).round(4)
        
        print("\nROUGE-Avg Scores:")
        print(rouge_pivot)
        print("\nMedical Preservation:")
        print(med_pivot)
        
        # Save pivot tables
        rouge_pivot.to_csv(OUTPUT_DIR / "rouge_pivot.csv")
        med_pivot.to_csv(OUTPUT_DIR / "medical_pivot.csv")
        
        # Best combination
        best_rouge = results_df.groupby(['asr_model', 'summarizer'])['rouge_avg'].mean().idxmax()
        best_rouge_val = results_df.groupby(['asr_model', 'summarizer'])['rouge_avg'].mean().max()
        best_med = results_df.groupby(['asr_model', 'summarizer'])['medical_preservation'].mean().idxmax()
        best_med_val = results_df.groupby(['asr_model', 'summarizer'])['medical_preservation'].mean().max()
        
        print(f"\n🥇 Best ROUGE: {best_rouge[0]} + {best_rouge[1]} = {best_rouge_val:.4f}")
        print(f"🩺 Best Medical: {best_med[0]} + {best_med[1]} = {best_med_val:.2%}")
        
        # Sample summaries
        print("\n" + "=" * 80)
        print("📝 SAMPLE SUMMARIES")
        print("=" * 80)
        
        for i, row in enumerate(detailed_df.head(10).iterrows()):
            idx, data = row
            print(f"\n{i+1}. {data['asr_model']} - {data['summarizer']}")
            print(f"   ASR:    {data['asr_text_preview'][:100]}...")
            print(f"   Ref:    {data['reference_preview'][:100]}...")
            print(f"   Sum:    {data['summary_preview'][:100]}...")
            print(f"   ROUGE: {data['rouge_avg']:.3f} | Medical: {data['medical_preservation']:.2%}")
        
        # Summary statistics JSON
        summary = {
            'total_samples': len(results_df),
            'asr_models': list(results_df['asr_model'].unique()),
            'summarizers': list(results_df['summarizer'].unique()),
            'best_rouge': {
                'asr_model': best_rouge[0],
                'summarizer': best_rouge[1],
                'score': float(best_rouge_val)
            },
            'best_medical': {
                'asr_model': best_med[0],
                'summarizer': best_med[1],
                'score': float(best_med_val)
            },
            'bert_extractive_rouge': float(results_df[results_df['summarizer']=='bert_extractive']['rouge_avg'].mean()),
            'bert_extractive_medical': float(results_df[results_df['summarizer']=='bert_extractive']['medical_preservation'].mean()),
            'mt5_small_rouge': float(results_df[results_df['summarizer']=='mt5_small']['rouge_avg'].mean()),
            'mt5_small_medical': float(results_df[results_df['summarizer']=='mt5_small']['medical_preservation'].mean())
        }
        
        with open(OUTPUT_DIR / "summary.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✅ All results saved to: {OUTPUT_DIR}/")
        print(f"   - all_summaries_final.csv (complete results)")
        print(f"   - example_summaries_final.csv (sample summaries)")
        print(f"   - rouge_pivot.csv (ROUGE by ASR)")
        print(f"   - medical_pivot.csv (medical by ASR)")
        print(f"   - summary.json (key statistics)")
        
    else:
        print("\n❌ No results generated!")

if __name__ == "__main__":
    main()