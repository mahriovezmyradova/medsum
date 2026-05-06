#!/usr/bin/env python3
"""
GERMAN SUMMARIZATION - BART ONLY (WORKING VERSION)
Model: Shahm/bart-german
"""

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pathlib import Path
import warnings
import time
from tqdm import tqdm
from rouge_score import rouge_scorer

warnings.filterwarnings("ignore")

# ================================
# CONFIGURATION
# ================================
INPUT_PATH = Path("/Users/mahriovezmyradova/MedicalASR-Summarization/data/outputs/full_dataset_analysis/all_transcriptions_copy.csv")
OUTPUT_DIR = Path("data/outputs/summarization_bart_german")

ASR_MODELS = [
    "whisper_tiny",
    "whisper_base",
    "whisper_small",
    "wav2vec2_facebook_original",
    "wav2vec2_jonatasgrosman"
]

SAMPLES_PER_MODEL = 50

MEDICAL_TERMS = [
    "diagnose", "krankheit", "erkrankung", "schmerz",
    "fieber", "husten", "therapie", "behandlung",
    "medikament", "arzt", "patient", "operation"
]

# ================================
# GERMAN BART SUMMARIZER
# ================================

class GermanBARTSummarizer:
    def __init__(self):
        self.model_name = "Shahm/bart-german"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print("\n🔄 Loading German BART...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

        self.model.to(self.device)
        self.model.eval()

        print(f"✓ Loaded on {self.device}")

    def summarize(self, text, max_length=120, min_length=30):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=1024,
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                num_beams=6,
                max_length=max_length,
                min_length=min_length,
                length_penalty=2.0,
                no_repeat_ngram_size=3,
                early_stopping=True
            )

        summary = self.tokenizer.decode(
            summary_ids[0],
            skip_special_tokens=True
        )

        return summary.strip()

# ================================
# METRICS
# ================================

class Metrics:
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"],
            use_stemmer=True
        )

    def calculate(self, reference, hypothesis):
        if not hypothesis or len(hypothesis) < 10:
            return 0, 0

        scores = self.scorer.score(reference, hypothesis)
        rouge_avg = np.mean([
            scores["rouge1"].fmeasure,
            scores["rouge2"].fmeasure,
            scores["rougeL"].fmeasure
        ])

        ref_lower = reference.lower()
        hyp_lower = hypothesis.lower()

        ref_terms = [t for t in MEDICAL_TERMS if t in ref_lower]
        if len(ref_terms) > 0:
            preserved = sum(1 for t in ref_terms if t in hyp_lower)
            medical_pres = preserved / len(ref_terms)
        else:
            medical_pres = 1.0

        return rouge_avg, medical_pres

# ================================
# MAIN
# ================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n📂 Loading data...")
    df = pd.read_csv(INPUT_PATH, sep=";", encoding="utf-8")

    summarizer = GermanBARTSummarizer()
    metrics = Metrics()

    results = []

    for asr_model in ASR_MODELS:
        if asr_model not in df["model_name"].unique():
            continue

        print(f"\n📊 {asr_model}")

        model_df = df[df["model_name"] == asr_model]
        test_df = model_df[model_df["split"] == "test"]

        if len(test_df) == 0:
            test_df = model_df

        n_samples = min(SAMPLES_PER_MODEL, len(test_df))
        sampled = test_df.sample(n=n_samples, random_state=42)

        for _, row in tqdm(sampled.iterrows(), total=len(sampled)):
            asr_text = str(row["asr_text"])
            reference = str(row["reference_text"])

            if len(asr_text.strip()) < 10:
                continue

            start = time.time()
            summary = summarizer.summarize(asr_text)
            proc_time = time.time() - start

            rouge_avg, medical_pres = metrics.calculate(reference, summary)

            results.append({
                "asr_model": asr_model,
                "rouge_avg": rouge_avg,
                "medical_preservation": medical_pres,
                "processing_time": proc_time
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "bart_results.csv", index=False)

    print("\n🏆 Overall Performance:")
    print(results_df.groupby("asr_model").mean().round(4))

    print(f"\n✅ Saved to {OUTPUT_DIR}/bart_results.csv")


if __name__ == "__main__":
    main()