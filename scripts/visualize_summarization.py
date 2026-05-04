# analyze_summarization_results.py

import pandas as pd
import json
from pathlib import Path

def analyze_results(results_dir='summarization_results'):
    """Load and analyze summarization results"""
    
    results_path = Path(results_dir)
    
    # Load results
    detailed = pd.read_csv(results_path / 'detailed_results.csv')
    summary_stats = pd.read_csv(results_path / 'summarizer_performance.csv', index_col=0)
    asr_performance = pd.read_csv(results_path / 'asr_model_performance.csv', index_col=0)
    
    with open(results_path / 'summary.json', 'r') as f:
        summary_json = json.load(f)
    
    print("\n" + "="*60)
    print("SUMMARIZATION COMPARISON SUMMARY")
    print("="*60)
    
    print(f"\nTotal samples processed: {summary_json['total_samples']}")
    print(f"ASR models compared: {', '.join(summary_json['asr_models'])}")
    print(f"Summarizers compared: {', '.join(summary_json['summarizers'])}")
    
    print(f"\n🏆 Best Overall Summarizer: {summary_json['best_summarizer']}")
    print(f"   Average F1 Score: {summary_json['best_f1_score']:.3f}")
    
    print("\n" + "-"*40)
    print("Summarizer Performance (F1 Score):")
    print("-"*40)
    print(summary_stats[('f1_score', 'mean')].sort_values(ascending=False))
    
    print("\n" + "-"*40)
    print("Performance by ASR Model:")
    print("-"*40)
    print(asr_performance)
    
    # Find best combination
    best_combination = detailed.groupby(['model_name', 'summarizer'])['f1_score'].mean().idxmax()
    best_score = detailed.groupby(['model_name', 'summarizer'])['f1_score'].mean().max()
    print(f"\n✨ Best combination: {best_combination[0]} + {best_combination[1]}")
    print(f"   F1 Score: {best_score:.3f}")
    
    # Impact of WER on summarization
    print("\n" + "-"*40)
    print("Impact of ASR Errors on Summarization:")
    print("-"*40)
    
    wer_correlation = detailed.groupby('summarizer')[['f1_score', 'wer']].corr().iloc[0::2, 1]
    for summarizer in wer_correlation.index.get_level_values(0).unique():
        corr = wer_correlation[summarizer]['wer']
        print(f"{summarizer}: correlation = {corr:.3f}")
    
    return detailed, summary_stats, asr_performance

if __name__ == "__main__":
    detailed, stats, asr_perf = analyze_results()