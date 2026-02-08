#!/usr/bin/env python3
"""
Analyze ASR comparison results and generate thesis-ready tables.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
import sys

# Add parent directory to path for imports
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

def create_thesis_tables():
    """Create tables suitable for thesis."""
    
    # Setup output directory
    output_dir = Path("data/outputs/asr_comparison")
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Try to load results from different possible files
    results_files = [
        output_dir / "quick_comparison_results.csv",  # From quick_comparison_fixed.py
        output_dir / "comparison_results.csv",        # From compare_asr_models.py
        output_dir / "results.csv"                    # Generic
    ]
    
    results_file = None
    for file_path in results_files:
        if file_path.exists():
            results_file = file_path
            print(f"✓ Found results file: {results_file}")
            break
    
    if not results_file:
        print(f"❌ No results file found in {output_dir}")
        print("Please run comparison script first!")
        print("Try: python scripts/quick_comparison_fixed.py")
        return
    
    results = pd.read_csv(results_file)
    
    print("=" * 80)
    print("THESIS-READY RESULTS ANALYSIS")
    print("=" * 80)
    
    print(f"\n📊 Results loaded:")
    print(results)
    
    # Table 1: Performance Comparison
    print("\n📊 Table 1: ASR Model Performance Comparison")
    print("-" * 80)
    
    # Determine column names (they might vary)
    wer_col = None
    time_col = None
    samples_col = None
    
    for col in results.columns:
        col_lower = col.lower()
        if 'wer' in col_lower:
            wer_col = col
        elif 'time' in col_lower or 'avg_time' in col_lower:
            time_col = col
        elif 'sample' in col_lower:
            samples_col = col
    
    # If column names not found, use defaults
    if wer_col is None:
        wer_col = 'avg_wer' if 'avg_wer' in results.columns else results.columns[1]
    if time_col is None:
        time_col = 'avg_time' if 'avg_time' in results.columns else results.columns[2]
    if samples_col is None:
        samples_col = 'samples' if 'samples' in results.columns else results.columns[3]
    
    model_col = results.columns[0] if len(results.columns) > 0 else 'model'
    
    print(f"{'Model':<25} {'WER':<12} {'Time (s)':<12} {'Samples':<10}")
    print("-" * 80)
    
    for _, row in results.iterrows():
        model = row[model_col]
        wer = row[wer_col]
        time_val = row[time_col]
        samples = row[samples_col] if samples_col in row else len(results)
        
        # Check if we have std_wer
        std_col = None
        for col in results.columns:
            if 'std' in col.lower() and 'wer' in col.lower():
                std_col = col
                break
        
        if std_col and std_col in row:
            wer_str = f"{wer:.3f}±{row[std_col]:.3f}"
        else:
            wer_str = f"{wer:.3f}"
        
        print(f"{str(model):<25} {wer_str:<12} {time_val:<12.2f} {samples:<10}")
    
    # Create visualizations
    plt.style.use('default')
    sns.set_style("whitegrid")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # WER comparison (bar chart)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    x = range(len(results))
    models = results[model_col].tolist()
    wers = results[wer_col].tolist()
    
    bars1 = ax1.bar(x, wers, alpha=0.8,
                   color=colors[:len(results)], edgecolor='black')
    ax1.set_title('Word Error Rate (WER) Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('WER (lower is better)', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, wer_value in zip(bars1, wers):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{wer_value:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Time comparison
    times = results[time_col].tolist()
    bars2 = ax2.bar(x, times, alpha=0.8,
                   color=colors[:len(results)], edgecolor='black')
    ax2.set_title('Processing Time Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Time per sample (seconds)', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, time_value in zip(bars2, times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{time_value:.2f}s', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'performance_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {figures_dir / 'performance_comparison.png'}")
    plt.show()
    
    # Create detailed analysis
    print("\n🔍 Key Findings for Thesis:")
    print("-" * 80)
    
    best_idx = results[wer_col].idxmin()
    best_model = results.iloc[best_idx]
    
    fastest_idx = results[time_col].idxmin()
    fastest_model = results.iloc[fastest_idx]
    
    print(f"1. Best performing model (lowest WER):")
    print(f"   → {best_model[model_col]} (WER: {best_model[wer_col]:.3f})")
    
    print(f"\n2. Fastest model:")
    print(f"   → {fastest_model[model_col]} ({fastest_model[time_col]:.2f}s per sample)")
    
    # Calculate improvement
    if len(results) > 1:
        worst_wer = results[wer_col].max()
        best_wer = results[wer_col].min()
        if worst_wer > 0:
            improvement = ((worst_wer - best_wer) / worst_wer) * 100
            print(f"\n3. WER improvement from worst to best:")
            print(f"   → {improvement:.1f}% reduction in errors")
    
    print(f"\n4. Model characteristics:")
    for _, row in results.iterrows():
        print(f"   • {row[model_col]}: WER={row[wer_col]:.3f}, Time={row[time_col]:.2f}s")
    
    print("\n5. Practical implications for medical applications:")
    print("   • Lower WER → Better transcription accuracy for medical records")
    print("   • Faster processing → More scalable for clinical use")
    print("   • Need to balance accuracy vs speed based on specific use case")
    
    # Save thesis-ready table
    thesis_table = results.copy()
    thesis_csv = output_dir / "thesis_results_table.csv"
    thesis_table.to_csv(thesis_csv, index=False)
    
    print(f"\n✓ Thesis-ready table saved to: {thesis_csv}")
    
    # Create LaTeX table for thesis
    print("\n📝 LaTeX Table for Thesis:")
    print("-" * 80)
    print("\\begin{table}[htbp]")
    print("\\centering")
    print("\\caption{ASR Model Performance Comparison on German Medical Conversations}")
    print("\\label{tab:asr_comparison}")
    print("\\begin{tabular}{lccc}")
    print("\\hline")
    print("Model & WER & Time (s) & Samples \\\\")
    print("\\hline")
    
    for _, row in results.iterrows():
        print(f"{row[model_col]} & {row[wer_col]:.3f} & {row[time_col]:.2f} & {row[samples_col] if samples_col in row else 'N/A'} \\\\")
    
    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")
    
    # Create markdown report
    print("\n📄 Generating comprehensive report...")
    
    report_file = output_dir / "thesis_analysis_report.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# ASR Model Comparison - Thesis Analysis Report\n\n")
        f.write(f"**Generated on:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Executive Summary\n\n")
        f.write(f"- **Best model:** {best_model[model_col]} (WER: {best_model[wer_col]:.3f})\n")
        f.write(f"- **Fastest model:** {fastest_model[model_col]} ({fastest_model[time_col]:.2f}s per sample)\n")
        f.write(f"- **Models evaluated:** {len(results)}\n")
        f.write(f"- **Sample size:** {results[samples_col].sum() if samples_col in results else 'N/A'} total samples\n\n")
        
        f.write("## Results\n\n")
        f.write("| Model | WER | Time per sample (s) | Samples |\n")
        f.write("|-------|-----|---------------------|---------|\n")
        
        for _, row in results.iterrows():
            samples_val = row[samples_col] if samples_col in row else 'N/A'
            f.write(f"| {row[model_col]} | {row[wer_col]:.3f} | {row[time_col]:.2f} | {samples_val} |\n")
        
        f.write("\n## Visualizations\n\n")
        f.write(f"![Performance Comparison](figures/performance_comparison.png)\n\n")
        
        f.write("## Discussion\n\n")
        f.write("### Implications for Medical ASR\n")
        f.write("1. **Accuracy is critical** for medical transcriptions to avoid misdiagnosis\n")
        f.write("2. **Speed matters** for clinical workflow efficiency\n")
        f.write("3. **German medical terminology** poses specific challenges\n\n")
        
        f.write("### Recommendations\n")
        f.write(f"1. **For highest accuracy:** Use {best_model[model_col]}\n")
        f.write(f"2. **For fastest processing:** Use {fastest_model[model_col]}\n")
        f.write("3. **For medical applications:** Prioritize accuracy over speed\n\n")
        
        f.write("## Next Steps for Research\n")
        f.write("1. Increase sample size for more statistical power\n")
        f.write("2. Test on the full test split (1,091 samples)\n")
        f.write("3. Analyze error patterns specific to medical terminology\n")
        f.write("4. Evaluate impact on downstream summarization tasks\n")
    
    print(f"✓ Comprehensive report saved to: {report_file}")
    
    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nYou now have everything you need for your thesis:")
    print("1. CSV tables with results ✓")
    print("2. Visualizations ✓")
    print("3. LaTeX table code ✓")
    print("4. Comprehensive analysis report ✓")
    print("\nTo scale up:")
    print("1. Edit quick_comparison_fixed.py and increase SAMPLE_COUNT")
    print("2. Run: python scripts/quick_comparison_fixed.py")
    print("3. Run: python scripts/analyze_comparison.py")

if __name__ == "__main__":
    create_thesis_tables()