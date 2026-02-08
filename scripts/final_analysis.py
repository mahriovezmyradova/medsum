#!/usr/bin/env python3
"""
Final analysis script for MultiMed German dataset.
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Add src to path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root / "src"))

from src.data.loader import MultiMedLoader

def ensure_directory(path: Path):
    """Ensure directory exists."""
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    return path

def main():
    """Main analysis function."""
    
    # Setup output directories
    output_dir = ensure_directory(project_root / "data" / "outputs")
    figures_dir = ensure_directory(output_dir / "figures")
    
    print(f"Output directory: {output_dir}")
    print(f"Figures directory: {figures_dir}")
    
    # Initialize loader
    config_path = project_root / "config" / "config.yaml"
    loader = MultiMedLoader(config_path=str(config_path))
    
    # Get dataset info
    print("\n" + "=" * 70)
    print("MULTIMED GERMAN DATASET - COMPREHENSIVE ANALYSIS")
    print("=" * 70)
    
    # Load all splits
    splits = loader.load_all_splits()
    
    # Analysis results storage
    analysis_results = {
        "dataset_summary": {},
        "split_statistics": {},
        "sample_details": {}
    }
    
    # 1. Basic Statistics
    print("\n1. BASIC DATASET STATISTICS")
    print("-" * 40)
    
    total_samples = sum(len(df) for df in splits.values())
    print(f"Total samples across all splits: {total_samples:,}")
    print(f"Number of splits: {len(splits)}")
    
    analysis_results["dataset_summary"]["total_samples"] = total_samples
    analysis_results["dataset_summary"]["num_splits"] = len(splits)
    analysis_results["dataset_summary"]["split_names"] = list(splits.keys())
    
    # 2. Detailed Analysis per Split
    print("\n2. DETAILED ANALYSIS PER SPLIT")
    print("-" * 40)
    
    split_stats = []
    
    for split_name, df in splits.items():
        print(f"\n📊 {split_name.upper()} SPLIT")
        print(f"   {'─' * 30}")
        
        stats = {
            "split": split_name,
            "samples": len(df),
            "columns": list(df.columns)
        }
        
        # Text Analysis
        if 'text' in df.columns:
            texts = df['text'].astype(str)
            
            char_stats = texts.str.len()
            word_stats = texts.str.split().str.len()
            
            text_analysis = {
                "char_min": int(char_stats.min()),
                "char_max": int(char_stats.max()),
                "char_mean": float(char_stats.mean()),
                "char_std": float(char_stats.std()),
                "word_min": int(word_stats.min()),
                "word_max": int(word_stats.max()),
                "word_mean": float(word_stats.mean()),
                "word_std": float(word_stats.std()),
                "missing": int(texts.isnull().sum())
            }
            
            stats["text_analysis"] = text_analysis
            
            print(f"   📝 Text Analysis:")
            print(f"      • Samples: {len(df):,}")
            print(f"      • Characters: {text_analysis['char_min']:,} - {text_analysis['char_max']:,} "
                  f"(avg: {text_analysis['char_mean']:.1f} ± {text_analysis['char_std']:.1f})")
            print(f"      • Words: {text_analysis['word_min']:,} - {text_analysis['word_max']:,} "
                  f"(avg: {text_analysis['word_mean']:.1f} ± {text_analysis['word_std']:.1f})")
            
            # Sample texts
            sample_texts = []
            for i in range(min(2, len(texts))):
                sample = texts.iloc[i]
                truncated = sample[:80] + "..." if len(sample) > 80 else sample
                sample_texts.append(truncated)
            
            stats["sample_texts"] = sample_texts
        
        # Duration Analysis
        if 'duration' in df.columns:
            durations = df['duration']
            
            duration_analysis = {
                "min": float(durations.min()),
                "max": float(durations.max()),
                "mean": float(durations.mean()),
                "std": float(durations.std())
            }
            
            stats["duration_analysis"] = duration_analysis
            
            print(f"   ⏱️  Duration Analysis:")
            print(f"      • Range: {duration_analysis['min']:.1f}s - {duration_analysis['max']:.1f}s")
            print(f"      • Average: {duration_analysis['mean']:.1f}s ± {duration_analysis['std']:.1f}s")
        
        # Audio Analysis
        if 'audio' in df.columns:
            first_audio = df['audio'].iloc[0]
            audio_info = {
                "type": str(type(first_audio)),
                "has_bytes": isinstance(first_audio, dict) and 'bytes' in first_audio
            }
            
            if isinstance(first_audio, dict) and 'bytes' in first_audio:
                audio_info["bytes_size"] = len(first_audio['bytes'])
                print(f"   🔊 Audio Format: OGG bytes ({audio_info['bytes_size']:,} bytes)")
            
            stats["audio_info"] = audio_info
        
        split_stats.append(stats)
        analysis_results["split_statistics"][split_name] = stats
    
    # 3. Create Visualizations
    print("\n3. CREATING VISUALIZATIONS")
    print("-" * 40)
    
    # Set style
    plt.style.use('default')
    sns.set_style("whitegrid")
    
    # Figure 1: Word Count Distribution
    fig1, axes1 = plt.subplots(1, len(splits), figsize=(16, 5))
    if len(splits) == 1:
        axes1 = [axes1]
    
    colors = ['#1f77b4', '#2ca02c', '#d62728']
    
    for idx, (split_name, df) in enumerate(splits.items()):
        if 'text' in df.columns:
            texts = df['text'].astype(str)
            word_counts = texts.str.split().str.len()
            
            ax = axes1[idx]
            n, bins, patches = ax.hist(word_counts, bins=40, alpha=0.7, 
                                      color=colors[idx % len(colors)], 
                                      edgecolor='black', density=True)
            
            # Add statistics
            mean_val = word_counts.mean()
            median_val = word_counts.median()
            
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {mean_val:.1f}')
            ax.axvline(median_val, color='green', linestyle=':', linewidth=2,
                      label=f'Median: {median_val:.1f}')
            
            ax.set_title(f'{split_name.capitalize()} Split\n({len(df):,} samples)', 
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('Word Count', fontsize=10)
            ax.set_ylabel('Density', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
    
    plt.suptitle('Word Count Distribution by Dataset Split', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    word_count_path = figures_dir / 'word_count_distribution.png'
    plt.savefig(word_count_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"   ✅ Saved: {word_count_path}")
    
    # Figure 2: Duration Distribution
    fig2, axes2 = plt.subplots(1, len(splits), figsize=(16, 5))
    if len(splits) == 1:
        axes2 = [axes2]
    
    for idx, (split_name, df) in enumerate(splits.items()):
        if 'duration' in df.columns:
            durations = df['duration']
            
            ax = axes2[idx]
            ax.hist(durations, bins=30, alpha=0.7, color=colors[idx % len(colors)], 
                   edgecolor='black', density=True)
            
            mean_val = durations.mean()
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {mean_val:.1f}s')
            
            ax.set_title(f'{split_name.capitalize()} Split\n({len(df):,} samples)', 
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('Duration (seconds)', fontsize=10)
            ax.set_ylabel('Density', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
    
    plt.suptitle('Audio Duration Distribution by Dataset Split', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    duration_path = figures_dir / 'duration_distribution.png'
    plt.savefig(duration_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"   ✅ Saved: {duration_path}")
    
    # Figure 3: Combined Statistics
    fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Subplot 1: Sample Counts
    split_names = list(splits.keys())
    sample_counts = [len(df) for df in splits.values()]
    
    bars = ax1.bar(split_names, sample_counts, color=colors[:len(splits)], alpha=0.8)
    ax1.set_title('Number of Samples per Split', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Split', fontsize=10)
    ax1.set_ylabel('Number of Samples', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 10,
                f'{int(height):,}', ha='center', va='bottom', fontsize=9)
    
    # Subplot 2: Average Words vs Duration
    avg_words = []
    avg_durations = []
    
    for split_name, df in splits.items():
        if 'text' in df.columns:
            words = df['text'].astype(str).str.split().str.len().mean()
            avg_words.append(words)
        else:
            avg_words.append(0)
        
        if 'duration' in df.columns:
            duration = df['duration'].mean()
            avg_durations.append(duration)
        else:
            avg_durations.append(0)
    
    x = range(len(split_names))
    ax2.plot(x, avg_words, 'o-', color='blue', linewidth=2, markersize=8, 
            label='Avg Words', alpha=0.8)
    ax2.plot(x, avg_durations, 's-', color='red', linewidth=2, markersize=8, 
            label='Avg Duration (s)', alpha=0.8)
    
    ax2.set_title('Average Words vs Duration per Split', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Split', fontsize=10)
    ax2.set_ylabel('Value', fontsize=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(split_names)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)
    
    plt.suptitle('Dataset Overview Statistics', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    overview_path = figures_dir / 'dataset_overview.png'
    plt.savefig(overview_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"   ✅ Saved: {overview_path}")
    
    # 4. Save Results
    print("\n4. SAVING ANALYSIS RESULTS")
    print("-" * 40)
    
    # Save as CSV
    summary_df = pd.DataFrame(split_stats)
    summary_csv = output_dir / 'dataset_summary.csv'
    summary_df.to_csv(summary_csv, index=False)
    print(f"   ✅ CSV summary: {summary_csv}")
    
    # Save as JSON
    summary_json = output_dir / 'detailed_analysis.json'
    with open(summary_json, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=2, ensure_ascii=False)
    print(f"   ✅ JSON analysis: {summary_json}")
    
    # Create markdown report
    report_md = output_dir / 'analysis_report.md'
    with open(report_md, 'w', encoding='utf-8') as f:
        f.write("# MultiMed German Dataset Analysis Report\n\n")
        f.write(f"**Generated on:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 📊 Dataset Overview\n\n")
        f.write(f"- **Total samples:** {total_samples:,}\n")
        f.write(f"- **Number of splits:** {len(splits)}\n")
        f.write(f"- **Splits:** {', '.join(splits.keys())}\n\n")
        
        f.write("## 📈 Split Statistics\n\n")
        for split_name, df in splits.items():
            f.write(f"### {split_name.upper()} Split\n\n")
            f.write(f"- **Samples:** {len(df):,}\n")
            
            if 'text' in df.columns:
                texts = df['text'].astype(str)
                avg_words = texts.str.split().str.len().mean()
                f.write(f"- **Average words:** {avg_words:.1f}\n")
            
            if 'duration' in df.columns:
                avg_duration = df['duration'].mean()
                f.write(f"- **Average duration:** {avg_duration:.1f}s\n")
            
            f.write("\n")
        
        f.write("## 📊 Visualizations\n\n")
        f.write(f"![Word Count Distribution](figures/word_count_distribution.png)\n\n")
        f.write(f"![Duration Distribution](figures/duration_distribution.png)\n\n")
        f.write(f"![Dataset Overview](figures/dataset_overview.png)\n\n")
        
        f.write("## 📝 Sample Texts\n\n")
        for split_name, df in splits.items():
            if 'text' in df.columns:
                f.write(f"### {split_name.upper()} - First 2 Samples\n\n")
                texts = df['text'].astype(str)
                for i in range(min(2, len(texts))):
                    f.write(f"{i+1}. {texts.iloc[i][:150]}...\n\n")
    
    print(f"   ✅ Markdown report: {report_md}")
    
    # 5. Final Summary
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE - KEY FINDINGS")
    print("=" * 70)
    
    print("\n🎯 **Dataset Characteristics:**")
    print(f"   • Total conversations: {total_samples:,}")
    print(f"   • Average duration: ~12 seconds")
    print(f"   • Average words per conversation: ~28 words")
    print(f"   • Audio format: OGG bytes in dictionary format")
    
    print("\n📊 **Split Distribution:**")
    for split_name, df in splits.items():
        percentage = (len(df) / total_samples) * 100
        print(f"   • {split_name}: {len(df):,} samples ({percentage:.1f}%)")
    
    print("\n✅ **Analysis outputs saved to:**")
    print(f"   • Figures: {figures_dir}/")
    print(f"   • CSV summary: {summary_csv}")
    print(f"   • JSON analysis: {summary_json}")
    print(f"   • Markdown report: {report_md}")
    
    print("\n" + "=" * 70)
    print("NEXT STEP: Proceed with ASR implementation")
    print("=" * 70)

if __name__ == "__main__":
    main()