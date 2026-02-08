#!/usr/bin/env python3
"""
Script to analyze the MultiMed German dataset.
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

from utils.data_loader import MultiMedLoader

def setup_output_directory():
    """Setup output directory."""
    output_dir = project_root / "data" / "outputs"
    
    # Ensure it's a directory
    if output_dir.exists():
        if output_dir.is_file():
            output_dir.unlink()
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    return output_dir

def main():
    """Main analysis function."""
    
    # Setup output
    output_dir = setup_output_directory()
    print(f"Output directory: {output_dir}")
    
    # Initialize loader
    config_path = project_root / "config" / "config.yaml"
    loader = MultiMedLoader(config_path=str(config_path))
    
    # Get dataset info
    print("=" * 60)
    print("MULTIMED GERMAN DATASET ANALYSIS")
    print("=" * 60)
    
    info_df = loader.get_dataset_info()
    print("\nDataset Overview:")
    print(info_df.to_string())
    
    # Load all splits
    splits = loader.load_all_splits()
    
    print("\n" + "=" * 60)
    print("DETAILED ANALYSIS PER SPLIT")
    print("=" * 60)
    
    analysis_results = {}
    
    for split_name, df in splits.items():
        print(f"\n{split_name.upper()} SPLIT:")
        print(f"-" * 40)
        
        # Basic info
        print(f"Total samples: {len(df):,}")
        print(f"Columns: {list(df.columns)}")
        
        # Store for later
        analysis_results[split_name] = {
            'total_samples': len(df),
            'columns': list(df.columns)
        }
        
        # Text analysis
        if 'text' in df.columns:
            texts = df['text'].astype(str)
            
            print(f"\nText Analysis:")
            print(f"  Missing values: {texts.isnull().sum():,}")
            
            # Length statistics
            char_lengths = texts.str.len()
            word_counts = texts.str.split().str.len()
            
            print(f"  Character length:")
            print(f"    Min: {char_lengths.min():,}")
            print(f"    Max: {char_lengths.max():,}")
            print(f"    Mean: {char_lengths.mean():.1f}")
            print(f"    Std: {char_lengths.std():.1f}")
            
            print(f"  Word count:")
            print(f"    Min: {word_counts.min():,}")
            print(f"    Max: {word_counts.max():,}")
            print(f"    Mean: {word_counts.mean():.1f}")
            print(f"    Std: {word_counts.std():.1f}")
            
            # Store statistics
            analysis_results[split_name]['text_stats'] = {
                'char_min': int(char_lengths.min()),
                'char_max': int(char_lengths.max()),
                'char_mean': float(char_lengths.mean()),
                'char_std': float(char_lengths.std()),
                'word_min': int(word_counts.min()),
                'word_max': int(word_counts.max()),
                'word_mean': float(word_counts.mean()),
                'word_std': float(word_counts.std()),
                'missing': int(texts.isnull().sum())
            }
            
            # Sample texts
            print(f"\n  Sample texts:")
            sample_texts = []
            for i in range(min(3, len(texts))):
                sample = texts.iloc[i]
                truncated = sample[:100] + "..." if len(sample) > 100 else sample
                print(f"    {i+1}. {truncated}")
                sample_texts.append(truncated)
            
            analysis_results[split_name]['sample_texts'] = sample_texts
        
        # Audio analysis
        if 'audio' in df.columns:
            print(f"\nAudio Analysis:")
            print(f"  Column type: {df['audio'].dtype}")
            
            # Check first audio sample
            first_audio = df['audio'].iloc[0]
            print(f"  First sample type: {type(first_audio)}")
            
            if isinstance(first_audio, dict):
                keys = list(first_audio.keys())
                print(f"  First sample keys: {keys}")
                
                if 'bytes' in first_audio:
                    byte_size = len(first_audio['bytes'])
                    print(f"  Audio bytes size: {byte_size:,} bytes")
                    
                    analysis_results[split_name]['audio_info'] = {
                        'type': 'dict_with_bytes',
                        'keys': keys,
                        'bytes_size': byte_size
                    }
        
        # Duration analysis
        if 'duration' in df.columns:
            durations = df['duration']
            print(f"\nDuration Analysis:")
            print(f"  Min: {durations.min():.1f}s")
            print(f"  Max: {durations.max():.1f}s")
            print(f"  Mean: {durations.mean():.1f}s")
            print(f"  Std: {durations.std():.1f}s")
            
            analysis_results[split_name]['duration_stats'] = {
                'min': float(durations.min()),
                'max': float(durations.max()),
                'mean': float(durations.mean()),
                'std': float(durations.std())
            }
    
    # Create visualizations
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)
    
    # Text length distribution
    fig, axes = plt.subplots(1, len(splits), figsize=(15, 5))
    
    if len(splits) == 1:
        axes = [axes]
    
    colors = ['skyblue', 'lightgreen', 'lightcoral']
    
    for idx, (split_name, df) in enumerate(splits.items()):
        if 'text' in df.columns:
            texts = df['text'].astype(str)
            word_counts = texts.str.split().str.len()
            
            ax = axes[idx]
            ax.hist(word_counts, bins=50, alpha=0.7, color=colors[idx % len(colors)], 
                   edgecolor='black', density=True)
            ax.set_title(f'{split_name.capitalize()} Split', fontsize=14, fontweight='bold')
            ax.set_xlabel('Word Count', fontsize=12)
            ax.set_ylabel('Density', fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            mean_val = word_counts.mean()
            median_val = word_counts.median()
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {mean_val:.1f}')
            ax.axvline(median_val, color='green', linestyle=':', linewidth=2,
                      label=f'Median: {median_val:.1f}')
            ax.legend()
    
    plt.suptitle('Word Count Distribution by Split', fontsize=16, fontweight='bold', y=1.05)
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / 'word_count_distribution.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to: {output_path}")
    plt.show()
    
    # Create duration distribution plot
    fig, axes = plt.subplots(1, len(splits), figsize=(15, 5))
    
    if len(splits) == 1:
        axes = [axes]
    
    for idx, (split_name, df) in enumerate(splits.items()):
        if 'duration' in df.columns:
            durations = df['duration']
            
            ax = axes[idx]
            ax.hist(durations, bins=30, alpha=0.7, color=colors[idx % len(colors)], 
                   edgecolor='black', density=True)
            ax.set_title(f'{split_name.capitalize()} Split', fontsize=14, fontweight='bold')
            ax.set_xlabel('Duration (seconds)', fontsize=12)
            ax.set_ylabel('Density', fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            mean_val = durations.mean()
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {mean_val:.1f}s')
            ax.legend()
    
    plt.suptitle('Audio Duration Distribution by Split', fontsize=16, fontweight='bold', y=1.05)
    plt.tight_layout()
    
    duration_path = output_dir / 'duration_distribution.png'
    plt.savefig(duration_path, dpi=150, bbox_inches='tight')
    print(f"Saved duration visualization to: {duration_path}")
    plt.show()
    
    # Save summary statistics
    summary_data = []
    
    for split_name, df in splits.items():
        summary = {
            'split': split_name,
            'samples': len(df),
            'columns': ', '.join(df.columns)
        }
        
        if 'text' in df.columns:
            texts = df['text'].astype(str)
            summary.update({
                'text_min_length': texts.str.len().min(),
                'text_max_length': texts.str.len().max(),
                'text_mean_length': texts.str.len().mean(),
                'text_std_length': texts.str.len().std(),
                'word_min_count': texts.str.split().str.len().min(),
                'word_max_count': texts.str.split().str.len().max(),
                'word_mean_count': texts.str.split().str.len().mean(),
                'word_std_count': texts.str.split().str.len().std(),
                'missing_texts': texts.isnull().sum()
            })
        
        if 'duration' in df.columns:
            durations = df['duration']
            summary.update({
                'duration_min': durations.min(),
                'duration_max': durations.max(),
                'duration_mean': durations.mean(),
                'duration_std': durations.std()
            })
        
        summary_data.append(summary)
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = output_dir / 'dataset_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    
    print(f"\nSummary statistics saved to: {summary_path}")
    
    # Save detailed analysis results
    analysis_path = output_dir / 'detailed_analysis.json'
    with open(analysis_path, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=2, ensure_ascii=False)
    
    print(f"Detailed analysis saved to: {analysis_path}")
    
    # Print final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    
    total_samples = sum([len(df) for df in splits.values()])
    print(f"Total samples across all splits: {total_samples:,}")
    print(f"Number of splits: {len(splits)}")
    
    for split_name in splits.keys():
        print(f"\n{split_name.upper()}:")
        print(f"  Samples: {len(splits[split_name]):,}")
        if 'text' in splits[split_name].columns:
            avg_words = splits[split_name]['text'].astype(str).str.split().str.len().mean()
            print(f"  Average words per sample: {avg_words:.1f}")
        if 'duration' in splits[split_name].columns:
            avg_duration = splits[split_name]['duration'].mean()
            print(f"  Average duration: {avg_duration:.1f}s")
    
    print(f"\nAnalysis complete! All outputs saved to: {output_dir}")

if __name__ == "__main__":
    main()