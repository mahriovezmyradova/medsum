#!/usr/bin/env python3
"""
Main script for running full ASR evaluation pipeline.
"""
import sys
from pathlib import Path
import logging
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add src to path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root / "src"))

from config.settings import config
from src.data.loader import MultiMedLoader
from asr.factory import ASRFactory
from evaluation.metrics import ASREvaluator
from evaluation.statistical import StatisticalAnalyzer
from evaluation.visualizations import VisualizationEngine
from evaluation.error_analysis import ErrorAnalyzer
from utils.logger import setup_logging

def setup_experiment():
    """Setup experiment environment."""
    # Setup logging
    log_file = config.paths.outputs_dir / config.experiment.log_file
    setup_logging(
        level=config.experiment.log_level,
        log_file=log_file if config.experiment.log_file else None
    )
    
    logger = logging.getLogger(__name__)
    
    # Create output directories
    output_dir = config.paths.outputs_dir / "asr_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config.save(output_dir / "experiment_config.yaml")
    
    logger.info("=" * 70)
    logger.info(f"ASR EVALUATION EXPERIMENT: {config.experiment.name}")
    logger.info(f"Version: {config.experiment.version}")
    logger.info(f"Description: {config.experiment.description}")
    logger.info("=" * 70)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Log file: {log_file}")
    
    return output_dir

def load_and_prepare_data():
    """Load and prepare dataset for evaluation."""
    logger = logging.getLogger(__name__)
    
    logger.info("\n" + "=" * 70)
    logger.info("DATA LOADING AND PREPARATION")
    logger.info("=" * 70)
    
    # Initialize data loader
    loader = MultiMedLoader(config)
    
    # Load evaluation split
    split_name = config.dataset.test_split
    logger.info(f"Loading {split_name} split...")
    
    samples = loader.load_samples(
        split_name, 
        max_samples=config.experiment.max_samples
    )
    
    logger.info(f"Loaded {len(samples)} samples from {split_name}")
    
    # Extract reference texts and audio
    reference_texts = [sample.text for sample in samples]
    audio_data = [sample.audio for sample in samples]
    
    # Prepare audio arrays and sample rates
    audio_arrays = [audio.array for audio in audio_data]
    sample_rates = [audio.sample_rate for audio in audio_data]
    
    # Log dataset statistics
    durations = [audio.duration for audio in audio_data]
    text_lengths = [len(text) for text in reference_texts]
    
    logger.info(f"Audio statistics:")
    logger.info(f"  Total duration: {sum(durations):.1f}s")
    logger.info(f"  Average duration: {np.mean(durations):.2f}s")
    logger.info(f"  Duration range: {min(durations):.1f}s - {max(durations):.1f}s")
    
    logger.info(f"Text statistics:")
    logger.info(f"  Average length: {np.mean(text_lengths):.1f} characters")
    logger.info(f"  Total words: {sum(len(text.split()) for text in reference_texts):,}")
    
    return reference_texts, audio_arrays, sample_rates, samples

def initialize_asr_models():
    """Initialize all ASR models for evaluation."""
    logger = logging.getLogger(__name__)
    
    logger.info("\n" + "=" * 70)
    logger.info("ASR MODEL INITIALIZATION")
    logger.info("=" * 70)
    
    # Create models using factory
    all_model_configs = config.asr.all_models
    
    logger.info(f"Creating {len(all_model_configs)} ASR models...")
    
    models = ASRFactory.create_all_models(all_model_configs)
    
    # Log model information
    for model_name, model in models.items():
        try:
            model_info = model.get_model_info()
            logger.info(f"  ✓ {model_name}: {model_info.model_type}, "
                       f"Parameters: {model_info.parameters:,}, "
                       f"Size: {model_info.size_gb:.2f} GB")
        except:
            logger.info(f"  ✓ {model_name}")
    
    logger.info(f"Successfully initialized {len(models)} models")
    
    return models

def run_transcriptions(models, audio_arrays, sample_rates, output_dir):
    """Run transcriptions for all models."""
    logger = logging.getLogger(__name__)
    
    logger.info("\n" + "=" * 70)
    logger.info("TRANSCRIPTION PROCESS")
    logger.info("=" * 70)
    
    transcripts = {}
    processing_times = {}
    
    for model_name, model in models.items():
        logger.info(f"\nTranscribing with {model_name}...")
        
        try:
            # Load model if not already loaded
            if not model.is_loaded():
                model.load_model()
            
            # Run transcription
            start_time = pd.Timestamp.now()
            
            if hasattr(model, 'transcribe_batch'):
                results = model.transcribe_batch(audio_arrays, sample_rates)
                model_transcripts = [result.text for result in results]
                model_confidences = [result.confidence for result in results]
            else:
                model_transcripts = []
                model_confidences = []
                for audio, sr in tqdm(zip(audio_arrays, sample_rates), 
                                    total=len(audio_arrays), 
                                    desc=f"{model_name}"):
                    result = model.transcribe(audio, sr)
                    model_transcripts.append(result.text)
                    model_confidences.append(result.confidence)
            
            end_time = pd.Timestamp.now()
            processing_time = (end_time - start_time).total_seconds()
            
            # Store results
            transcripts[model_name] = model_transcripts
            processing_times[model_name] = {
                "total_seconds": processing_time,
                "avg_per_sample": processing_time / len(audio_arrays),
                "samples_per_second": len(audio_arrays) / processing_time if processing_time > 0 else 0
            }
            
            # Save transcripts
            transcripts_file = output_dir / "transcripts" / f"{model_name}_transcripts.json"
            transcripts_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(transcripts_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "model": model_name,
                    "transcripts": model_transcripts,
                    "confidences": model_confidences,
                    "processing_time": processing_times[model_name]
                }, f, indent=2, ensure_ascii=False)
            
            logger.info(f"  ✓ Completed: {len(model_transcripts)} transcriptions")
            logger.info(f"  ✓ Processing time: {processing_time:.2f}s "
                       f"({processing_times[model_name]['avg_per_sample']:.2f}s per sample)")
            logger.info(f"  ✓ Saved to: {transcripts_file}")
            
        except Exception as e:
            logger.error(f"  ✗ Failed to transcribe with {model_name}: {e}")
            transcripts[model_name] = [""] * len(audio_arrays)
            processing_times[model_name] = {
                "total_seconds": 0,
                "avg_per_sample": 0,
                "samples_per_second": 0
            }
    
    return transcripts, processing_times

def evaluate_models(reference_texts, transcripts, output_dir):
    """Evaluate all models."""
    logger = logging.getLogger(__name__)
    
    logger.info("\n" + "=" * 70)
    logger.info("MODEL EVALUATION")
    logger.info("=" * 70)
    
    # Initialize evaluator
    evaluator = ASREvaluator(language="de")
    
    # Evaluate each model
    evaluation_results = {}
    
    for model_name, model_transcripts in transcripts.items():
        logger.info(f"\nEvaluating {model_name}...")
        
        try:
            # Filter out empty transcripts
            valid_pairs = [(ref, hyp) for ref, hyp in zip(reference_texts, model_transcripts) 
                          if ref.strip() and hyp.strip()]
            
            if not valid_pairs:
                logger.warning(f"  No valid transcription pairs for {model_name}")
                continue
            
            ref_valid, hyp_valid = zip(*valid_pairs)
            
            # Compute metrics
            metrics = evaluator.evaluate(
                list(ref_valid), 
                list(hyp_valid),
                compute_ci=True
            )
            
            evaluation_results[model_name] = metrics
            
            # Save metrics
            metrics_file = output_dir / "metrics" / f"{model_name}_metrics.json"
            metrics_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "model": model_name,
                    "wer": metrics.wer,
                    "cer": metrics.cer,
                    "mer": metrics.mer,
                    "wil": metrics.wil,
                    "precision": metrics.precision,
                    "recall": metrics.recall,
                    "f1": metrics.f1,
                    "bleu": metrics.bleu,
                    "rouge": metrics.rouge,
                    "bert_score": metrics.bert_score,
                    "word_accuracy": metrics.word_accuracy,
                    "character_accuracy": metrics.character_accuracy,
                    "num_samples": metrics.num_samples,
                    "total_words": metrics.total_words
                }, f, indent=2, ensure_ascii=False)
            
            logger.info(f"  ✓ WER: {metrics.wer:.4f}")
            logger.info(f"  ✓ CER: {metrics.cer:.4f}")
            logger.info(f"  ✓ F1: {metrics.f1:.4f}")
            
        except Exception as e:
            logger.error(f"  ✗ Failed to evaluate {model_name}: {e}")
    
    return evaluation_results

def perform_statistical_analysis(evaluation_results, output_dir):
    """Perform statistical analysis on evaluation results."""
    logger = logging.getLogger(__name__)
    
    logger.info("\n" + "=" * 70)
    logger.info("STATISTICAL ANALYSIS")
    logger.info("=" * 70)
    
    # Initialize statistical analyzer
    analyzer = StatisticalAnalyzer(alpha=0.05)
    
    # Prepare data for statistical tests
    # Note: For proper statistical tests, we need per-sample scores
    # For now, we'll analyze the aggregated metrics
    
    # Create comparison DataFrame
    results_data = []
    for model_name, metrics in evaluation_results.items():
        results_data.append({
            "model": model_name,
            "wer": metrics.wer,
            "cer": metrics.cer,
            "mer": metrics.mer,
            "wil": metrics.wil,
            "f1": metrics.f1,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "bleu": metrics.bleu if metrics.bleu else 0,
            "rouge1": metrics.rouge["rouge1"] if metrics.rouge else 0,
            "word_accuracy": metrics.word_accuracy
        })
    
    results_df = pd.DataFrame(results_data)
    
    # Perform statistical analysis
    statistical_analysis = analyzer.analyze_model_comparison(results_df, metric="wer")
    
    # Save statistical analysis
    stats_file = output_dir / "statistical_analysis.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(statistical_analysis, f, indent=2, ensure_ascii=False)
    
    # Generate and save report
    report = analyzer.generate_statistical_report(statistical_analysis)
    report_file = output_dir / "statistical_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(report)
    logger.info(f"Saved statistical analysis to: {stats_file}")
    
    return results_df, statistical_analysis

def generate_visualizations(results_df, evaluation_results, output_dir):
    """Generate visualizations for the evaluation results."""
    logger = logging.getLogger(__name__)
    
    logger.info("\n" + "=" * 70)
    logger.info("VISUALIZATION GENERATION")
    logger.info("=" * 70)
    
    # Initialize visualization engine
    viz_engine = VisualizationEngine()
    
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    # Generate visualizations
    try:
        # 1. WER Comparison Bar Chart
        wer_fig = viz_engine.plot_wer_comparison(results_df)
        wer_file = figures_dir / "wer_comparison.png"
        wer_fig.savefig(wer_file, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Created WER comparison: {wer_file}")
        
        # 2. CER Comparison Bar Chart
        cer_fig = viz_engine.plot_cer_comparison(results_df)
        cer_file = figures_dir / "cer_comparison.png"
        cer_fig.savefig(cer_file, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Created CER comparison: {cer_file}")
        
        # 3. Radar Chart for Multiple Metrics
        radar_fig = viz_engine.plot_radar_chart(results_df)
        radar_file = figures_dir / "performance_radar.png"
        radar_fig.savefig(radar_file, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Created radar chart: {radar_file}")
        
        # 4. Processing Time Comparison
        # (Would need processing times data)
        
        # 5. Error Type Breakdown
        # (Would need detailed error analysis)
        
        logger.info(f"All visualizations saved to: {figures_dir}")
        
    except Exception as e:
        logger.error(f"Failed to generate visualizations: {e}")

def perform_error_analysis(reference_texts, transcripts, output_dir):
    """Perform detailed error analysis."""
    logger = logging.getLogger(__name__)
    
    logger.info("\n" + "=" * 70)
    logger.info("ERROR ANALYSIS")
    logger.info("=" * 70)
    
    # Initialize error analyzer
    error_analyzer = ErrorAnalyzer(language="de")
    
    error_analysis_dir = output_dir / "error_analysis"
    error_analysis_dir.mkdir(exist_ok=True)
    
    for model_name, model_transcripts in transcripts.items():
        logger.info(f"\nAnalyzing errors for {model_name}...")
        
        try:
            # Analyze errors
            error_analysis = error_analyzer.analyze(
                reference_texts, 
                model_transcripts,
                model_name
            )
            
            # Save error analysis
            error_file = error_analysis_dir / f"{model_name}_errors.json"
            with open(error_file, 'w', encoding='utf-8') as f:
                json.dump(error_analysis, f, indent=2, ensure_ascii=False)
            
            # Generate error report
            error_report = error_analyzer.generate_report(error_analysis)
            report_file = error_analysis_dir / f"{model_name}_error_report.txt"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(error_report)
            
            logger.info(f"  ✓ Error analysis saved to: {error_file}")
            logger.info(f"  ✓ Error patterns: {error_analysis.get('common_error_patterns', [])[:3]}")
            
        except Exception as e:
            logger.error(f"  ✗ Failed to analyze errors for {model_name}: {e}")

def generate_final_report(results_df, statistical_analysis, output_dir):
    """Generate final comprehensive report."""
    logger = logging.getLogger(__name__)
    
    logger.info("\n" + "=" * 70)
    logger.info("FINAL REPORT GENERATION")
    logger.info("=" * 70)
    
    # Create report directory
    report_dir = output_dir / "reports"
    report_dir.mkdir(exist_ok=True)
    
    # Generate markdown report
    report_md = generate_markdown_report(results_df, statistical_analysis, output_dir)
    
    report_file = report_dir / "asr_evaluation_report.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_md)
    
    # Generate HTML report
    try:
        import markdown
        html_report = markdown.markdown(report_md, extensions=['tables', 'fenced_code'])
        
        # Add HTML template
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>ASR Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .best {{ background-color: #d4edda !important; }}
                .worst {{ background-color: #f8d7da !important; }}
                .metric {{ font-weight: bold; color: #3498db; }}
            </style>
        </head>
        <body>
            {html_report}
        </body>
        </html>
        """
        
        html_file = report_dir / "asr_evaluation_report.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_template)
        
        logger.info(f"✓ HTML report: {html_file}")
        
    except ImportError:
        logger.warning("Markdown package not installed, skipping HTML report")
    
    logger.info(f"✓ Markdown report: {report_file}")
    
    return report_file

def generate_markdown_report(results_df, statistical_analysis, output_dir):
    """Generate markdown report."""
    
    report = []
    
    # Header
    report.append(f"# ASR Model Evaluation Report")
    report.append("")
    report.append(f"**Experiment:** {config.experiment.name}")
    report.append(f"**Version:** {config.experiment.version}")
    report.append(f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Executive Summary
    report.append("## Executive Summary")
    report.append("")
    
    best_model = results_df.loc[results_df['wer'].idxmin()]
    worst_model = results_df.loc[results_df['wer'].idxmax()]
    
    report.append(f"- **Best performing model:** {best_model['model']} (WER: {best_model['wer']:.3f})")
    report.append(f"- **Worst performing model:** {worst_model['model']} (WER: {worst_model['wer']:.3f})")
    report.append(f"- **Number of models evaluated:** {len(results_df)}")
    report.append(f"- **Average WER across models:** {results_df['wer'].mean():.3f}")
    report.append("")
    
    # Results Table
    report.append("## Results Summary")
    report.append("")
    
    # Sort by WER
    results_sorted = results_df.sort_values('wer').reset_index(drop=True)
    
    report.append("| Rank | Model | WER ↓ | CER ↓ | F1 ↑ | BLEU ↑ |")
    report.append("|------|-------|-------|-------|------|--------|")
    
    for idx, row in results_sorted.iterrows():
        report.append(f"| {idx+1} | {row['model']} | {row['wer']:.3f} | {row['cer']:.3f} | "
                     f"{row['f1']:.3f} | {row.get('bleu', 0):.3f} |")
    report.append("")
    
    # Statistical Analysis
    if statistical_analysis:
        report.append("## Statistical Analysis")
        report.append("")
        
        if 'best_model' in statistical_analysis:
            report.append(f"**Best model statistically:** {statistical_analysis['best_model']}")
            report.append("")
        
        if 'descriptive_statistics' in statistical_analysis:
            stats = statistical_analysis['descriptive_statistics']
            report.append("### Descriptive Statistics for WER")
            report.append("")
            report.append(f"- **Mean:** {stats.get('mean', 0):.3f}")
            report.append(f"- **Median:** {stats.get('median', 0):.3f}")
            report.append(f"- **Standard Deviation:** {stats.get('std', 0):.3f}")
            report.append(f"- **Range:** {stats.get('min', 0):.3f} - {stats.get('max', 0):.3f}")
            report.append("")
    
    # Visualizations
    report.append("## Visualizations")
    report.append("")
    
    figures = [
        ("wer_comparison.png", "Word Error Rate Comparison"),
        ("cer_comparison.png", "Character Error Rate Comparison"),
        ("performance_radar.png", "Performance Radar Chart")
    ]
    
    for fig_file, fig_title in figures:
        fig_path = f"figures/{fig_file}"
        if (output_dir / fig_path).exists():
            report.append(f"### {fig_title}")
            report.append(f"![{fig_title}]({fig_path})")
            report.append("")
    
    # Conclusions and Recommendations
    report.append("## Conclusions and Recommendations")
    report.append("")
    
    report.append("### Key Findings")
    report.append("")
    report.append("1. **Best overall model:** Based on WER, CER, and F1 scores")
    report.append("2. **Trade-offs:** Note any speed/accuracy trade-offs")
    report.append("3. **Error patterns:** Common errors across models")
    report.append("")
    
    report.append("### Recommendations")
    report.append("")
    report.append("1. **For medical applications:** Consider models with lower CER for medical terms")
    report.append("2. **For real-time applications:** Consider processing speed")
    report.append("3. **For deployment:** Consider model size and resource requirements")
    report.append("")
    
    # Appendices
    report.append("## Appendices")
    report.append("")
    report.append("### Detailed Results")
    report.append("")
    report.append("Complete results are available in the following files:")
    report.append("- `results_summary.csv`: CSV file with all metrics")
    report.append("- `statistical_analysis.json`: Detailed statistical analysis")
    report.append("- `error_analysis/`: Directory with error analysis for each model")
    report.append("- `transcripts/`: Directory with raw transcripts")
    report.append("")
    
    return "\n".join(report)

def main():
    """Main function to run the full evaluation pipeline."""
    
    # Setup
    output_dir = setup_experiment()
    
    try:
        # 1. Load data
        reference_texts, audio_arrays, sample_rates, samples = load_and_prepare_data()
        
        # 2. Initialize models
        models = initialize_asr_models()
        
        if not models:
            logger.error("No models initialized. Exiting.")
            return
        
        # 3. Run transcriptions
        transcripts, processing_times = run_transcriptions(
            models, audio_arrays, sample_rates, output_dir
        )
        
        # 4. Evaluate models
        evaluation_results = evaluate_models(reference_texts, transcripts, output_dir)
        
        if not evaluation_results:
            logger.error("No evaluation results. Exiting.")
            return
        
        # 5. Statistical analysis
        results_df, statistical_analysis = perform_statistical_analysis(
            evaluation_results, output_dir
        )
        
        # 6. Generate visualizations
        generate_visualizations(results_df, evaluation_results, output_dir)
        
        # 7. Error analysis
        perform_error_analysis(reference_texts, transcripts, output_dir)
        
        # 8. Generate final report
        generate_final_report(results_df, statistical_analysis, output_dir)
        
        # 9. Save final results
        results_file = output_dir / "results_summary.csv"
        results_df.to_csv(results_file, index=False)
        
        # Final summary
        logger.info("\n" + "=" * 70)
        logger.info("EXPERIMENT COMPLETE")
        logger.info("=" * 70)
        logger.info(f"All results saved to: {output_dir}")
        logger.info(f"Final results: {results_file}")
        logger.info("")
        logger.info("Top 3 models by WER:")
        top_3 = results_df.sort_values('wer').head(3)
        for idx, row in top_3.iterrows():
            logger.info(f"  {idx+1}. {row['model']}: WER={row['wer']:.3f}, CER={row['cer']:.3f}")
        logger.info("")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()