"""
Comprehensive evaluation metrics for ASR systems.
"""
import jiwer
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import logging
from dataclasses import dataclass, field
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
import evaluate
import warnings
warnings.filterwarnings("ignore")

# src/evaluation/metrics.py
# Update line 15:

# Change from:
# from ..utils.logger import get_logger

# To:
import sys
from pathlib import Path

# Add parent directory to path for imports
try:
    from utils.logger import get_logger
except ImportError:
    # Alternative import path
    sys.path.append(str(Path(__file__).parent.parent))
    from utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    # Error rates
    wer: float  # Word Error Rate
    cer: float  # Character Error Rate
    mer: float  # Match Error Rate
    wil: float  # Word Information Lost
    
    # Precision/Recall/F1
    precision: float
    recall: float
    f1: float
    
    # Text similarity metrics
    bleu: Optional[float] = None
    rouge: Optional[Dict[str, float]] = None
    bert_score: Optional[Dict[str, float]] = None
    
    # Additional metrics
    word_accuracy: float = 0.0
    character_accuracy: float = 0.0
    substitution_rate: float = 0.0
    deletion_rate: float = 0.0
    insertion_rate: float = 0.0
    
    # Processing information
    num_samples: int = 0
    total_words: int = 0
    total_chars: int = 0
    
    # Confidence intervals
    wer_ci: Optional[Tuple[float, float]] = None
    cer_ci: Optional[Tuple[float, float]] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

class ASREvaluator:
    """Professional evaluator for ASR systems."""
    
    def __init__(self, language: str = "de"):
        """
        Initialize ASR evaluator.
        
        Args:
            language: Language for evaluation (affects BERT score, tokenization)
        """
        self.language = language
        
        # Initialize metric calculators
        self.bleu_metric = evaluate.load("bleu")
        self.rouge_metric = evaluate.load("rouge")
        
        # Try to load BERT score for German
        try:
            self.bert_metric = evaluate.load("bertscore")
        except:
            logger.warning("BERT score not available, skipping")
            self.bert_metric = None
        
        logger.info(f"Initialized ASREvaluator for language: {language}")
    
    def compute_basic_metrics(self, reference: List[str], hypothesis: List[str]) -> Dict[str, float]:
        """
        Compute basic error rate metrics using jiwer.
        
        Args:
            reference: List of reference transcripts
            hypothesis: List of hypothesis transcripts
            
        Returns:
            Dictionary with basic metrics
        """
        # Clean and prepare texts
        reference_clean = [self._clean_text(text) for text in reference]
        hypothesis_clean = [self._clean_text(text) for text in hypothesis]
        
        # Compute metrics
        metrics = {}
        
        # Word Error Rate
        try:
            wer = jiwer.wer(reference_clean, hypothesis_clean)
            metrics["wer"] = wer
        except:
            metrics["wer"] = 1.0
        
        # Character Error Rate
        try:
            cer = jiwer.cer(reference_clean, hypothesis_clean)
            metrics["cer"] = cer
        except:
            metrics["cer"] = 1.0
        
        # Match Error Rate
        try:
            mer = jiwer.mer(reference_clean, hypothesis_clean)
            metrics["mer"] = mer
        except:
            metrics["mer"] = 1.0
        
        # Word Information Lost
        try:
            wil = jiwer.wil(reference_clean, hypothesis_clean)
            metrics["wil"] = wil
        except:
            metrics["wil"] = 1.0
        
        # Word and Character Accuracy
        metrics["word_accuracy"] = 1.0 - metrics["wer"]
        metrics["character_accuracy"] = 1.0 - metrics["cer"]
        
        # Compute error breakdown
        try:
            breakdown = self._compute_error_breakdown(reference_clean, hypothesis_clean)
            metrics.update(breakdown)
        except:
            pass
        
        return metrics
    
    def _clean_text(self, text: str) -> str:
        """Clean text for evaluation."""
        import re
        
        # Convert to lowercase for German
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove punctuation (keep for CER but remove for WER)
        text = re.sub(r'[^\w\säöüß]', '', text)
        
        return text.strip()
    
    def _compute_error_breakdown(self, reference: List[str], hypothesis: List[str]) -> Dict[str, float]:
        """Compute detailed error breakdown."""
        from jiwer import compute_measures
        
        total_substitutions = 0
        total_deletions = 0
        total_insertions = 0
        total_hits = 0
        total_reference_words = 0
        
        for ref, hyp in zip(reference, hypothesis):
            measures = compute_measures(ref, hyp)
            
            total_substitutions += measures['substitutions']
            total_deletions += measures['deletions']
            total_insertions += measures['insertions']
            total_hits += measures['hits']
            total_reference_words += measures['substitutions'] + measures['deletions'] + measures['hits']
        
        if total_reference_words == 0:
            return {
                "substitution_rate": 0.0,
                "deletion_rate": 0.0,
                "insertion_rate": 0.0
            }
        
        return {
            "substitution_rate": total_substitutions / total_reference_words,
            "deletion_rate": total_deletions / total_reference_words,
            "insertion_rate": total_insertions / total_reference_words
        }
    
    def compute_precision_recall_f1(self, reference: List[str], hypothesis: List[str]) -> Dict[str, float]:
        """
        Compute precision, recall, and F1 score at word level.
        
        Args:
            reference: List of reference transcripts
            hypothesis: List of hypothesis transcripts
            
        Returns:
            Dictionary with precision, recall, f1
        """
        # Tokenize texts
        ref_tokens = [text.split() for text in reference]
        hyp_tokens = [text.split() for text in hypothesis]
        
        # Flatten for overall metrics
        all_ref_tokens = [token for tokens in ref_tokens for token in tokens]
        all_hyp_tokens = [token for tokens in hyp_tokens for token in tokens]
        
        # Create vocabulary
        vocabulary = list(set(all_ref_tokens + all_hyp_tokens))
        vocab_index = {word: i for i, word in enumerate(vocabulary)}
        
        # Convert to binary matrices
        ref_matrix = np.zeros((len(ref_tokens), len(vocabulary)))
        hyp_matrix = np.zeros((len(hyp_tokens), len(vocabulary)))
        
        for i, tokens in enumerate(ref_tokens):
            for token in tokens:
                if token in vocab_index:
                    ref_matrix[i, vocab_index[token]] = 1
        
        for i, tokens in enumerate(hyp_tokens):
            for token in tokens:
                if token in vocab_index:
                    hyp_matrix[i, vocab_index[token]] = 1
        
        # Compute metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            ref_matrix.flatten(),
            hyp_matrix.flatten(),
            average='binary',
            zero_division=0
        )
        
        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1)
        }
    
    def compute_bleu(self, reference: List[str], hypothesis: List[str]) -> float:
        """
        Compute BLEU score.
        
        Args:
            reference: List of reference transcripts
            hypothesis: List of hypothesis transcripts
            
        Returns:
            BLEU score
        """
        try:
            # BLEU expects list of references for each hypothesis
            references = [[ref] for ref in reference]
            
            result = self.bleu_metric.compute(
                predictions=hypothesis,
                references=references
            )
            
            return result["bleu"]
        except Exception as e:
            logger.warning(f"Failed to compute BLEU: {e}")
            return 0.0
    
    def compute_rouge(self, reference: List[str], hypothesis: List[str]) -> Dict[str, float]:
        """
        Compute ROUGE scores.
        
        Args:
            reference: List of reference transcripts
            hypothesis: List of hypothesis transcripts
            
        Returns:
            Dictionary with ROUGE scores
        """
        try:
            result = self.rouge_metric.compute(
                predictions=hypothesis,
                references=reference,
                use_aggregator=True
            )
            
            return {
                "rouge1": result["rouge1"],
                "rouge2": result["rouge2"],
                "rougeL": result["rougeL"],
                "rougeLsum": result["rougeLsum"]
            }
        except Exception as e:
            logger.warning(f"Failed to compute ROUGE: {e}")
            return {
                "rouge1": 0.0,
                "rouge2": 0.0,
                "rougeL": 0.0,
                "rougeLsum": 0.0
            }
    
    def compute_bert_score(self, reference: List[str], hypothesis: List[str]) -> Optional[Dict[str, float]]:
        """
        Compute BERT score for semantic similarity.
        
        Args:
            reference: List of reference transcripts
            hypothesis: List of hypothesis transcripts
            
        Returns:
            Dictionary with BERT score metrics or None
        """
        if self.bert_metric is None:
            return None
        
        try:
            # Use German BERT model if available
            model_type = "bert-base-german-cased" if self.language == "de" else "bert-base-uncased"
            
            result = self.bert_metric.compute(
                predictions=hypothesis,
                references=reference,
                lang=self.language,
                model_type=model_type
            )
            
            return {
                "bert_precision": float(np.mean(result["precision"])),
                "bert_recall": float(np.mean(result["recall"])),
                "bert_f1": float(np.mean(result["f1"]))
            }
        except Exception as e:
            logger.warning(f"Failed to compute BERT score: {e}")
            return None
    
    def compute_confidence_intervals(self, reference: List[str], hypothesis: List[str], 
                                   metric: str = "wer", n_bootstrap: int = 1000, 
                                   confidence_level: float = 0.95) -> Tuple[float, float]:
        """
        Compute bootstrap confidence intervals for a metric.
        
        Args:
            reference: List of reference transcripts
            hypothesis: List of hypothesis transcripts
            metric: Metric to compute CI for ("wer", "cer")
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        n = len(reference)
        bootstrap_metrics = []
        
        for _ in range(n_bootstrap):
            # Sample with replacement
            indices = np.random.choice(n, n, replace=True)
            ref_sample = [reference[i] for i in indices]
            hyp_sample = [hypothesis[i] for i in indices]
            
            # Compute metric
            if metric == "wer":
                metric_val = jiwer.wer(ref_sample, hyp_sample)
            elif metric == "cer":
                metric_val = jiwer.cer(ref_sample, hyp_sample)
            else:
                raise ValueError(f"Unsupported metric for CI: {metric}")
            
            bootstrap_metrics.append(metric_val)
        
        # Compute confidence interval
        alpha = (1 - confidence_level) / 2
        lower = np.percentile(bootstrap_metrics, alpha * 100)
        upper = np.percentile(bootstrap_metrics, (1 - alpha) * 100)
        
        return float(lower), float(upper)
    
    def evaluate(self, reference: List[str], hypothesis: List[str], 
                compute_ci: bool = False) -> EvaluationMetrics:
        """
        Comprehensive evaluation of ASR transcriptions.
        
        Args:
            reference: List of reference transcripts
            hypothesis: List of hypothesis transcripts
            compute_ci: Whether to compute confidence intervals
            
        Returns:
            EvaluationMetrics object
        """
        # Validate inputs
        if len(reference) != len(hypothesis):
            raise ValueError(f"Reference and hypothesis must have same length. "
                           f"Got {len(reference)} vs {len(hypothesis)}")
        
        n_samples = len(reference)
        
        logger.info(f"Evaluating {n_samples} samples...")
        
        # Compute all metrics
        basic_metrics = self.compute_basic_metrics(reference, hypothesis)
        prf_metrics = self.compute_precision_recall_f1(reference, hypothesis)
        bleu_score = self.compute_bleu(reference, hypothesis)
        rouge_scores = self.compute_rouge(reference, hypothesis)
        bert_scores = self.compute_bert_score(reference, hypothesis)
        
        # Compute confidence intervals if requested
        wer_ci = None
        cer_ci = None
        
        if compute_ci and n_samples >= 10:  # Need enough samples for CI
            try:
                wer_ci = self.compute_confidence_intervals(reference, hypothesis, "wer")
                cer_ci = self.compute_confidence_intervals(reference, hypothesis, "cer")
            except Exception as e:
                logger.warning(f"Failed to compute confidence intervals: {e}")
        
        # Calculate totals
        total_words = sum(len(text.split()) for text in reference)
        total_chars = sum(len(text) for text in reference)
        
        # Create metrics object
        metrics = EvaluationMetrics(
            wer=basic_metrics["wer"],
            cer=basic_metrics["cer"],
            mer=basic_metrics["mer"],
            wil=basic_metrics["wil"],
            precision=prf_metrics["precision"],
            recall=prf_metrics["recall"],
            f1=prf_metrics["f1"],
            bleu=bleu_score,
            rouge=rouge_scores,
            bert_score=bert_scores,
            word_accuracy=basic_metrics["word_accuracy"],
            character_accuracy=basic_metrics["character_accuracy"],
            substitution_rate=basic_metrics.get("substitution_rate", 0.0),
            deletion_rate=basic_metrics.get("deletion_rate", 0.0),
            insertion_rate=basic_metrics.get("insertion_rate", 0.0),
            num_samples=n_samples,
            total_words=total_words,
            total_chars=total_chars,
            wer_ci=wer_ci,
            cer_ci=cer_ci,
            metadata={
                "language": self.language,
                "compute_ci": compute_ci,
                "has_bert_score": bert_scores is not None
            }
        )
        
        # Log results
        self._log_results(metrics)
        
        return metrics
    
    def _log_results(self, metrics: EvaluationMetrics):
        """Log evaluation results."""
        logger.info("=" * 60)
        logger.info("ASR EVALUATION RESULTS")
        logger.info("=" * 60)
        logger.info(f"Samples evaluated: {metrics.num_samples}")
        logger.info(f"Total words: {metrics.total_words:,}")
        logger.info(f"Total characters: {metrics.total_chars:,}")
        logger.info("")
        logger.info("Error Rates:")
        logger.info(f"  Word Error Rate (WER):     {metrics.wer:.4f}")
        if metrics.wer_ci:
            logger.info(f"   95% CI: [{metrics.wer_ci[0]:.4f}, {metrics.wer_ci[1]:.4f}]")
        logger.info(f"  Character Error Rate (CER): {metrics.cer:.4f}")
        if metrics.cer_ci:
            logger.info(f"   95% CI: [{metrics.cer_ci[0]:.4f}, {metrics.cer_ci[1]:.4f}]")
        logger.info(f"  Match Error Rate (MER):     {metrics.mer:.4f}")
        logger.info(f"  Word Information Lost (WIL): {metrics.wil:.4f}")
        logger.info("")
        logger.info("Word-level Metrics:")
        logger.info(f"  Precision: {metrics.precision:.4f}")
        logger.info(f"  Recall:    {metrics.recall:.4f}")
        logger.info(f"  F1 Score:  {metrics.f1:.4f}")
        logger.info("")
        logger.info("Text Similarity:")
        if metrics.bleu is not None:
            logger.info(f"  BLEU:      {metrics.bleu:.4f}")
        if metrics.rouge:
            logger.info(f"  ROUGE-1:   {metrics.rouge['rouge1']:.4f}")
            logger.info(f"  ROUGE-2:   {metrics.rouge['rouge2']:.4f}")
            logger.info(f"  ROUGE-L:   {metrics.rouge['rougeL']:.4f}")
        if metrics.bert_score:
            logger.info(f"  BERT F1:   {metrics.bert_score['bert_f1']:.4f}")
        logger.info("=" * 60)
    
    def compare_models(self, reference: List[str], 
                      hypotheses: Dict[str, List[str]]) -> pd.DataFrame:
        """
        Compare multiple ASR models.
        
        Args:
            reference: List of reference transcripts
            hypotheses: Dictionary of model_name: hypothesis_list
            
        Returns:
            DataFrame with comparison results
        """
        results = []
        
        for model_name, hypothesis in hypotheses.items():
            logger.info(f"Evaluating model: {model_name}")
            
            try:
                metrics = self.evaluate(reference, hypothesis, compute_ci=True)
                
                results.append({
                    "model": model_name,
                    "wer": metrics.wer,
                    "cer": metrics.cer,
                    "mer": metrics.mer,
                    "wil": metrics.wil,
                    "precision": metrics.precision,
                    "recall": metrics.recall,
                    "f1": metrics.f1,
                    "bleu": metrics.bleu if metrics.bleu else None,
                    "rouge1": metrics.rouge["rouge1"] if metrics.rouge else None,
                    "rouge2": metrics.rouge["rouge2"] if metrics.rouge else None,
                    "rougeL": metrics.rouge["rougeL"] if metrics.rouge else None,
                    "bert_f1": metrics.bert_score["bert_f1"] if metrics.bert_score else None,
                    "word_accuracy": metrics.word_accuracy,
                    "character_accuracy": metrics.character_accuracy,
                    "substitution_rate": metrics.substitution_rate,
                    "deletion_rate": metrics.deletion_rate,
                    "insertion_rate": metrics.insertion_rate,
                    "num_samples": metrics.num_samples
                })
                
            except Exception as e:
                logger.error(f"Failed to evaluate model {model_name}: {e}")
                results.append({
                    "model": model_name,
                    "wer": 1.0,
                    "cer": 1.0,
                    "mer": 1.0,
                    "wil": 1.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "bleu": 0.0,
                    "rouge1": 0.0,
                    "rouge2": 0.0,
                    "rougeL": 0.0,
                    "bert_f1": 0.0,
                    "word_accuracy": 0.0,
                    "character_accuracy": 0.0,
                    "substitution_rate": 0.0,
                    "deletion_rate": 0.0,
                    "insertion_rate": 0.0,
                    "num_samples": 0
                })
        
        df = pd.DataFrame(results)
        
        # Sort by WER (lower is better)
        df = df.sort_values("wer")
        
        return df