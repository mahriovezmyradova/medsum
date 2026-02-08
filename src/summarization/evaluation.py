# src/summarization/evaluation.py
"""
Evaluation metrics for summarization quality.
"""
import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field
import evaluate
from sentence_transformers import SentenceTransformer, util
import logging

logger = logging.getLogger(__name__)

@dataclass
class SummaryMetrics:
    """Comprehensive summarization metrics."""
    # Text similarity metrics
    rouge1: float = 0.0
    rouge2: float = 0.0
    rougeL: float = 0.0
    rougeLsum: float = 0.0
    
    bleu: float = 0.0
    bert_score_f1: float = 0.0
    semantic_similarity: float = 0.0
    
    # Content metrics
    compression_ratio: float = 0.0
    informativeness: float = 0.0
    coherence_score: float = 0.0
    
    # Medical-specific metrics
    medical_keyword_coverage: float = 0.0
    medical_content_preserved: float = 0.0
    
    # Metadata
    num_reference_words: int = 0
    num_summary_words: int = 0
    num_reference_sents: int = 0
    num_summary_sents: int = 0

class SummarizationEvaluator:
    """Evaluate summarization quality."""
    
    def __init__(self, language: str = "de"):
        self.language = language
        
        # Load metrics
        self.rouge = evaluate.load("rouge")
        self.bleu = evaluate.load("bleu")
        
        # Try to load BERT score
        try:
            self.bert = evaluate.load("bertscore")
        except:
            logger.warning("BERT score not available")
            self.bert = None
        
        # Load sentence transformer for semantic similarity
        try:
            if language == "de":
                self.sentence_model = SentenceTransformer(
                    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                )
            else:
                self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        except:
            logger.warning("Sentence transformer not available")
            self.sentence_model = None
    
    # src/summarization/evaluation.py
# Update the evaluate method:

    def evaluate(self, reference: str, summary: str) -> SummaryMetrics:
        """Evaluate summary against reference."""
        metrics = SummaryMetrics()
        
        # Basic statistics
        ref_words = reference.split()
        sum_words = summary.split()
        
        metrics.num_reference_words = len(ref_words)
        metrics.num_summary_words = len(sum_words)
        
        if len(ref_words) > 0:
            metrics.compression_ratio = len(sum_words) / len(ref_words)
        
        # ROUGE scores
        rouge_result = self.rouge.compute(
            predictions=[summary],
            references=[reference],
            use_stemmer=True
        )
        
        metrics.rouge1 = rouge_result["rouge1"]
        metrics.rouge2 = rouge_result["rouge2"]
        metrics.rougeL = rouge_result["rougeL"]
        metrics.rougeLsum = rouge_result["rougeLsum"]
        
        # BLEU score with German tokenization
        try:
            # Tokenize German text for BLEU
            import nltk
            
            # Simple word tokenization for German
            ref_tokens = nltk.word_tokenize(reference.lower(), language='german')
            sum_tokens = nltk.word_tokenize(summary.lower(), language='german')
            
            bleu_result = self.bleu.compute(
                predictions=[sum_tokens],
                references=[[ref_tokens]]
            )
            metrics.bleu = bleu_result["bleu"]
        except:
            # Fallback to simple split
            bleu_result = self.bleu.compute(
                predictions=[summary.split()],
                references=[[reference.split()]]
            )
            metrics.bleu = bleu_result["bleu"]
        
        # BERT score
        if self.bert:
            bert_result = self.bert.compute(
                predictions=[summary],
                references=[reference],
                lang=self.language
            )
            metrics.bert_score_f1 = float(np.mean(bert_result["f1"]))
        
        # Semantic similarity
        if self.sentence_model:
            embeddings = self.sentence_model.encode([reference, summary])
            similarity = util.cos_sim(embeddings[0], embeddings[1])
            metrics.semantic_similarity = float(similarity[0][0])
        
        # Medical keyword coverage (simplified)
        metrics.medical_keyword_coverage = min(1.0, self._calculate_medical_coverage(reference, summary))
        
        return metrics
    
        def _calculate_medical_coverage(self, reference: str, summary: str) -> float:

        # Simple medical keywords for German
            medical_keywords = {
                'patient', 'arzt', 'krankheit', 'symptom', 'diagnose',
                'behandlung', 'medikament', 'therapie', 'operation',
                'schmerz', 'fieber', 'blut', 'druck', 'herz', 'lunge',
                'infektion', 'antibiotikum', 'kopfschmerzen', 'halsschmerzen',
                'übelkeit', 'temperatur', 'wasser', 'ruhe', 'kontrolle'
            }
            
            ref_lower = reference.lower()
            sum_lower = summary.lower()
            
            # Find unique keywords in reference
            ref_keywords = set()
            for kw in medical_keywords:
                if kw in ref_lower:
                    ref_keywords.add(kw)
            
            # Find keywords in summary
            sum_keywords = set()
            for kw in medical_keywords:
                if kw in sum_lower:
                    sum_keywords.add(kw)
            
            if not ref_keywords:
                return 0.0
            
            # Coverage = keywords in summary / keywords in reference
            coverage = len(sum_keywords.intersection(ref_keywords)) / len(ref_keywords)
            return min(1.0, coverage)  # Cap at 1.0
    
    def evaluate_batch(self, references: List[str], summaries: List[str]) -> Dict[str, float]:
        """Evaluate batch of summaries."""
        if len(references) != len(summaries):
            raise ValueError("References and summaries must have same length")
        
        all_metrics = []
        
        for ref, sum in zip(references, summaries):
            metrics = self.evaluate(ref, sum)
            all_metrics.append(metrics)
        
        # Aggregate results
        aggregated = {}
        for field in SummaryMetrics.__dataclass_fields__:
            values = [getattr(m, field) for m in all_metrics]
            aggregated[f"{field}_mean"] = float(np.mean(values))
            aggregated[f"{field}_std"] = float(np.std(values))
        
        return aggregated