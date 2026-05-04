"""
Medical summarization framework for German medical conversations.

Three summarizers are provided:
  - ExtractiveSummarizer  : sentence-ranking with German BERT embeddings
  - BartSummarizer        : philschmid/bart-large-german-samsum  (dialogue-tuned)
  - MT5Summarizer         : google/mt5-base  with German task prefix

MedicalSummarizer is the high-level class that combines extractive → abstractive
for long texts and exposes the same interface as the individual summarizers.

SummarizationEvaluator computes ROUGE, BLEU, BERTScore and medical-term
preservation for any summarizer on any dataset.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Centralised medical vocabulary – single source of truth
from src.medical.terminology import extract_medical_terms

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Data containers
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SummaryResult:
    summary: str
    processing_time: float
    model_name: str
    input_length: int
    compression_ratio: float
    medical_terms_preserved: float
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# Base class
# ──────────────────────────────────────────────────────────────────────────────

class BaseSummarizer:
    """Abstract base for all summarizers."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = config.get("name", self.__class__.__name__)
        self.language = config.get("language", "de")
        self.device = config.get(
            "device",
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else "cpu",
        )
        self._model = None
        self._tokenizer = None
        self._is_loaded = False
        logger.info("Initialized %s on %s", self.name, self.device)

    def load_model(self):
        raise NotImplementedError

    def summarize(self, text: str) -> SummaryResult:
        raise NotImplementedError

    # ── helpers ──────────────────────────────────────────────────────────────

    def _medical_preservation(self, original: str, summary: str) -> float:
        """Use centralised terminology for term-preservation scoring."""
        from src.medical.terminology import medical_term_preservation
        return medical_term_preservation(original, summary)

    def _empty_result(self, text: str, t0: float, error: str = "") -> SummaryResult:
        return SummaryResult(
            summary="",
            processing_time=time.time() - t0,
            model_name=self.name,
            input_length=len(text.split()),
            compression_ratio=0.0,
            medical_terms_preserved=0.0,
            metadata={"error": error},
        )


# ──────────────────────────────────────────────────────────────────────────────
# Extractive summarizer
# ──────────────────────────────────────────────────────────────────────────────

class ExtractiveSummarizer(BaseSummarizer):
    """
    Sentence-ranking extractive summarizer for German medical text.

    Scores each sentence by:
      - cosine similarity to the document centroid (via multilingual MiniLM)
      - density of medical terms
      - positional bias (first/last sentences carry more information)
    """

    _SENTENCE_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.num_sentences = config.get("num_sentences", 3)
        self._sentence_encoder = None

    def load_model(self):
        if self._is_loaded:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._sentence_encoder = SentenceTransformer(
                self._SENTENCE_MODEL, device=self.device
            )
            logger.info("Loaded sentence encoder: %s", self._SENTENCE_MODEL)
        except Exception as exc:
            logger.warning("sentence-transformers not available (%s) – using TF-IDF fallback", exc)
        self._is_loaded = True

    # ── sentence splitting ────────────────────────────────────────────────────

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Split German text into sentences, preserving medical abbreviations."""
        text = re.sub(r"\s+", " ", text.strip())
        # Split at sentence boundaries followed by a capital letter (German nouns)
        parts = re.split(r"(?<=[.!?])\s+(?=[A-ZÄÖÜ])", text)
        result: List[str] = []
        for part in parts:
            if len(part.split()) > 25 and ";" in part:
                result.extend(s.strip() for s in part.split(";") if s.strip())
            else:
                result.append(part)
        return [s for s in result if len(s.split()) >= 3]

    # ── scoring ───────────────────────────────────────────────────────────────

    def _score_sentences(self, sentences: List[str], text: str) -> List[float]:
        n = len(sentences)
        scores = []

        if self._sentence_encoder is not None:
            try:
                embeddings = self._sentence_encoder.encode(
                    sentences + [text], convert_to_numpy=True, show_progress_bar=False
                )
                doc_vec = embeddings[-1]
                sent_vecs = embeddings[:-1]
                from sklearn.metrics.pairwise import cosine_similarity
                sims = cosine_similarity(sent_vecs, doc_vec.reshape(1, -1)).flatten()
            except Exception:
                sims = np.full(n, 0.5)
        else:
            # TF-IDF fallback
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.metrics.pairwise import cosine_similarity
                vect = TfidfVectorizer(min_df=1)
                mat = vect.fit_transform(sentences + [text])
                sims = cosine_similarity(mat[:-1], mat[-1]).flatten()
            except Exception:
                sims = np.full(n, 0.5)

        for i, (sent, sim) in enumerate(zip(sentences, sims)):
            score = float(sim)
            # Positional bias
            if i == 0 or i == n - 1:
                score += 0.15
            elif i < n * 0.2 or i > n * 0.8:
                score += 0.08
            # Medical term density
            score += min(len(extract_medical_terms(sent)) * 0.05, 0.25)
            # Ideal sentence length
            wc = len(sent.split())
            if 6 <= wc <= 25:
                score += 0.10
            elif wc < 4:
                score -= 0.15
            scores.append(max(0.0, min(1.0, score)))

        return scores

    # ── public interface ──────────────────────────────────────────────────────

    def summarize(self, text: str) -> SummaryResult:
        t0 = time.time()
        if not self._is_loaded:
            self.load_model()
        try:
            sentences = self._split_sentences(text)
            if not sentences:
                return self._empty_result(text, t0, "no sentences found")

            scores = self._score_sentences(sentences, text)
            k = min(self.num_sentences, len(sentences))
            top_idx = sorted(np.argsort(scores)[-k:])
            summary = " ".join(sentences[i] for i in top_idx)

            input_len = len(text.split())
            return SummaryResult(
                summary=summary,
                processing_time=time.time() - t0,
                model_name=self.name,
                input_length=input_len,
                compression_ratio=len(summary.split()) / input_len if input_len else 0.0,
                medical_terms_preserved=self._medical_preservation(text, summary),
                metadata={
                    "strategy": "extractive",
                    "selected_indices": [int(i) for i in top_idx],
                },
            )
        except Exception as exc:
            logger.error("Extractive summarization failed: %s", exc)
            return self._empty_result(text, t0, str(exc))


# ──────────────────────────────────────────────────────────────────────────────
# BART summarizer (German dialogue-tuned)
# ──────────────────────────────────────────────────────────────────────────────

class BartSummarizer(BaseSummarizer):
    """
    Abstractive summarizer using philschmid/bart-large-german-samsum.

    This model is fine-tuned on SAMSum (dialogue → summary) with German text,
    making it well-suited for patient–doctor conversation summarization.
    """

    _DEFAULT_MODEL = "philschmid/bart-large-german-samsum"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_name = config.get("model_name", self._DEFAULT_MODEL)
        self.max_length = config.get("max_length", 180)
        self.min_length = config.get("min_length", 30)
        self.num_beams = config.get("num_beams", 6)
        self.length_penalty = config.get("length_penalty", 2.0)

    def load_model(self):
        if self._is_loaded:
            return
        logger.info("Loading BART model: %s", self.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self._model.to(self.device)
        self._model.eval()
        self._is_loaded = True
        logger.info("BART model loaded on %s", self.device)

    def summarize(self, text: str) -> SummaryResult:
        t0 = time.time()
        if not self._is_loaded:
            self.load_model()
        try:
            inputs = self._tokenizer(
                text, return_tensors="pt", truncation=True, max_length=1024
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                ids = self._model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=self.max_length,
                    min_length=self.min_length,
                    num_beams=self.num_beams,
                    length_penalty=self.length_penalty,
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                )
            summary = self._tokenizer.decode(ids[0], skip_special_tokens=True).strip()

            input_len = len(text.split())
            return SummaryResult(
                summary=summary,
                processing_time=time.time() - t0,
                model_name=self.name,
                input_length=input_len,
                compression_ratio=len(summary.split()) / input_len if input_len else 0.0,
                medical_terms_preserved=self._medical_preservation(text, summary),
                metadata={"model": self.model_name, "num_beams": self.num_beams},
            )
        except Exception as exc:
            logger.error("BART summarization failed: %s", exc)
            return self._empty_result(text, t0, str(exc))


# ──────────────────────────────────────────────────────────────────────────────
# mT5 summarizer
# ──────────────────────────────────────────────────────────────────────────────

class MT5Summarizer(BaseSummarizer):
    """
    Abstractive summarizer using google/mt5-base.

    The German task prefix "zusammenfassen: " guides the multilingual model
    toward the summarization task.  mt5-base is preferred over mt5-small for
    better German fluency while remaining feasible on CPU.
    """

    _DEFAULT_MODEL = "google/mt5-base"
    _TASK_PREFIX = "zusammenfassen: "   # German – do NOT use English "summarize:"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_name = config.get("model_name", self._DEFAULT_MODEL)
        self.max_length = config.get("max_length", 150)
        self.min_length = config.get("min_length", 20)
        self.num_beams = config.get("num_beams", 4)

    def load_model(self):
        if self._is_loaded:
            return
        logger.info("Loading mT5 model: %s", self.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self._model.to(self.device)
        self._model.eval()
        self._is_loaded = True
        logger.info("mT5 model loaded on %s", self.device)

    def summarize(self, text: str) -> SummaryResult:
        t0 = time.time()
        if not self._is_loaded:
            self.load_model()
        try:
            prefixed = self._TASK_PREFIX + text
            inputs = self._tokenizer(
                prefixed, return_tensors="pt", truncation=True, max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                ids = self._model.generate(
                    inputs["input_ids"],
                    max_length=self.max_length,
                    min_length=self.min_length,
                    num_beams=self.num_beams,
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                )
            summary = self._tokenizer.decode(ids[0], skip_special_tokens=True).strip()

            input_len = len(text.split())
            return SummaryResult(
                summary=summary,
                processing_time=time.time() - t0,
                model_name=self.name,
                input_length=input_len,
                compression_ratio=len(summary.split()) / input_len if input_len else 0.0,
                medical_terms_preserved=self._medical_preservation(text, summary),
                metadata={"model": self.model_name, "prefix": self._TASK_PREFIX},
            )
        except Exception as exc:
            logger.error("mT5 summarization failed: %s", exc)
            return self._empty_result(text, t0, str(exc))


# ──────────────────────────────────────────────────────────────────────────────
# High-level medical summarizer (extractive → abstractive pipeline)
# ──────────────────────────────────────────────────────────────────────────────

class MedicalSummarizer(BaseSummarizer):
    """
    Two-stage pipeline:
      1. Extractive pass condenses very long conversations.
      2. Abstractive pass (BART by default) generates a fluent German summary.

    For texts ≤ `extractive_threshold` words, only the abstractive pass runs.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.extractive = ExtractiveSummarizer({**config, "name": "extractive"})
        abstractive_cls = MT5Summarizer if config.get("use_mt5") else BartSummarizer
        self.abstractive = abstractive_cls({**config, "name": config.get("abstractive_name", "bart")})
        self.extractive_threshold = config.get("extractive_threshold", 120)

    def load_model(self):
        self.extractive.load_model()
        self.abstractive.load_model()
        self._is_loaded = True
        logger.info("MedicalSummarizer ready (extractive + %s)", self.abstractive.name)

    def summarize(self, text: str) -> SummaryResult:
        t0 = time.time()
        if not self._is_loaded:
            self.load_model()
        text = _preprocess_german_medical(text)

        if len(text.split()) > self.extractive_threshold:
            ext = self.extractive.summarize(text)
            intermediate = ext.summary or text
            result = self.abstractive.summarize(intermediate)
            result.model_name = self.name
            result.input_length = len(text.split())
            result.compression_ratio = (
                len(result.summary.split()) / result.input_length
                if result.input_length else 0.0
            )
            result.medical_terms_preserved = self._medical_preservation(text, result.summary)
            result.metadata.update(
                strategy="extractive_then_abstractive",
                intermediate_length=len(intermediate.split()),
            )
            result.processing_time = time.time() - t0
        else:
            result = self.abstractive.summarize(text)
            result.model_name = self.name
            result.metadata.update(strategy="abstractive_only")

        return result


# ──────────────────────────────────────────────────────────────────────────────
# Text preprocessing helper
# ──────────────────────────────────────────────────────────────────────────────

_ABBREVIATIONS = {
    r"\bDr\.\b": "Doktor",
    r"\bProf\.\b": "Professor",
    r"\bMed\.\b": "Medizin",
    r"\bPat\.\b": "Patient",
    r"\bca\.\b": "circa",
    r"\bz\.B\.\b": "zum Beispiel",
    r"\bd\.h\.\b": "das heißt",
    r"\busw\.\b": "und so weiter",
    r"\bzw\.\b": "beziehungsweise",
    r"\bggf\.\b": "gegebenenfalls",
    r"\bi\.d\.R\.\b": "in der Regel",
}


def _preprocess_german_medical(text: str) -> str:
    """Normalise whitespace and expand common German medical abbreviations."""
    text = re.sub(r"\s+", " ", text.strip())
    text = re.sub(r"\.([A-ZÄÖÜ])", r". \1", text)
    for pattern, replacement in _ABBREVIATIONS.items():
        text = re.sub(pattern, replacement, text)
    return text


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation helper
# ──────────────────────────────────────────────────────────────────────────────

class SummarizationEvaluator:
    """Compute ROUGE, BLEU, BERTScore and medical-term preservation."""

    def __init__(self):
        self._rouge = None
        self._bert_score_available = False
        self._init_metrics()

    def _init_metrics(self):
        try:
            from rouge_score import rouge_scorer
            self._rouge = rouge_scorer.RougeScorer(
                ["rouge1", "rouge2", "rougeL"], use_stemmer=True
            )
        except ImportError:
            logger.warning("rouge-score not installed – ROUGE will be 0")
        try:
            import bert_score  # noqa: F401
            self._bert_score_available = True
        except ImportError:
            logger.warning("bert-score not installed – BERTScore will be 0")

    # ── individual metrics ────────────────────────────────────────────────────

    def compute_rouge(self, reference: str, hypothesis: str) -> Dict[str, float]:
        if self._rouge is None or not hypothesis:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        scores = self._rouge.score(reference, hypothesis)
        return {k: v.fmeasure for k, v in scores.items()}

    def compute_bleu(self, reference: str, hypothesis: str) -> float:
        if not hypothesis:
            return 0.0
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            smooth = SmoothingFunction().method4
            return sentence_bleu(
                [reference.split()], hypothesis.split(), smoothing_function=smooth
            )
        except Exception:
            return 0.0

    def compute_bert_score(self, reference: str, hypothesis: str) -> float:
        if not self._bert_score_available or not hypothesis:
            return 0.0
        try:
            from bert_score import score as bs
            _, _, F1 = bs([hypothesis], [reference], lang="de", verbose=False)
            return float(F1.mean())
        except Exception:
            return 0.0

    def compute_medical_preservation(self, reference: str, hypothesis: str) -> float:
        from src.medical.terminology import medical_term_preservation
        return medical_term_preservation(reference, hypothesis)

    # ── combined evaluation ───────────────────────────────────────────────────

    def evaluate(self, reference: str, hypothesis: str) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        metrics.update(self.compute_rouge(reference, hypothesis))
        metrics["bleu"] = self.compute_bleu(reference, hypothesis)
        metrics["bert_score"] = self.compute_bert_score(reference, hypothesis)
        metrics["medical_preservation"] = self.compute_medical_preservation(
            reference, hypothesis
        )
        ref_len = len(reference.split())
        hyp_len = len(hypothesis.split())
        metrics["compression_ratio"] = hyp_len / ref_len if ref_len else 0.0
        return metrics

    def evaluate_batch(
        self,
        references: List[str],
        hypotheses: List[str],
    ) -> Dict[str, float]:
        """Return mean metric values over a list of (reference, hypothesis) pairs."""
        rows = [self.evaluate(r, h) for r, h in zip(references, hypotheses)]
        keys = rows[0].keys() if rows else []
        return {k: float(np.mean([r[k] for r in rows])) for k in keys}


__all__ = [
    "SummaryResult",
    "BaseSummarizer",
    "ExtractiveSummarizer",
    "BartSummarizer",
    "MT5Summarizer",
    "MedicalSummarizer",
    "SummarizationEvaluator",
]
