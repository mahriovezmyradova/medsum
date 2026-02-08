# src/summarization/__init__.py
"""
Summarization module for German medical conversations.
"""

from .base import BaseSummarizer, SummaryResult
from .extractive import ExtractiveSummarizer
from .abstractive import AbstractiveSummarizer
from .medical_summarizer import MedicalSummarizer
from .evaluation import SummarizationEvaluator

__all__ = [
    'BaseSummarizer', 
    'SummaryResult',
    'ExtractiveSummarizer',
    'AbstractiveSummarizer', 
    'MedicalSummarizer',
    'SummarizationEvaluator'
]