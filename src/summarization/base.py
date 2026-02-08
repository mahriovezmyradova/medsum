# src/summarization/base.py
"""
Base classes for summarization models.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

@dataclass
class SummaryResult:
    """Result of a summarization."""
    summary: str
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self):
        return f"Summary({len(self.summary)} chars, {self.processing_time:.2f}s)"

class BaseSummarizer(ABC):
    """Abstract base class for all summarizers."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize summarizer.
        
        Args:
            config: Model configuration
        """
        self.config = config
        self.name = config.get("name", self.__class__.__name__)
        self.language = config.get("language", "de")
        self._is_loaded = False
        
        logger.info(f"Initialized {self.name} for language: {self.language}")
    
    @abstractmethod
    def load_model(self):
        """Load the summarization model."""
        pass
    
    @abstractmethod
    def summarize(self, text: str, **kwargs) -> SummaryResult:
        """
        Summarize text.
        
        Args:
            text: Text to summarize
            
        Returns:
            SummaryResult object
        """
        pass
    
    def summarize_batch(self, texts: List[str], **kwargs) -> List[SummaryResult]:
        """
        Summarize a batch of texts.
        
        Args:
            texts: List of texts to summarize
            
        Returns:
            List of SummaryResult objects
        """
        if not self._is_loaded:
            self.load_model()
        
        results = []
        for i, text in enumerate(texts):
            try:
                result = self.summarize(text, **kwargs)
                results.append(result)
                
                # Log progress
                if (i + 1) % 10 == 0:
                    logger.info(f"Summarized {i + 1}/{len(texts)} texts "
                              f"({(i + 1)/len(texts)*100:.1f}%)")
                    
            except Exception as e:
                logger.error(f"Failed to summarize text {i}: {e}")
                results.append(SummaryResult(
                    summary="",
                    processing_time=0.0,
                    metadata={"error": str(e)}
                ))
        
        logger.info(f"Batch summarization complete: {len(texts)} texts")
        return results
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        pass
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded