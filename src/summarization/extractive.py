"""
Extractive summarization for German medical text - FIXED VERSION.
"""
import numpy as np
from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize
import logging
import time

from .base import BaseSummarizer, SummaryResult

logger = logging.getLogger(__name__)

class ExtractiveSummarizer(BaseSummarizer):
    """BERT-based extractive summarization."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Configuration
        self.model_name = config.get("model_name", "bert-base-german-cased")
        self.num_sentences = config.get("num_sentences", 3)
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        
        # Sentence transformer for embeddings
        self.sentence_model_name = config.get("sentence_model", 
                                            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        
        # Initialize
        self.tokenizer = None
        self.model = None
        self.sentence_model = None
        
        # Download NLTK data if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        # German sentence tokenizer
        try:
            nltk.data.find('tokenizers/punkt_tab/german.pickle')
        except LookupError:
            nltk.download('punkt_tab', quiet=True)
    
    def load_model(self):
        """Load BERT model for extractive summarization."""
        if self._is_loaded:
            return
        
        logger.info(f"Loading extractive summarization model: {self.model_name}")
        
        try:
            # Load BERT for embeddings
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            
            # Load sentence transformer for better sentence embeddings
            self.sentence_model = SentenceTransformer(self.sentence_model_name)
            self.sentence_model = self.sentence_model.to(self.device)
            
            self._is_loaded = True
            logger.info(f"Extractive model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load extractive model: {e}")
            raise
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split German text into sentences."""
        sentences = sent_tokenize(text, language='german')
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_sentence_embeddings(self, sentences: List[str]) -> np.ndarray:
        """Get embeddings for sentences."""
        if self.sentence_model:
            # Use sentence transformer
            embeddings = self.sentence_model.encode(
                sentences, 
                convert_to_tensor=True,
                show_progress_bar=False
            )
            return embeddings.cpu().numpy()
        else:
            # Fallback: Use BERT CLS token
            embeddings = []
            for sentence in sentences:
                inputs = self.tokenizer(
                    sentence, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # Use CLS token embedding
                    cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    embeddings.append(cls_embedding[0])
            
            return np.array(embeddings)
    
    def _calculate_sentence_scores(self, embeddings: np.ndarray) -> np.ndarray:
        """Calculate sentence importance scores."""
        # Simple method: Use sentence centrality (similarity to other sentences)
        n = len(embeddings)
        if n <= 1:
            return np.ones(n)
        
        # Calculate cosine similarity matrix
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-8)
        similarity_matrix = np.dot(normalized, normalized.T)
        
        # Score = average similarity to other sentences
        scores = np.sum(similarity_matrix, axis=1) - 1  # Subtract self-similarity
        if n > 1:
            scores = scores / (n - 1)
        
        # Normalize scores
        if scores.max() > scores.min():
            scores = (scores - scores.min()) / (scores.max() - scores.min())
        
        return scores
    
    def summarize(self, text: str, **kwargs) -> SummaryResult:
        """Summarize text using extractive methods."""
        if not self._is_loaded:
            self.load_model()
        
        start_time = time.time()
        
        try:
            # Get configuration
            num_sentences = kwargs.get('num_sentences', self.num_sentences)
            
            # Split into sentences
            sentences = self._split_into_sentences(text)
            
            if len(sentences) <= num_sentences:
                # Text is already short enough
                summary = ' '.join(sentences)
                processing_time = time.time() - start_time
                
                return SummaryResult(
                    summary=summary,
                    processing_time=processing_time,
                    metadata={
                        'method': 'extractive',
                        'num_original_sentences': len(sentences),
                        'num_summary_sentences': len(sentences),
                        'model': self.model_name
                    }
                )
            
            # Get sentence embeddings
            embeddings = self._get_sentence_embeddings(sentences)
            
            # Calculate sentence scores
            scores = self._calculate_sentence_scores(embeddings)
            
            # Ensure scores is numpy array
            if not isinstance(scores, np.ndarray):
                scores = np.array(scores)
            
            # Select top sentences
            top_indices = np.argsort(scores)[-num_sentences:]
            top_indices = sorted(top_indices)  # Maintain original order
            
            # Create summary
            summary_sentences = [sentences[i] for i in top_indices]
            summary = ' '.join(summary_sentences)
            
            processing_time = time.time() - start_time
            
            # Prepare metadata
            metadata = {
                'method': 'extractive',
                'model': self.model_name,
                'num_original_sentences': len(sentences),
                'num_summary_sentences': len(summary_sentences),
                'selected_indices': top_indices.tolist(),
                'sentence_scores': scores.tolist(),
                'compression_ratio': len(summary) / len(text) if len(text) > 0 else 0
            }
            
            return SummaryResult(
                summary=summary,
                processing_time=processing_time,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Extractive summarization failed: {e}")
            import traceback
            traceback.print_exc()
            return SummaryResult(
                summary="",
                processing_time=time.time() - start_time,
                metadata={"error": str(e)}
            )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'name': f"Extractive Summarizer ({self.model_name})",
            'type': 'extractive',
            'language': self.language,
            'model_name': self.model_name,
            'device': self.device,
            'num_sentences': self.num_sentences
        }