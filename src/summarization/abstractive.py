# src/summarization/abstractive.py
"""
Abstractive summarization using LLMs.
"""
import openai
from typing import List, Dict, Any, Optional
import logging
import time
import os

from .base import BaseSummarizer, SummaryResult

logger = logging.getLogger(__name__)

class AbstractiveSummarizer(BaseSummarizer):
    """Abstractive summarization using LLMs."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.model_type = config.get("model_type", "openai")  # openai, huggingface, local
        self.model_name = config.get("model_name", "gpt-3.5-turbo")
        self.max_length = config.get("max_length", 150)
        self.temperature = config.get("temperature", 0.3)
        
        # OpenAI-specific
        self.api_key = config.get("api_key", os.environ.get("OPENAI_API_KEY"))
        
        # HuggingFace-specific
        self.hf_model = None
        self.hf_tokenizer = None
        
        logger.info(f"Initialized abstractive summarizer: {self.model_type}/{self.model_name}")
    
    def load_model(self):
        """Load abstractive summarization model."""
        if self._is_loaded:
            return
        
        logger.info(f"Loading abstractive model: {self.model_type}/{self.model_name}")
        
        try:
            if self.model_type == "openai":
                if not self.api_key:
                    raise ValueError("OpenAI API key not provided")
                openai.api_key = self.api_key
                
            elif self.model_type == "huggingface":
                from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
                
                self.hf_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.hf_model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
                
                # Move to GPU if available
                import torch
                if torch.cuda.is_available():
                    self.hf_model = self.hf_model.cuda()
            
            elif self.model_type == "local":
                # Load local model (e.g., from disk)
                # Implementation depends on your local setup
                pass
            
            self._is_loaded = True
            logger.info(f"Abstractive model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load abstractive model: {e}")
            raise
    
    def _summarize_openai(self, text: str, **kwargs) -> SummaryResult:
        """Summarize using OpenAI API."""
        start_time = time.time()
        
        try:
            # Medical-specific prompt for German
            prompt = f"""Zusammenfassung des Arzt-Patient-Gesprächs auf Deutsch:

Gespräch: "{text}"

Erstelle eine professionelle medizinische Zusammenfassung in Deutsch mit folgenden Abschnitten:
1. Hauptbeschwerden des Patienten
2. Befunde und Diagnosen
3. Empfohlene Behandlungen
4. Weitere Empfehlungen

Zusammenfassung:"""
            
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "Du bist ein medizinischer Experte, der Arzt-Patient-Gespräche zusammenfasst."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_length,
                temperature=self.temperature,
                n=1
            )
            
            summary = response.choices[0].message.content.strip()
            processing_time = time.time() - start_time
            
            return SummaryResult(
                summary=summary,
                processing_time=processing_time,
                metadata={
                    'method': 'abstractive',
                    'model': self.model_name,
                    'provider': 'openai',
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens
                }
            )
            
        except Exception as e:
            logger.error(f"OpenAI summarization failed: {e}")
            return SummaryResult(
                summary="",
                processing_time=time.time() - start_time,
                metadata={"error": str(e)}
            )
    
    def _summarize_huggingface(self, text: str, **kwargs) -> SummaryResult:
        """Summarize using HuggingFace models."""
        start_time = time.time()
        
        try:
            import torch
            
            # Tokenize input
            inputs = self.hf_tokenizer(
                text, 
                max_length=512, 
                truncation=True, 
                return_tensors="pt"
            )
            
            # Move to device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate summary
            with torch.no_grad():
                summary_ids = self.hf_model.generate(
                    **inputs,
                    max_length=self.max_length,
                    min_length=30,
                    length_penalty=2.0,
                    num_beams=4,
                    early_stopping=True
                )
            
            # Decode
            summary = self.hf_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            
            processing_time = time.time() - start_time
            
            return SummaryResult(
                summary=summary,
                processing_time=processing_time,
                metadata={
                    'method': 'abstractive',
                    'model': self.model_name,
                    'provider': 'huggingface'
                }
            )
            
        except Exception as e:
            logger.error(f"HuggingFace summarization failed: {e}")
            return SummaryResult(
                summary="",
                processing_time=time.time() - start_time,
                metadata={"error": str(e)}
            )
    
    def summarize(self, text: str, **kwargs) -> SummaryResult:
        """Summarize text using abstractive methods."""
        if not self._is_loaded:
            self.load_model()
        
        # Override config with kwargs
        max_length = kwargs.get('max_length', self.max_length)
        temperature = kwargs.get('temperature', self.temperature)
        
        if self.model_type == "openai":
            return self._summarize_openai(text, max_length=max_length, temperature=temperature)
        
        elif self.model_type == "huggingface":
            return self._summarize_huggingface(text, max_length=max_length)
        
        else:
            # Fallback or custom implementation
            return SummaryResult(
                summary="Abstractive summarization not implemented for this model type",
                processing_time=0.0,
                metadata={"error": "Not implemented"}
            )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'name': f"Abstractive Summarizer ({self.model_type}/{self.model_name})",
            'type': 'abstractive',
            'language': self.language,
            'model_type': self.model_type,
            'model_name': self.model_name,
            'max_length': self.max_length
        }