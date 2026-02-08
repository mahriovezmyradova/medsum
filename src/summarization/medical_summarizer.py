# src/summarization/medical_summarizer.py
"""
Medical-specific summarization for German doctor-patient conversations.
"""
import re
from typing import List, Dict, Any, Optional
import logging
from collections import defaultdict

from .base import BaseSummarizer, SummaryResult
from .extractive import ExtractiveSummarizer

logger = logging.getLogger(__name__)

class MedicalSummarizer(BaseSummarizer):
    """
    Medical-specific summarizer that extracts key medical information.
    Combines extractive and rule-based approaches.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Medical categories to extract
        self.categories = config.get("categories", [
            "symptoms", "diagnosis", "medication", "treatment", 
            "recommendations", "follow_up"
        ])
        
        # Medical vocabulary for German
        self._init_medical_vocabulary()
        
        # Optional: Use extractive summarizer as backbone
        self.use_extractive = config.get("use_extractive", True)
        if self.use_extractive:
            self.extractive_summarizer = ExtractiveSummarizer(config)
    
    # Update the medical summarizer to handle German medical terms better:

# In src/summarization/medical_summarizer.py, update the _init_medical_vocabulary method:

    def _init_medical_vocabulary(self):
        """Initialize German medical vocabulary."""
        self.medical_keywords = {
            "symptoms": {
                'schmerzen', 'schmerz', 'weh', 'beschwerden', 'probleme',
                'schwierigkeiten', 'schwer', 'trocken', 'trockenheit',
                'schlucken', 'schluckbeschwerden', 'schluckstörung',
                'mund', 'rachen', 'rachenraum', 'halsschmerzen',
                'entzündung', 'schwellung', 'röte', 'juckreiz'
            },
            "diagnosis": {
                'diagnose', 'befund', 'erkrankung', 'krankheit',
                'infektion', 'entzündung', 'trockenheit',
                'schluckstörung', 'dysphagie', 'pharyngitis'
            },
            "medication": {
                'medikament', 'tablette', 'pille', 'saft', 'tropfen',
                'salzwasser', 'salz', 'wasser', 'gurgellösung',
                'spülung', 'lösung', 'spray'
            },
            "treatment": {
                'therapie', 'behandlung', 'gurgeln', 'spülen',
                'anwendung', 'durchführung', 'wiederholung',
                'täglich', 'mehrmals', 'viermal', 'regelmäßig'
            }
        }
        
        # Medical patterns (regex)
        self.medical_patterns = {
            "frequency": r'\b(mindestens\s+)?(\d+)\s*(mal|×)\s*(täglich|pro tag|am tag|die woche)\b',
            "duration": r'\bfür\s+(\d+)\s*(Tage?|Wochen?|Monate?|Jahre?)\b',
            "instruction": r'\b(gurgeln|spülen|anwenden|einnehmen)\s+(sie\s+)?bitte\b',
            "temperature": r'\b(warmem|warmen|kaltem|kalten)\b'
        }
    
    def load_model(self):
        """Load medical summarizer."""
        if self._is_loaded:
            return
        
        logger.info("Loading medical summarizer")
        
        try:
            if self.use_extractive:
                self.extractive_summarizer.load_model()
            
            self._is_loaded = True
            logger.info("Medical summarizer loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load medical summarizer: {e}")
            raise
    
    def _extract_medical_sections(self, text: str) -> Dict[str, List[str]]:
        """Extract medical information by category."""
        sections = {category: [] for category in self.categories}
        text_lower = text.lower()
        
        # Extract by keywords
        for category, keywords in self.medical_keywords.items():
            if category in self.categories:
                for keyword in keywords:
                    if keyword in text_lower:
                        # Find sentences containing the keyword
                        sentences = re.split(r'[.!?]', text)
                        for sentence in sentences:
                            if keyword in sentence.lower() and sentence.strip():
                                sections[category].append(sentence.strip())
        
        # Extract by patterns
        for pattern_name, pattern in self.medical_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                sections["medication"].extend([str(match) for match in matches])
        
        # Remove duplicates while preserving order
        for category in sections:
            seen = set()
            unique_sentences = []
            for sentence in sections[category]:
                if sentence not in seen:
                    seen.add(sentence)
                    unique_sentences.append(sentence)
            sections[category] = unique_sentences
        
        return sections
    
    def summarize(self, text: str, **kwargs) -> SummaryResult:
        """Summarize medical conversation."""
        if not self._is_loaded:
            self.load_model()
        
        import time
        start_time = time.time()
        
        try:
            # Extract medical sections
            sections = self._extract_medical_sections(text)
            
            # Create structured summary
            summary_parts = []
            
            for category in self.categories:
                if sections[category]:
                    # Format section header
                    if category == "symptoms":
                        header = "HAUPTSYMPTOME:"
                    elif category == "diagnosis":
                        header = "DIAGNOSE:"
                    elif category == "medication":
                        header = "MEDIKATION:"
                    elif category == "treatment":
                        header = "BEHANDLUNG:"
                    elif category == "recommendations":
                        header = "EMPFERLUNGEN:"
                    elif category == "follow_up":
                        header = "NÄCHSTE TERMINE:"
                    else:
                        header = f"{category.upper()}:"
                    
                    summary_parts.append(header)
                    
                    # Add content (limit to 3 items per category)
                    for i, sentence in enumerate(sections[category][:3]):
                        summary_parts.append(f"  - {sentence}")
                    
                    summary_parts.append("")  # Empty line between sections
            
            # Join summary
            structured_summary = "\n".join(summary_parts).strip()
            
            # If structured summary is too short, use extractive summarizer as fallback
            if len(structured_summary.split()) < 20 and self.use_extractive:
                extractive_result = self.extractive_summarizer.summarize(text)
                fallback_summary = f"Zusammenfassung:\n{extractive_result.summary}"
                
                # Combine both
                if structured_summary:
                    final_summary = f"{structured_summary}\n\n{fallback_summary}"
                else:
                    final_summary = fallback_summary
            else:
                final_summary = structured_summary
            
            processing_time = time.time() - start_time
            
            # Prepare metadata
            metadata = {
                'method': 'medical_structured',
                'categories_extracted': [c for c in self.categories if sections[c]],
                'num_sections': len([c for c in self.categories if sections[c]]),
                'total_items': sum(len(items) for items in sections.values()),
                'use_extractive_fallback': len(structured_summary.split()) < 20
            }
            
            return SummaryResult(
                summary=final_summary,
                processing_time=processing_time,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Medical summarization failed: {e}")
            return SummaryResult(
                summary="",
                processing_time=time.time() - start_time,
                metadata={"error": str(e)}
            )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'name': "Medical Summarizer",
            'type': 'medical_structured',
            'language': self.language,
            'categories': self.categories,
            'use_extractive': self.use_extractive
        }