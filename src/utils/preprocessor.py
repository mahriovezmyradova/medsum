"""
Text preprocessing utilities for German medical text.
"""

import re
import unicodedata
from typing import List, Optional, Callable
import spacy
from loguru import logger

class TextPreprocessor:
    """Preprocess German medical text for ASR and summarization."""
    
    def __init__(self, language: str = "de"):
        """
        Initialize text preprocessor.
        
        Args:
            language: Language code ('de' for German)
        """
        self.language = language
        
        # Try to load spaCy model
        self.nlp = None
        try:
            if language == "de":
                self.nlp = spacy.load("de_core_news_sm")
            logger.info(f"Loaded spaCy model for {language}")
        except:
            logger.warning(f"Could not load spaCy model for {language}")
            self.nlp = None
    
    def clean_text(self, text: str, remove_numbers: bool = False) -> str:
        """
        Clean text by removing unnecessary characters and normalizing.
        
        Args:
            text: Input text
            remove_numbers: Whether to remove numbers
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            text = str(text)
        
        # Normalize unicode
        text = unicodedata.normalize('NFKC', text)
        
        # Replace common German quotes and special characters
        replacements = {
            '„': '"',
            '“': '"',
            '”': '"',
            '«': '"',
            '»': '"',
            '´': "'",
            '`': "'",
            '’': "'",
            '‘': "'",
            '–': '-',  # en dash
            '—': '-',  # em dash
            '…': '...',
            '\xa0': ' ',  # non-breaking space
            '\u2028': ' ',  # line separator
            '\u2029': ' ',  # paragraph separator
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove numbers if requested
        if remove_numbers:
            text = re.sub(r'\b\d+\b', '', text)
        
        # Trim
        text = text.strip()
        
        return text
    
    def normalize_medical_text(self, text: str) -> str:
        """
        Normalize medical text for consistency.
        
        Args:
            text: Input medical text
            
        Returns:
            Normalized text
        """
        text = self.clean_text(text)
        
        # Standardize common medical abbreviations
        medical_replacements = {
            r'\bPat\.\b': 'Patient',
            r'\bPät\.\b': 'Patient',
            r'\bpat\.\b': 'patient',
            r'\bDr\.\b': 'Doktor',
            r'\bProf\.\b': 'Professor',
            r'\bMed\.\b': 'Medizin',
            r'\busw\.\b': 'und so weiter',
            r'\bz\.B\.\b': 'zum Beispiel',
            r'\bbzw\.\b': 'beziehungsweise',
            r'\bd\.h\.\b': 'das heißt',
            r'\bca\.\b': 'circa',
        }
        
        for pattern, replacement in medical_replacements.items():
            text = re.sub(pattern, replacement, text)
        
        # Ensure proper sentence spacing
        text = re.sub(r'\.([A-ZÄÖÜ])', r'. \1', text)
        
        return text
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        text = self.clean_text(text)
        
        if self.nlp is not None:
            # Use spaCy for sentence segmentation
            doc = self.nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents]
            sentences = [s for s in sentences if s]  # Remove empty
        else:
            # Fallback: simple regex-based sentence splitting for German
            sentences = re.split(r'(?<=[.!?])\s+(?=[A-ZÄÖÜ])', text)
            sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def extract_medical_keywords(self, text: str, keywords: Optional[List[str]] = None) -> List[str]:
        """
        Extract medical keywords from text.
        
        Args:
            text: Input text
            keywords: Optional list of keywords to look for
            
        Returns:
            List of found keywords
        """
        if keywords is None:
            # Common German medical keywords
            keywords = [
                'patient', 'arzt', 'krankheit', 'medikament', 'behandlung',
                'symptom', 'diagnose', 'therapie', 'untersuchung', 'klinik',
                'krankenhaus', 'schmerz', 'operation', 'blut', 'temperatur',
                'druck', 'herz', 'lunge', 'magen', 'kopf', 'rücken',
                'infektion', 'entzündung', 'allergie', 'impfung', 'rezept'
            ]
        
        text_lower = text.lower()
        found_keywords = []
        
        for keyword in keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)
        
        return found_keywords
    
    def calculate_readability_metrics(self, text: str) -> dict:
        """
        Calculate basic readability metrics for German text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with readability metrics
        """
        text = self.clean_text(text)
        
        # Split into words and sentences
        words = text.split()
        sentences = self.split_into_sentences(text)
        
        if not words or not sentences:
            return {
                'word_count': 0,
                'sentence_count': 0,
                'avg_sentence_length': 0,
                'avg_word_length': 0
            }
        
        # Calculate metrics
        word_count = len(words)
        sentence_count = len(sentences)
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        
        # Average word length (in characters)
        char_count = sum(len(word) for word in words)
        avg_word_length = char_count / word_count if word_count > 0 else 0
        
        return {
            'word_count': word_count,
            'sentence_count': sentence_count,
            'avg_sentence_length': avg_sentence_length,
            'avg_word_length': avg_word_length
        }

# Example usage
if __name__ == "__main__":
    preprocessor = TextPreprocessor(language="de")
    
    test_text = "Patient Müller hat Symptome wie Kopfschmerzen und Fieber. Dr. Schmidt empfiehlt eine Untersuchung."
    
    print("Original text:", test_text)
    print("Cleaned text:", preprocessor.clean_text(test_text))
    print("Normalized medical text:", preprocessor.normalize_medical_text(test_text))
    print("Sentences:", preprocessor.split_into_sentences(test_text))
    print("Medical keywords:", preprocessor.extract_medical_keywords(test_text))
    print("Readability metrics:", preprocessor.calculate_readability_metrics(test_text))