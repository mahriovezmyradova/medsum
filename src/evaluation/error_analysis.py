# src/evaluation/error_analysis.py
"""
Error analysis for medical ASR transcriptions - FIXED VERSION.
"""
import re
import numpy as np
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass, field
import logging
import jiwer
from collections import Counter

logger = logging.getLogger(__name__)

@dataclass
class MedicalError:
    """Individual medical error."""
    original_word: str
    transcribed_word: str
    error_type: str  # substitution, deletion, insertion
    category: str  # medical_term, medication, dosage, symptom, etc.
    position: int
    confidence: float = 1.0
    
@dataclass
class ErrorAnalysisResult:
    """Results of medical error analysis."""
    # Basic metrics
    wer: float = 0.0
    cer: float = 0.0
    mer: float = 0.0  # Match Error Rate
    wil: float = 0.0  # Word Information Lost
    medical_wer: float = 0.0  # WER for medical terms only
    
    # Error counts
    total_errors: int = 0
    medical_term_errors: int = 0
    medication_errors: int = 0
    dosage_errors: int = 0
    symptom_errors: int = 0
    temporal_errors: int = 0
    
    # Error types
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    
    # Detailed errors
    individual_errors: List[MedicalError] = field(default_factory=list)
    
    # Impact on summarization
    summary_impact_score: float = 0.0
    critical_errors: List[MedicalError] = field(default_factory=list)
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            'wer': self.wer,
            'cer': self.cer,
            'mer': self.mer,
            'wil': self.wil,
            'medical_wer': self.medical_wer,
            'total_errors': self.total_errors,
            'medical_term_errors': self.medical_term_errors,
            'medication_errors': self.medication_errors,
            'dosage_errors': self.dosage_errors,
            'symptom_errors': self.symptom_errors,
            'temporal_errors': self.temporal_errors,
            'substitutions': self.substitutions,
            'deletions': self.deletions,
            'insertions': self.insertions,
            'summary_impact_score': self.summary_impact_score,
            'critical_errors_count': len(self.critical_errors)
        }

class MedicalErrorAnalyzer:
    """
    Analyzes medical-specific errors in ASR transcriptions
    and their impact on summarization.
    """
    
    def __init__(self, language: str = "de"):
        self.language = language
        self._load_medical_resources()
    
    def _load_medical_resources(self):
        """Load German medical vocabulary and patterns."""
        # German medical terms database - expanded for your dataset
        self.medical_terms = {
            'symptom': {
                'schmerzen', 'schmerz', 'weh', 'beschwerden', 'probleme',
                'fieber', 'temperatur', 'kopfschmerzen', 'bauchschmerzen',
                'rückenschmerzen', 'übelkeit', 'erbrechen', 'durchfall',
                'verstopfung', 'müdigkeit', 'erschöpfung', 'schwindel',
                'atemnot', 'husten', 'schnupfen', 'halsschmerzen',
                'schlucken', 'schluckbeschwerden', 'schluckstörung', 'dysphagie',
                'trockenheit', 'trocken', 'entzündung', 'schwellung',
                'röte', 'juckreiz', 'brennen', 'kribbeln', 'konzentrieren',
                'konzentration', 'schwierigkeiten'
            },
            'diagnosis': {
                'diagnose', 'befund', 'erkrankung', 'krankheit', 'zustand',
                'infektion', 'entzündung', 'tumor', 'krebs', 'diabetes',
                'bluthochdruck', 'herzinsuffizienz', 'asthma', 'arthrose',
                'arthritis', 'osteoporose', 'depression', 'angst', 'migräne',
                'allergie', 'grippe', 'erkältung', 'bronchitis', 'pneumonie',
                'gastritis', 'ulcus', 'colitis', 'hepatitis', 'adhs'
            },
            'medication': {
                'medikament', 'tablette', 'pille', 'saft', 'tropfen', 'ibuprofen',
                'injektion', 'spritze', 'antibiotikum', 'schmerzmittel', 'analgetikum',
                'antidepressivum', 'blutdrucksenker', 'insulin', 'cortison',
                'antihistaminikum', 'vitamin', 'mineral', 'salbe', 'creme',
                'gel', 'spray', 'lösung', 'wasser', 'salzwasser'
            },
            'treatment': {
                'therapie', 'behandlung', 'operation', 'eingriff', 'untersuchung',
                'physiotherapie', 'rehabilitation', 'bestrahlung', 'chemotherapie',
                'massage', 'akupunktur', 'training', 'diät', 'ernährung',
                'ruhe', 'schonung', 'gurgeln', 'spülen', 'inhalation',
                'kompresse', 'verbund', 'frühstück', 'pflege', 'besuch'
            },
            'body_part': {
                'kopf', 'hals', 'rachen', 'rachenraum', 'mund', 'zunge', 'zähne',
                'rücken', 'brust', 'bauch', 'magen', 'darm', 'herz', 'lunge',
                'leber', 'niere', 'arme', 'beine', 'hände', 'füße', 'haut',
                'knochen', 'muskeln', 'gelenke', 'gehirn'
            },
            'patient_group': {
                'neugeborenen', 'säuglingen', 'kleinkindern', 'schulkindern',
                'jugendlichen', 'erwachsenen', 'senioren', 'patient', 'patienten',
                'kollegen', 'büro', 'team', 'pflegeteam'
            }
        }
        
        # Medication patterns
        self.medication_patterns = [
            r'\b(\d+)[.,]?\d*\s*(mg|g|ml|µg|IE|Einheiten?|Tabletten?|Pillen?)\b',
            r'\b(\d+)\s*(mal|×)\s*(täglich|pro tag|die woche|am tag)\b',
            r'\b(vor|nach|zu|mit)\s+(dem|der|den)\s+(essen|mahlzeit|nahrung)\b',
            r'\b(mindestens|maximal|etwa|circa|ungefähr)\s+(\d+)\b',
            r'\b(ein|zwei|drei|vier|fünf)\s+(paar|paare)\s+(stunden|tage|wochen)\b'
        ]
        
        # Critical terms (errors in these are more serious)
        self.critical_terms = {
            'nicht', 'kein', 'keine', 'niemals', 'verboten', 'gefährlich',
            'riskant', 'kontraindiziert', 'allergie', 'unverträglich',
            'nebenwirkung', 'überdosierung', 'vergiftung', 'tod', 'sterben'
        }
    
    def analyze_transcription_errors(self, 
                                   reference: str, 
                                   hypothesis: str) -> ErrorAnalysisResult:
        """
        Analyze errors between reference and ASR transcription.
        
        Args:
            reference: Original text
            hypothesis: ASR transcription
            
        Returns:
            ErrorAnalysisResult with detailed analysis
        """
        result = ErrorAnalysisResult()
        
        # Clean texts for comparison
        ref_clean = self._clean_text(reference)
        hyp_clean = self._clean_text(hypothesis)
        
        # Basic WER/CER using jiwer v3.0+ compatible method
        try:
            # For jiwer v3.0+
            transformation = jiwer.Compose([
                jiwer.ToLowerCase(),
                jiwer.RemoveMultipleSpaces(),
                jiwer.Strip(),
                jiwer.RemovePunctuation()
            ])
            
            result.wer = jiwer.wer(ref_clean, hyp_clean, 
                                  truth_transform=transformation,
                                  hypothesis_transform=transformation)
            result.cer = jiwer.cer(ref_clean, hyp_clean,
                                  truth_transform=transformation,
                                  hypothesis_transform=transformation)
            
            # Calculate MER and WIL
            result.mer = jiwer.mer(ref_clean, hyp_clean,
                                  truth_transform=transformation,
                                  hypothesis_transform=transformation)
            result.wil = jiwer.wil(ref_clean, hyp_clean,
                                  truth_transform=transformation,
                                  hypothesis_transform=transformation)
            
        except Exception as e:
            logger.warning(f"Error calculating jiwer metrics: {e}")
            # Fallback to simple calculation
            result.wer = self._simple_wer(ref_clean, hyp_clean)
            result.cer = self._simple_cer(ref_clean, hyp_clean)
        
        # Tokenize
        ref_words = ref_clean.lower().split()
        hyp_words = hyp_clean.lower().split()
        
        # Calculate error counts using custom method
        error_counts = self._count_errors(ref_words, hyp_words)
        result.substitutions = error_counts['substitutions']
        result.deletions = error_counts['deletions']
        result.insertions = error_counts['insertions']
        result.total_errors = sum([result.substitutions, result.deletions, result.insertions])
        
        # Analyze each word
        self._analyze_word_level_errors(ref_words, hyp_words, result)
        
        # Calculate medical-specific WER
        result.medical_wer = self._calculate_medical_wer(ref_clean, hyp_clean)
        
        # Calculate impact on summarization
        result.summary_impact_score = self._calculate_summary_impact(result)
        
        # Identify critical errors
        result.critical_errors = self._identify_critical_errors(result.individual_errors)
        
        return result
    
    def _clean_text(self, text: str) -> str:
        """Clean text for comparison."""
        import re
        # Remove quotes and special characters
        text = re.sub(r'[„"«»"\'`]', '', text)
        # Replace multiple spaces
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _simple_wer(self, reference: str, hypothesis: str) -> float:
        """Simple WER calculation as fallback."""
        ref_words = reference.split()
        hyp_words = hypothesis.split()
        
        # Simple Levenshtein distance for words
        n, m = len(ref_words), len(hyp_words)
        
        # Create matrix
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        
        for i in range(n + 1):
            dp[i][0] = i
        for j in range(m + 1):
            dp[0][j] = j
            
        # Fill matrix
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if ref_words[i-1] == hyp_words[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(
                        dp[i-1][j] + 1,    # deletion
                        dp[i][j-1] + 1,    # insertion
                        dp[i-1][j-1] + 1   # substitution
                    )
        
        errors = dp[n][m]
        return errors / max(n, 1)
    
    def _simple_cer(self, reference: str, hypothesis: str) -> float:
        """Simple CER calculation as fallback."""
        # Character-level comparison
        n, m = len(reference), len(hypothesis)
        
        # Create matrix
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        
        for i in range(n + 1):
            dp[i][0] = i
        for j in range(m + 1):
            dp[0][j] = j
            
        # Fill matrix
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if reference[i-1] == hypothesis[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(
                        dp[i-1][j] + 1,    # deletion
                        dp[i][j-1] + 1,    # insertion
                        dp[i-1][j-1] + 1   # substitution
                    )
        
        errors = dp[n][m]
        return errors / max(n, 1)
    
    def _count_errors(self, ref_words: List[str], hyp_words: List[str]) -> Dict[str, int]:
        """Count substitution, deletion, and insertion errors."""
        # Simple alignment for counting
        n, m = len(ref_words), len(hyp_words)
        
        # Initialize counts
        counts = {'substitutions': 0, 'deletions': 0, 'insertions': 0}
        
        i = j = 0
        while i < n and j < m:
            if ref_words[i] == hyp_words[j]:
                i += 1
                j += 1
            else:
                # Check if it's substitution, deletion, or insertion
                if i + 1 < n and ref_words[i + 1] == hyp_words[j]:
                    counts['deletions'] += 1
                    i += 1
                elif j + 1 < m and ref_words[i] == hyp_words[j + 1]:
                    counts['insertions'] += 1
                    j += 1
                else:
                    counts['substitutions'] += 1
                    i += 1
                    j += 1
        
        # Remaining deletions or insertions
        counts['deletions'] += n - i
        counts['insertions'] += m - j
        
        return counts
    
    def _analyze_word_level_errors(self, ref_words: List[str], 
                                 hyp_words: List[str], 
                                 result: ErrorAnalysisResult):
        """Analyze errors at word level."""
        # Simple word-by-word comparison
        min_len = min(len(ref_words), len(hyp_words))
        
        for i in range(min_len):
            ref_word = ref_words[i]
            hyp_word = hyp_words[i]
            
            if ref_word != hyp_word:
                # Classify error
                error_type = 'substitution'
                category = self._classify_medical_category(ref_word)
                
                error = MedicalError(
                    original_word=ref_word,
                    transcribed_word=hyp_word,
                    error_type=error_type,
                    category=category,
                    position=i
                )
                
                result.individual_errors.append(error)
                
                # Update medical category counters
                if category == 'medication':
                    result.medication_errors += 1
                elif category == 'symptom':
                    result.symptom_errors += 1
                elif category in ['patient_group', 'body_part', 'diagnosis', 'treatment']:
                    result.medical_term_errors += 1
        
        # Handle deletions and insertions
        if len(ref_words) > len(hyp_words):
            for i in range(len(hyp_words), len(ref_words)):
                ref_word = ref_words[i]
                category = self._classify_medical_category(ref_word)
                
                error = MedicalError(
                    original_word=ref_word,
                    transcribed_word="",
                    error_type='deletion',
                    category=category,
                    position=i
                )
                
                result.individual_errors.append(error)
                
                if category in self.medical_terms:
                    result.medical_term_errors += 1
        
        elif len(hyp_words) > len(ref_words):
            for i in range(len(ref_words), len(hyp_words)):
                hyp_word = hyp_words[i]
                
                error = MedicalError(
                    original_word="",
                    transcribed_word=hyp_word,
                    error_type='insertion',
                    category='other',
                    position=i
                )
                
                result.individual_errors.append(error)
    
    def _classify_medical_category(self, word: str) -> str:
        """Classify word into medical category."""
        if not word:
            return 'other'
            
        word_lower = word.lower()
        
        for category, terms in self.medical_terms.items():
            if word_lower in terms:
                return category
        
        # Check medication patterns
        for pattern in self.medication_patterns:
            if re.search(pattern, word_lower):
                return 'medication'
        
        return 'other'
    
    def _calculate_medical_wer(self, reference: str, hypothesis: str) -> float:
        """Calculate WER for medical terms only."""
        ref_words = reference.lower().split()
        hyp_words = hypothesis.lower().split()
        
        medical_ref_words = []
        medical_hyp_words = []
        
        # Extract medical terms from reference
        for word in ref_words:
            if any(word in terms for terms in self.medical_terms.values()):
                medical_ref_words.append(word)
        
        # Extract medical terms from hypothesis
        for word in hyp_words:
            if any(word in terms for terms in self.medical_terms.values()):
                medical_hyp_words.append(word)
        
        if not medical_ref_words:
            return 0.0
        
        # Calculate simple medical WER
        errors = 0
        matched = 0
        
        for ref_word in medical_ref_words:
            if ref_word in medical_hyp_words:
                matched += 1
            else:
                errors += 1
        
        return errors / len(medical_ref_words)
    
    def _calculate_summary_impact(self, result: ErrorAnalysisResult) -> float:
        """
        Calculate how much errors impact summarization quality.
        
        Higher score = more impact on summarization.
        """
        # Weights for different error types
        weights = {
            'medication': 3.0,
            'dosage': 4.0,
            'symptom': 2.0,
            'diagnosis': 3.5,
            'treatment': 2.5,
            'body_part': 1.5,
            'patient_group': 1.0,
            'other': 0.5
        }
        
        impact_score = 0.0
        
        for error in result.individual_errors:
            weight = weights.get(error.category, 0.5)
            
            # Critical terms get extra weight
            if error.original_word.lower() in self.critical_terms:
                weight *= 2.0
            
            impact_score += weight
        
        # Normalize
        total_errors = len(result.individual_errors)
        if total_errors > 0:
            impact_score = impact_score / total_errors
        
        return min(impact_score, 1.0)
    
    def _identify_critical_errors(self, errors: List[MedicalError]) -> List[MedicalError]:
        """Identify critical errors that could seriously affect medical decisions."""
        critical = []
        
        for error in errors:
            # Errors in medication/dosage are critical
            if error.category in ['medication', 'dosage']:
                critical.append(error)
            
            # Errors involving critical terms
            elif error.original_word.lower() in self.critical_terms:
                critical.append(error)
            
            # Medication dosage errors (pattern matching)
            elif self._is_dosage_error(error.original_word, error.transcribed_word):
                critical.append(error)
        
        return critical
    
    def _is_dosage_error(self, word1: str, word2: str) -> bool:
        """Check if it's a dosage-related error."""
        dosage_patterns = [
            r'\b(\d+)[.,]?\d*\s*(mg|g|ml)\b',
            r'\b(\d+)\s*(mal|×)\s*täglich\b'
        ]
        
        for pattern in dosage_patterns:
            match1 = re.search(pattern, word1.lower())
            match2 = re.search(pattern, word2.lower())
            
            if (match1 and not match2) or (not match1 and match2):
                return True
        
        return False
    
    def compare_summaries(self, 
                         original_summary: str, 
                         asr_summary: str) -> Dict[str, float]:
        """
        Compare summaries from original vs ASR text.
        
        Args:
            original_summary: Summary from original text
            asr_summary: Summary from ASR transcription
            
        Returns:
            Dictionary with comparison metrics
        """
        try:
            from ..summarization.evaluation import SummarizationEvaluator
            evaluator = SummarizationEvaluator(language=self.language)
            
            # Basic metrics
            metrics = evaluator.evaluate(original_summary, asr_summary)
            
            return {
                'rouge1': metrics.rouge1,
                'rougeL': metrics.rougeL,
                'bleu': metrics.bleu,
                'semantic_similarity': metrics.semantic_similarity,
                'medical_coverage': metrics.medical_keyword_coverage
            }
        except:
            # Fallback if evaluator fails
            return {
                'rouge1': 0.0,
                'rougeL': 0.0,
                'bleu': 0.0,
                'semantic_similarity': 0.0,
                'medical_coverage': 0.0
            }
    
    def generate_report(self, error_analysis: ErrorAnalysisResult,
                       original_text: str,
                       asr_text: str) -> str:
        """Generate comprehensive error analysis report."""
        
        report = []
        report.append("=" * 80)
        report.append("MEDICAL ASR ERROR ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Basic statistics
        report.append("📊 BASIC STATISTICS")
        report.append("-" * 40)
        report.append(f"Word Error Rate (WER): {error_analysis.wer:.3f}")
        report.append(f"Character Error Rate (CER): {error_analysis.cer:.3f}")
        report.append(f"Match Error Rate (MER): {error_analysis.mer:.3f}")
        report.append(f"Word Information Lost (WIL): {error_analysis.wil:.3f}")
        report.append(f"Medical Term WER: {error_analysis.medical_wer:.3f}")
        report.append(f"Total Errors: {error_analysis.total_errors}")
        report.append("")
        
        # Error breakdown
        report.append("🔍 ERROR BREAKDOWN")
        report.append("-" * 40)
        report.append(f"Substitutions: {error_analysis.substitutions}")
        report.append(f"Deletions: {error_analysis.deletions}")
        report.append(f"Insertions: {error_analysis.insertions}")
        report.append("")
        report.append(f"Medical Term Errors: {error_analysis.medical_term_errors}")
        report.append(f"Medication Errors: {error_analysis.medication_errors}")
        report.append(f"Symptom Errors: {error_analysis.symptom_errors}")
        report.append("")
        
        # Example errors from the actual text
        report.append("📝 EXAMPLE ERRORS IN TEXT")
        report.append("-" * 40)
        
        # Find and highlight differences
        ref_words = original_text.split()
        hyp_words = asr_text.split()
        
        examples = []
        for i in range(min(len(ref_words), len(hyp_words), 15)):  # Check first 15 words
            if ref_words[i].lower() != hyp_words[i].lower():
                examples.append(f"  Word {i+1}: '{ref_words[i]}' → '{hyp_words[i]}'")
        
        if examples:
            for example in examples[:5]:  # Show first 5 examples
                report.append(example)
        else:
            report.append("  No word-level differences found in first 15 words")
        
        report.append("")
        
        # Summary impact
        report.append("📈 SUMMARY IMPACT ASSESSMENT")
        report.append("-" * 40)
        report.append(f"Impact Score: {error_analysis.summary_impact_score:.3f}")
        
        if error_analysis.summary_impact_score > 0.7:
            report.append("  ⚠️  HIGH IMPACT: Errors likely to significantly affect summary quality")
        elif error_analysis.summary_impact_score > 0.4:
            report.append("  ⚠️  MODERATE IMPACT: Some effect on summary quality expected")
        else:
            report.append("  ✅ LOW IMPACT: Errors unlikely to seriously affect summary")
        
        report.append("")
        
        # Critical errors
        if error_analysis.critical_errors:
            report.append("⚠️  CRITICAL ERRORS DETECTED")
            report.append("-" * 40)
            for error in error_analysis.critical_errors[:3]:  # Show top 3
                report.append(f"  • '{error.original_word}' → '{error.transcribed_word}' "
                            f"({error.category})")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)