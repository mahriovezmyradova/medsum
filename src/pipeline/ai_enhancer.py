"""
AI Enhancement step: GPT cleans up raw ASR output before summarization.

Why this step exists
--------------------
ASR models produce noisy German text – mis-heard words, missing punctuation,
broken compound words, hallucinated filler syllables.  Passing this noise
directly to a summarizer compounds errors.  A single GPT call can:

  - restore correct German medical terminology (e.g. "hyper toni" → "Hypertonie")
  - add sentence-ending punctuation
  - remove ASR artefacts ("[NOISE]", repeated words, filler sounds)
  - keep the content identical – it must NOT add medical information

The class is intentionally thin so that it can be:
  - disabled with a flag (enhancement_enabled=False) to measure its delta
  - swapped for a local model without changing call sites

Usage
-----
    enhancer = AIEnhancer()
    result   = enhancer.enhance(raw_asr_text)
    improved = result.enhanced_text
    delta    = result.improvement_score   # 0-1, heuristic
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# System prompt
# ──────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """Du bist ein medizinischer Textkorrekturassistent für deutsche Arzt-Patient-Gespräche.

Deine Aufgabe:
1. Korrigiere Fehler aus der automatischen Spracherkennung (ASR): falsch erkannte Wörter, fehlende Satzzeichen, zusammengesetzte Wörter die getrennt wurden.
2. Stelle korrekte medizinische Fachbegriffe wieder her (z.B. "Hyper toni" → "Hypertonie").
3. Entferne ASR-Artefakte wie "[NOISE]", "[inaudible]", Wiederholungen von Silben.
4. Füge fehlende Satzzeichen hinzu damit der Text lesbar ist.

Wichtige Regeln:
- Ändere KEINE medizinischen Inhalte – füge keine eigenen Diagnosen oder Medikamente hinzu.
- Behalte die originale Bedeutung vollständig bei.
- Antworte NUR mit dem korrigierten Text – kein Kommentar, keine Erklärung."""

# ──────────────────────────────────────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class EnhancementResult:
    original_text: str
    enhanced_text: str
    model_used: str
    processing_time: float
    improvement_score: float   # heuristic 0–1
    tokens_used: int = 0
    error: Optional[str] = None

    @property
    def was_enhanced(self) -> bool:
        return self.enhanced_text != self.original_text and not self.error


# ──────────────────────────────────────────────────────────────────────────────
# Enhancer
# ──────────────────────────────────────────────────────────────────────────────

class AIEnhancer:
    """
    Wraps an OpenAI chat-completion call that cleans German ASR output.

    Parameters
    ----------
    model : str
        OpenAI model name.  "gpt-4o-mini" is recommended – it is fast, cheap,
        and accurate enough for medical text correction.
    api_key : str | None
        If None, reads OPENAI_API_KEY from environment.
    enabled : bool
        When False, enhance() returns the input unchanged (useful for A/B
        comparison in evaluation runs).
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        enabled: bool = True,
    ):
        self.model = model
        self.enabled = enabled
        self._client = None

        if enabled:
            self._init_client(api_key)

    # ── initialisation ────────────────────────────────────────────────────────

    def _init_client(self, api_key: Optional[str]):
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            logger.warning(
                "OPENAI_API_KEY not set – AI enhancement disabled. "
                "Set the env variable to enable the GPT correction step."
            )
            self.enabled = False
            return
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=key)
            logger.info("AIEnhancer ready with model %s", self.model)
        except ImportError:
            logger.warning("openai package not installed – AI enhancement disabled.")
            self.enabled = False

    # ── public interface ──────────────────────────────────────────────────────

    def enhance(self, text: str) -> EnhancementResult:
        """
        Clean *text* (raw ASR output) and return an EnhancementResult.

        Falls back to returning the original text when:
          - the enhancer is disabled
          - the OpenAI API call fails
        """
        t0 = time.time()

        if not self.enabled or self._client is None or not text.strip():
            return EnhancementResult(
                original_text=text,
                enhanced_text=text,
                model_used="none",
                processing_time=time.time() - t0,
                improvement_score=0.0,
            )

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                temperature=0.1,      # low temperature = deterministic, conservative
                max_tokens=min(len(text.split()) * 3, 2048),
            )
            enhanced = response.choices[0].message.content.strip()
            tokens = response.usage.total_tokens if response.usage else 0

            score = self._improvement_score(text, enhanced)
            logger.info(
                "AIEnhancer: %d → %d words, improvement_score=%.3f, tokens=%d",
                len(text.split()), len(enhanced.split()), score, tokens,
            )

            return EnhancementResult(
                original_text=text,
                enhanced_text=enhanced,
                model_used=self.model,
                processing_time=time.time() - t0,
                improvement_score=score,
                tokens_used=tokens,
            )

        except Exception as exc:
            logger.error("AIEnhancer call failed: %s", exc)
            return EnhancementResult(
                original_text=text,
                enhanced_text=text,
                model_used=self.model,
                processing_time=time.time() - t0,
                improvement_score=0.0,
                error=str(exc),
            )

    # ── heuristic improvement score ───────────────────────────────────────────

    @staticmethod
    def _improvement_score(original: str, enhanced: str) -> float:
        """
        Heuristic 0–1 score estimating how much the enhancement helped.

        Higher is better.  Combines:
          - punctuation density (more punctuation → more readable)
          - capitalisation ratio (proper nouns / sentence starts)
          - reduction in all-caps tokens (ASR artefacts)
        """
        if not enhanced or original == enhanced:
            return 0.0

        orig_words = original.split()
        enh_words = enhanced.split()
        if not orig_words:
            return 0.0

        # Punctuation density
        orig_punct = sum(c in ".!?,;" for c in original) / max(len(original), 1)
        enh_punct = sum(c in ".!?,;" for c in enhanced) / max(len(enhanced), 1)
        punct_gain = max(0.0, enh_punct - orig_punct)

        # Capitalisation (German: most nouns are capitalised)
        orig_cap = sum(w[0].isupper() for w in orig_words if w) / len(orig_words)
        enh_cap = sum(w[0].isupper() for w in enh_words if w) / len(enh_words)
        cap_gain = max(0.0, enh_cap - orig_cap)

        # All-caps artefact removal
        orig_allcaps = sum(w.isupper() and len(w) > 1 for w in orig_words) / len(orig_words)
        enh_allcaps = sum(w.isupper() and len(w) > 1 for w in enh_words) / len(enh_words)
        artefact_reduction = max(0.0, orig_allcaps - enh_allcaps)

        score = punct_gain * 2.0 + cap_gain * 1.5 + artefact_reduction * 1.0
        return min(1.0, score)
