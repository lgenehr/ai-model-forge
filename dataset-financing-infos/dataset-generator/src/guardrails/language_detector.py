"""
Language detection for filtering non-target languages.
"""

from dataclasses import dataclass
from typing import Any

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class LanguageResult:
    """Result of language detection."""

    language: str
    confidence: float
    is_target: bool
    all_languages: list[tuple[str, float]]


class LanguageDetector:
    """
    Detects language of text using langdetect.
    """

    # Language code mappings
    LANGUAGE_NAMES = {
        "pt": "Portuguese",
        "en": "English",
        "es": "Spanish",
        "fr": "French",
        "de": "German",
        "it": "Italian",
        "nl": "Dutch",
        "ru": "Russian",
        "ja": "Japanese",
        "zh-cn": "Chinese (Simplified)",
        "zh-tw": "Chinese (Traditional)",
        "ko": "Korean",
        "ar": "Arabic",
    }

    def __init__(
        self,
        target_languages: list[str] | None = None,
        min_confidence: float = 0.95,
    ) -> None:
        """
        Initialize language detector.

        Args:
            target_languages: List of target language codes (e.g., ['pt', 'en'])
            min_confidence: Minimum confidence for detection
        """
        self.target_languages = target_languages or ["pt"]
        self.min_confidence = min_confidence
        self._detector = None

    def _get_detector(self):
        """Lazy load language detector."""
        if self._detector is None:
            try:
                from langdetect import DetectorFactory, detect_langs

                # Make detection deterministic
                DetectorFactory.seed = 0
                self._detector = detect_langs
            except ImportError:
                logger.warning(
                    "langdetect not available, using fallback detection"
                )
                self._detector = self._fallback_detect
        return self._detector

    def _fallback_detect(self, text: str) -> list:
        """Fallback detection using simple heuristics."""
        # Simple heuristic based on common words
        text_lower = text.lower()

        # Portuguese markers
        pt_words = ["que", "não", "para", "com", "uma", "são", "está", "isso"]
        pt_count = sum(1 for w in pt_words if f" {w} " in f" {text_lower} ")

        # English markers
        en_words = ["the", "and", "that", "for", "are", "with", "this", "from"]
        en_count = sum(1 for w in en_words if f" {w} " in f" {text_lower} ")

        # Spanish markers
        es_words = ["que", "para", "con", "una", "por", "pero", "como", "esta"]
        es_count = sum(1 for w in es_words if f" {w} " in f" {text_lower} ")

        # Create pseudo results
        class FakeResult:
            def __init__(self, lang: str, prob: float):
                self.lang = lang
                self.prob = prob

        results = []
        total = pt_count + en_count + es_count + 1

        if pt_count > 0:
            results.append(FakeResult("pt", pt_count / total))
        if en_count > 0:
            results.append(FakeResult("en", en_count / total))
        if es_count > 0:
            results.append(FakeResult("es", es_count / total))

        if not results:
            results.append(FakeResult("pt", 0.5))  # Default to Portuguese

        results.sort(key=lambda x: x.prob, reverse=True)
        return results

    def detect(self, text: str) -> LanguageResult:
        """
        Detect language of text.

        Args:
            text: Text to analyze

        Returns:
            LanguageResult with detection
        """
        if not text or len(text.strip()) < 10:
            return LanguageResult(
                language="unknown",
                confidence=0.0,
                is_target=False,
                all_languages=[],
            )

        try:
            detector = self._get_detector()
            results = detector(text)

            all_languages = [(r.lang, r.prob) for r in results]

            if results:
                top_lang = results[0].lang
                top_conf = results[0].prob

                # Check if it's a target language
                is_target = (
                    top_lang in self.target_languages
                    and top_conf >= self.min_confidence
                )

                return LanguageResult(
                    language=top_lang,
                    confidence=top_conf,
                    is_target=is_target,
                    all_languages=all_languages,
                )

        except Exception as e:
            logger.warning("Language detection failed", error=str(e))

        return LanguageResult(
            language="unknown",
            confidence=0.0,
            is_target=False,
            all_languages=[],
        )

    def is_target_language(self, text: str) -> bool:
        """Quick check if text is in target language."""
        return self.detect(text).is_target

    def filter_by_language(
        self,
        texts: list[str],
        return_non_target: bool = False,
    ) -> tuple[list[str], list[str]]:
        """
        Filter texts by language.

        Args:
            texts: List of texts
            return_non_target: Whether to return non-target texts

        Returns:
            Tuple of (target_language_texts, non_target_texts)
        """
        target = []
        non_target = []

        for text in texts:
            if self.is_target_language(text):
                target.append(text)
            else:
                non_target.append(text)

        return target, non_target

    def get_stats(self, texts: list[str]) -> dict[str, Any]:
        """Get language detection statistics."""
        language_counts: dict[str, int] = {}
        target_count = 0
        non_target_count = 0
        low_confidence = 0

        for text in texts:
            result = self.detect(text)

            lang = result.language
            language_counts[lang] = language_counts.get(lang, 0) + 1

            if result.is_target:
                target_count += 1
            else:
                non_target_count += 1

            if result.confidence < self.min_confidence:
                low_confidence += 1

        return {
            "total": len(texts),
            "target_language": target_count,
            "non_target_language": non_target_count,
            "target_rate": target_count / max(1, len(texts)),
            "low_confidence": low_confidence,
            "language_breakdown": language_counts,
            "target_languages": self.target_languages,
            "min_confidence": self.min_confidence,
        }
