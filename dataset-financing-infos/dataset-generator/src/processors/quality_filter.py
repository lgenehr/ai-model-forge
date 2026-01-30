"""
Quality filtering for dataset entries.
"""

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from ..config.settings import QualitySettings
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class QualityScore:
    """Quality score breakdown."""

    total_score: float
    length_score: float
    special_char_score: float
    word_repetition_score: float
    sentence_score: float
    passed: bool
    filter_reasons: list[str]


class QualityFilter:
    """
    Quality filter for text data.
    """

    def __init__(
        self,
        min_tokens: int = 50,
        max_tokens: int = 4096,
        min_quality_score: float = 0.6,
        max_special_char_ratio: float = 0.1,
        max_word_repetition: float = 0.3,
        min_sentence_length: int = 3,
        min_avg_word_length: float = 2.0,
        max_avg_word_length: float = 15.0,
    ) -> None:
        """
        Initialize quality filter.

        Args:
            min_tokens: Minimum token count
            max_tokens: Maximum token count
            min_quality_score: Minimum quality score to pass
            max_special_char_ratio: Maximum ratio of special characters
            max_word_repetition: Maximum word repetition ratio
            min_sentence_length: Minimum average sentence length (words)
            min_avg_word_length: Minimum average word length
            max_avg_word_length: Maximum average word length
        """
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.min_quality_score = min_quality_score
        self.max_special_char_ratio = max_special_char_ratio
        self.max_word_repetition = max_word_repetition
        self.min_sentence_length = min_sentence_length
        self.min_avg_word_length = min_avg_word_length
        self.max_avg_word_length = max_avg_word_length

    @classmethod
    def from_settings(cls, settings: QualitySettings) -> "QualityFilter":
        """Create filter from settings."""
        return cls(
            min_tokens=settings.min_tokens,
            max_tokens=settings.max_tokens,
            min_quality_score=settings.min_quality_score,
            max_special_char_ratio=settings.max_special_char_ratio,
            max_word_repetition=settings.max_word_repetition,
        )

    def _count_tokens_approx(self, text: str) -> int:
        """
        Approximate token count.
        For accurate counting, use tiktoken.
        """
        # Simple approximation: ~4 chars per token for Portuguese
        return len(text) // 4

    def _get_words(self, text: str) -> list[str]:
        """Extract words from text."""
        return re.findall(r"\b\w+\b", text.lower())

    def _get_sentences(self, text: str) -> list[str]:
        """Extract sentences from text."""
        sentences = re.split(r"[.!?]+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _calculate_special_char_ratio(self, text: str) -> float:
        """Calculate ratio of special characters."""
        if not text:
            return 0.0

        special = sum(
            1
            for c in text
            if not c.isalnum() and not c.isspace() and c not in ".,!?;:'\"-"
        )
        return special / len(text)

    def _calculate_word_repetition(self, words: list[str]) -> float:
        """Calculate word repetition ratio."""
        if not words:
            return 0.0

        word_counts = Counter(words)
        if len(word_counts) == 0:
            return 0.0

        # Ratio of repeated words (appearing more than once)
        repeated = sum(1 for count in word_counts.values() if count > 1)
        return repeated / len(word_counts)

    def _calculate_length_score(self, token_count: int) -> float:
        """Score based on length."""
        if token_count < self.min_tokens:
            return token_count / self.min_tokens
        if token_count > self.max_tokens:
            return max(0, 1 - (token_count - self.max_tokens) / self.max_tokens)
        return 1.0

    def _calculate_sentence_score(self, sentences: list[str]) -> float:
        """Score based on sentence structure."""
        if not sentences:
            return 0.0

        # Average words per sentence
        words_per_sentence = []
        for sentence in sentences:
            words = self._get_words(sentence)
            if words:
                words_per_sentence.append(len(words))

        if not words_per_sentence:
            return 0.0

        avg_sentence_length = sum(words_per_sentence) / len(words_per_sentence)

        # Penalize very short or very long sentences
        if avg_sentence_length < self.min_sentence_length:
            return avg_sentence_length / self.min_sentence_length
        if avg_sentence_length > 50:
            return max(0.5, 1 - (avg_sentence_length - 50) / 100)
        return 1.0

    def calculate_quality_score(self, text: str) -> QualityScore:
        """
        Calculate comprehensive quality score.

        Args:
            text: Text to score

        Returns:
            QualityScore with breakdown
        """
        filter_reasons = []

        # Basic checks
        if not text or not text.strip():
            return QualityScore(
                total_score=0.0,
                length_score=0.0,
                special_char_score=0.0,
                word_repetition_score=0.0,
                sentence_score=0.0,
                passed=False,
                filter_reasons=["empty_text"],
            )

        # Token count
        token_count = self._count_tokens_approx(text)
        length_score = self._calculate_length_score(token_count)

        if token_count < self.min_tokens:
            filter_reasons.append("too_short")
        elif token_count > self.max_tokens:
            filter_reasons.append("too_long")

        # Special characters
        special_ratio = self._calculate_special_char_ratio(text)
        special_char_score = max(0, 1 - special_ratio / self.max_special_char_ratio)

        if special_ratio > self.max_special_char_ratio:
            filter_reasons.append("too_many_special_chars")

        # Word analysis
        words = self._get_words(text)

        # Word repetition
        word_rep = self._calculate_word_repetition(words)
        word_repetition_score = max(0, 1 - word_rep / self.max_word_repetition)

        if word_rep > self.max_word_repetition:
            filter_reasons.append("high_word_repetition")

        # Average word length
        if words:
            avg_word_len = sum(len(w) for w in words) / len(words)
            if avg_word_len < self.min_avg_word_length:
                filter_reasons.append("words_too_short")
                word_repetition_score *= 0.8
            elif avg_word_len > self.max_avg_word_length:
                filter_reasons.append("words_too_long")
                word_repetition_score *= 0.8

        # Sentence analysis
        sentences = self._get_sentences(text)
        sentence_score = self._calculate_sentence_score(sentences)

        if sentence_score < 0.5:
            filter_reasons.append("poor_sentence_structure")

        # Calculate total score
        total_score = (
            length_score * 0.3
            + special_char_score * 0.2
            + word_repetition_score * 0.25
            + sentence_score * 0.25
        )

        passed = total_score >= self.min_quality_score and not filter_reasons

        return QualityScore(
            total_score=round(total_score, 4),
            length_score=round(length_score, 4),
            special_char_score=round(special_char_score, 4),
            word_repetition_score=round(word_repetition_score, 4),
            sentence_score=round(sentence_score, 4),
            passed=passed,
            filter_reasons=filter_reasons,
        )

    def filter(self, text: str) -> tuple[bool, QualityScore]:
        """
        Filter text based on quality.

        Args:
            text: Text to filter

        Returns:
            Tuple of (passed, quality_score)
        """
        score = self.calculate_quality_score(text)
        return score.passed, score

    def get_filter_stats(self, texts: list[str]) -> dict[str, Any]:
        """
        Get filtering statistics for a batch.

        Args:
            texts: List of texts to analyze

        Returns:
            Statistics dictionary
        """
        passed = 0
        failed = 0
        reason_counts: dict[str, int] = {}
        scores = []

        for text in texts:
            score = self.calculate_quality_score(text)
            scores.append(score.total_score)

            if score.passed:
                passed += 1
            else:
                failed += 1
                for reason in score.filter_reasons:
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1

        return {
            "total": len(texts),
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / max(1, len(texts)),
            "avg_score": sum(scores) / max(1, len(scores)),
            "min_score": min(scores) if scores else 0,
            "max_score": max(scores) if scores else 0,
            "filter_reasons": reason_counts,
        }
