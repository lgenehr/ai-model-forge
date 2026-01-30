"""
Content filtering for inappropriate or harmful content.
"""

import re
from dataclasses import dataclass
from typing import Any

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ContentFilterResult:
    """Result of content filtering."""

    passed: bool
    reasons: list[str]
    flagged_patterns: list[str]
    confidence: float


class ContentFilter:
    """
    Filters inappropriate, harmful, or low-quality content.
    """

    # Profanity and offensive words (Portuguese)
    OFFENSIVE_PATTERNS = [
        r"\b(porra|caralho|merda|foder|filho\s*da\s*puta)\b",
        r"\b(desgraça|desgraçado|vagabundo|arrombado)\b",
        r"\b(viado|bicha|sapatão)\b",  # Homophobic slurs
        r"\b(preto|negro|macaco)\s+(fedido|imundo|burro)\b",  # Racist combinations
    ]

    # Spam patterns
    SPAM_PATTERNS = [
        r"(ganhe\s+dinheiro|fique\s+rico)\s+(rápido|fácil)",
        r"(clique\s+aqui|acesse\s+agora).*http",
        r"(curso\s+gratuito|método\s+secreto)",
        r"(apenas\s+r\$|somente\s+r\$|só\s+r\$)\s*\d+",
        r"(promoção\s+imperdível|última\s+chance)",
        r"(compartilhe|repasse)\s+(para|com)\s+\d+\s+(pessoas|amigos)",
    ]

    # Dangerous content patterns
    DANGEROUS_PATTERNS = [
        r"(como\s+fazer|receita\s+de)\s+(bomba|explosivo)",
        r"(como\s+)?hackear\s+(banco|conta)",
        r"(venda|compra)\s+de\s+(drogas|armas)",
        r"(suicídio|se\s+matar)\s+(como|método)",
    ]

    # Copyright/piracy patterns
    COPYRIGHT_PATTERNS = [
        r"(download|baixar)\s+(grátis|gratuito).*?(filme|série|livro|jogo)",
        r"(torrent|pirata|crackeado)",
        r"(site|link)\s+(pirata|ilegal)",
    ]

    def __init__(
        self,
        filter_offensive: bool = True,
        filter_spam: bool = True,
        filter_dangerous: bool = True,
        filter_copyright: bool = True,
        min_confidence: float = 0.7,
    ) -> None:
        """
        Initialize content filter.

        Args:
            filter_offensive: Filter offensive content
            filter_spam: Filter spam content
            filter_dangerous: Filter dangerous content
            filter_copyright: Filter copyright-infringing content
            min_confidence: Minimum confidence threshold
        """
        self.filter_offensive = filter_offensive
        self.filter_spam = filter_spam
        self.filter_dangerous = filter_dangerous
        self.filter_copyright = filter_copyright
        self.min_confidence = min_confidence

        # Compile patterns
        self._offensive_re = [
            re.compile(p, re.IGNORECASE) for p in self.OFFENSIVE_PATTERNS
        ]
        self._spam_re = [
            re.compile(p, re.IGNORECASE) for p in self.SPAM_PATTERNS
        ]
        self._dangerous_re = [
            re.compile(p, re.IGNORECASE) for p in self.DANGEROUS_PATTERNS
        ]
        self._copyright_re = [
            re.compile(p, re.IGNORECASE) for p in self.COPYRIGHT_PATTERNS
        ]

    def _check_patterns(
        self,
        text: str,
        patterns: list[re.Pattern],
        category: str,
    ) -> list[str]:
        """Check text against patterns."""
        matches = []
        for pattern in patterns:
            if pattern.search(text):
                matches.append(f"{category}: {pattern.pattern[:50]}")
        return matches

    def filter(self, text: str) -> ContentFilterResult:
        """
        Filter content for inappropriate material.

        Args:
            text: Text to filter

        Returns:
            ContentFilterResult with pass/fail and details
        """
        if not text:
            return ContentFilterResult(
                passed=True,
                reasons=[],
                flagged_patterns=[],
                confidence=1.0,
            )

        reasons = []
        flagged = []

        # Check offensive content
        if self.filter_offensive:
            matches = self._check_patterns(text, self._offensive_re, "offensive")
            if matches:
                reasons.append("contains_offensive_content")
                flagged.extend(matches)

        # Check spam
        if self.filter_spam:
            matches = self._check_patterns(text, self._spam_re, "spam")
            if matches:
                reasons.append("contains_spam")
                flagged.extend(matches)

        # Check dangerous content
        if self.filter_dangerous:
            matches = self._check_patterns(text, self._dangerous_re, "dangerous")
            if matches:
                reasons.append("contains_dangerous_content")
                flagged.extend(matches)

        # Check copyright content
        if self.filter_copyright:
            matches = self._check_patterns(text, self._copyright_re, "copyright")
            if matches:
                reasons.append("contains_copyright_content")
                flagged.extend(matches)

        # Calculate confidence
        confidence = 1.0 - (len(flagged) * 0.2)
        confidence = max(0.0, min(1.0, confidence))

        passed = len(reasons) == 0

        if not passed:
            logger.debug(
                "Content filtered",
                reasons=reasons,
                flagged_count=len(flagged),
            )

        return ContentFilterResult(
            passed=passed,
            reasons=reasons,
            flagged_patterns=flagged,
            confidence=confidence,
        )

    def is_safe(self, text: str) -> bool:
        """Quick check if content is safe."""
        return self.filter(text).passed

    def get_stats(self, texts: list[str]) -> dict[str, Any]:
        """Get filtering statistics for batch."""
        passed = 0
        failed = 0
        reason_counts: dict[str, int] = {}

        for text in texts:
            result = self.filter(text)
            if result.passed:
                passed += 1
            else:
                failed += 1
                for reason in result.reasons:
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1

        return {
            "total": len(texts),
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / max(1, len(texts)),
            "reason_breakdown": reason_counts,
        }
