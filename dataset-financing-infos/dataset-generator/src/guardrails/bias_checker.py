"""
Bias detection and checking.
"""

import re
from dataclasses import dataclass
from typing import Any

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BiasResult:
    """Result of bias checking."""

    has_bias: bool
    bias_types: list[str]
    flagged_phrases: list[str]
    severity: str  # low, medium, high
    confidence: float


class BiasChecker:
    """
    Checks for various types of bias in text.
    """

    # Gender bias patterns
    GENDER_BIAS_PATTERNS = [
        # Stereotypical phrases
        (r"mulher(es)?\s+(não\s+)?(sabe|entende|consegue)", "gender", "high"),
        (r"homem\s+(não\s+)?chora", "gender", "medium"),
        (r"trabalho\s+de\s+mulher", "gender", "medium"),
        (r"coisa\s+de\s+(homem|mulher)", "gender", "low"),
        (r"sexo\s+frágil", "gender", "high"),
        (r"lugar\s+de\s+mulher", "gender", "high"),
    ]

    # Racial bias patterns
    RACIAL_BIAS_PATTERNS = [
        (r"(preto|negro)\s+(vagabundo|bandido|ladrão)", "racial", "high"),
        (r"(branco|europeu)\s+(superior|melhor)", "racial", "high"),
        (r"raça\s+(inferior|superior)", "racial", "high"),
        (r"cor\s+da\s+pele.*?(determina|define)", "racial", "medium"),
    ]

    # Political bias patterns (extreme positions)
    POLITICAL_BIAS_PATTERNS = [
        (r"(comunista|fascista)\s+(lixo|merda)", "political", "high"),
        (r"(direita|esquerda)\s+(burra|ignorante)", "political", "medium"),
        (r"todo\s+(petista|bolsonarista)\s+(é|são)", "political", "medium"),
        (r"(exterminar|matar)\s+(esquerdista|direitista)", "political", "high"),
    ]

    # Religious bias patterns
    RELIGIOUS_BIAS_PATTERNS = [
        (r"(cristão|judeu|muçulmano|ateu)\s+(idiota|burro)", "religious", "high"),
        (r"(religião|ateísmo)\s+(doença|câncer)", "religious", "high"),
        (r"(crentes|ateus)\s+são\s+todos", "religious", "medium"),
    ]

    # Age bias patterns
    AGE_BIAS_PATTERNS = [
        (r"(velho|idoso)\s+(inútil|imprestável)", "age", "high"),
        (r"(jovem|adolescente)\s+(irresponsável|burro)", "age", "medium"),
        (r"geração\s+\w+\s+(fracassada|perdida)", "age", "medium"),
    ]

    # Socioeconomic bias patterns
    SOCIOECONOMIC_BIAS_PATTERNS = [
        (r"(pobre|favelado)\s+(vagabundo|bandido)", "socioeconomic", "high"),
        (r"(rico|burguês)\s+(explorador|ladrão)", "socioeconomic", "medium"),
        (r"(nordestino|baiano)\s+(preguiçoso|burro)", "regional", "high"),
    ]

    def __init__(
        self,
        check_gender: bool = True,
        check_racial: bool = True,
        check_political: bool = True,
        check_religious: bool = True,
        check_age: bool = True,
        check_socioeconomic: bool = True,
        min_severity: str = "low",
    ) -> None:
        """
        Initialize bias checker.

        Args:
            check_*: Whether to check each bias type
            min_severity: Minimum severity to flag (low, medium, high)
        """
        self.check_gender = check_gender
        self.check_racial = check_racial
        self.check_political = check_political
        self.check_religious = check_religious
        self.check_age = check_age
        self.check_socioeconomic = check_socioeconomic
        self.min_severity = min_severity

        # Compile patterns
        self._patterns: list[tuple[re.Pattern, str, str]] = []

        if check_gender:
            self._patterns.extend(
                [(re.compile(p, re.IGNORECASE), t, s) for p, t, s in self.GENDER_BIAS_PATTERNS]
            )
        if check_racial:
            self._patterns.extend(
                [(re.compile(p, re.IGNORECASE), t, s) for p, t, s in self.RACIAL_BIAS_PATTERNS]
            )
        if check_political:
            self._patterns.extend(
                [(re.compile(p, re.IGNORECASE), t, s) for p, t, s in self.POLITICAL_BIAS_PATTERNS]
            )
        if check_religious:
            self._patterns.extend(
                [(re.compile(p, re.IGNORECASE), t, s) for p, t, s in self.RELIGIOUS_BIAS_PATTERNS]
            )
        if check_age:
            self._patterns.extend(
                [(re.compile(p, re.IGNORECASE), t, s) for p, t, s in self.AGE_BIAS_PATTERNS]
            )
        if check_socioeconomic:
            self._patterns.extend(
                [(re.compile(p, re.IGNORECASE), t, s) for p, t, s in self.SOCIOECONOMIC_BIAS_PATTERNS]
            )

        self._severity_order = {"low": 0, "medium": 1, "high": 2}

    def _severity_passes(self, severity: str) -> bool:
        """Check if severity passes minimum threshold."""
        return self._severity_order.get(severity, 0) >= self._severity_order.get(
            self.min_severity, 0
        )

    def check(self, text: str) -> BiasResult:
        """
        Check text for bias.

        Args:
            text: Text to check

        Returns:
            BiasResult with findings
        """
        if not text:
            return BiasResult(
                has_bias=False,
                bias_types=[],
                flagged_phrases=[],
                severity="low",
                confidence=1.0,
            )

        bias_types = set()
        flagged_phrases = []
        max_severity = "low"

        for pattern, bias_type, severity in self._patterns:
            if not self._severity_passes(severity):
                continue

            matches = pattern.findall(text)
            if matches:
                bias_types.add(bias_type)
                # Get the matched text
                for match in pattern.finditer(text):
                    flagged_phrases.append(match.group()[:100])

                if self._severity_order.get(severity, 0) > self._severity_order.get(
                    max_severity, 0
                ):
                    max_severity = severity

        has_bias = len(bias_types) > 0

        # Calculate confidence based on number of matches
        confidence = min(0.5 + len(flagged_phrases) * 0.1, 1.0) if has_bias else 0.0

        if has_bias:
            logger.debug(
                "Bias detected",
                types=list(bias_types),
                severity=max_severity,
                phrases_count=len(flagged_phrases),
            )

        return BiasResult(
            has_bias=has_bias,
            bias_types=list(bias_types),
            flagged_phrases=flagged_phrases[:10],  # Limit to 10
            severity=max_severity,
            confidence=confidence,
        )

    def has_bias(self, text: str) -> bool:
        """Quick check if text has bias."""
        return self.check(text).has_bias

    def get_stats(self, texts: list[str]) -> dict[str, Any]:
        """Get bias statistics for batch."""
        with_bias = 0
        without_bias = 0
        type_counts: dict[str, int] = {}
        severity_counts: dict[str, int] = {"low": 0, "medium": 0, "high": 0}

        for text in texts:
            result = self.check(text)
            if result.has_bias:
                with_bias += 1
                severity_counts[result.severity] += 1
                for bias_type in result.bias_types:
                    type_counts[bias_type] = type_counts.get(bias_type, 0) + 1
            else:
                without_bias += 1

        return {
            "total": len(texts),
            "with_bias": with_bias,
            "without_bias": without_bias,
            "bias_rate": with_bias / max(1, len(texts)),
            "type_breakdown": type_counts,
            "severity_breakdown": severity_counts,
        }
