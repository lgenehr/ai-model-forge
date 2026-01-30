"""
PII (Personally Identifiable Information) detection and removal.
"""

import re
from dataclasses import dataclass
from typing import Any

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PIIMatch:
    """A detected PII match."""

    type: str
    value: str
    start: int
    end: int
    replacement: str


@dataclass
class PIIResult:
    """Result of PII detection."""

    has_pii: bool
    matches: list[PIIMatch]
    cleaned_text: str
    types_found: list[str]


class PIIDetector:
    """
    Detects and removes PII from text.

    Supports Brazilian PII formats:
    - CPF (Cadastro de Pessoas Físicas)
    - CNPJ (Cadastro Nacional de Pessoa Jurídica)
    - Phone numbers
    - Email addresses
    - Addresses
    - Names (basic detection)
    """

    # Brazilian CPF pattern: XXX.XXX.XXX-XX
    CPF_PATTERN = re.compile(
        r"\b\d{3}[.\s]?\d{3}[.\s]?\d{3}[-.\s]?\d{2}\b"
    )

    # Brazilian CNPJ pattern: XX.XXX.XXX/XXXX-XX
    CNPJ_PATTERN = re.compile(
        r"\b\d{2}[.\s]?\d{3}[.\s]?\d{3}[/.\s]?\d{4}[-.\s]?\d{2}\b"
    )

    # Brazilian phone numbers
    PHONE_PATTERN = re.compile(
        r"\b(?:\+55\s?)?(?:\(?\d{2}\)?[-.\s]?)?\d{4,5}[-.\s]?\d{4}\b"
    )

    # Email pattern
    EMAIL_PATTERN = re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        re.IGNORECASE,
    )

    # Brazilian CEP (postal code)
    CEP_PATTERN = re.compile(r"\b\d{5}[-.\s]?\d{3}\b")

    # RG (identity document) - varies by state
    RG_PATTERN = re.compile(r"\b\d{1,2}[.\s]?\d{3}[.\s]?\d{3}[-.\s]?[\dXx]\b")

    # Credit card numbers (basic)
    CREDIT_CARD_PATTERN = re.compile(
        r"\b(?:\d{4}[-.\s]?){3}\d{4}\b"
    )

    # IP addresses
    IP_PATTERN = re.compile(
        r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
    )

    # Brazilian addresses (basic patterns)
    ADDRESS_PATTERNS = [
        re.compile(r"\b(rua|av\.|avenida|travessa|alameda)\s+[\w\s]+,?\s*\d+", re.IGNORECASE),
        re.compile(r"\bcep\s*:?\s*\d{5}[-.\s]?\d{3}\b", re.IGNORECASE),
    ]

    # Common Brazilian names (for basic detection)
    NAME_PATTERN = re.compile(
        r"\b(Maria|José|João|Ana|Paulo|Pedro|Carlos|Lucas|"
        r"Marcos|Luis|Gabriel|Rafael|Daniel|Fernanda|"
        r"Juliana|Camila|Larissa|Beatriz|Amanda)\s+"
        r"(?:da\s+|de\s+|dos\s+|das\s+)?"
        r"[A-Z][a-zà-ú]+(?:\s+[A-Z][a-zà-ú]+)?",
        re.UNICODE,
    )

    def __init__(
        self,
        detect_cpf: bool = True,
        detect_cnpj: bool = True,
        detect_phone: bool = True,
        detect_email: bool = True,
        detect_cep: bool = True,
        detect_rg: bool = True,
        detect_credit_card: bool = True,
        detect_ip: bool = True,
        detect_address: bool = True,
        detect_names: bool = False,  # More false positives
    ) -> None:
        """
        Initialize PII detector.

        Args:
            detect_*: Whether to detect each PII type
        """
        self.detect_cpf = detect_cpf
        self.detect_cnpj = detect_cnpj
        self.detect_phone = detect_phone
        self.detect_email = detect_email
        self.detect_cep = detect_cep
        self.detect_rg = detect_rg
        self.detect_credit_card = detect_credit_card
        self.detect_ip = detect_ip
        self.detect_address = detect_address
        self.detect_names = detect_names

        # Replacement tokens
        self.replacements = {
            "cpf": "[CPF]",
            "cnpj": "[CNPJ]",
            "phone": "[TELEFONE]",
            "email": "[EMAIL]",
            "cep": "[CEP]",
            "rg": "[RG]",
            "credit_card": "[CARTAO]",
            "ip": "[IP]",
            "address": "[ENDERECO]",
            "name": "[NOME]",
        }

    def _find_matches(
        self,
        text: str,
        pattern: re.Pattern,
        pii_type: str,
    ) -> list[PIIMatch]:
        """Find all matches of a pattern."""
        matches = []
        for match in pattern.finditer(text):
            matches.append(
                PIIMatch(
                    type=pii_type,
                    value=match.group(),
                    start=match.start(),
                    end=match.end(),
                    replacement=self.replacements.get(pii_type, "[PII]"),
                )
            )
        return matches

    def detect(self, text: str) -> PIIResult:
        """
        Detect PII in text.

        Args:
            text: Text to analyze

        Returns:
            PIIResult with matches and cleaned text
        """
        if not text:
            return PIIResult(
                has_pii=False,
                matches=[],
                cleaned_text="",
                types_found=[],
            )

        all_matches: list[PIIMatch] = []

        # Detect each PII type
        if self.detect_cpf:
            all_matches.extend(
                self._find_matches(text, self.CPF_PATTERN, "cpf")
            )

        if self.detect_cnpj:
            all_matches.extend(
                self._find_matches(text, self.CNPJ_PATTERN, "cnpj")
            )

        if self.detect_phone:
            all_matches.extend(
                self._find_matches(text, self.PHONE_PATTERN, "phone")
            )

        if self.detect_email:
            all_matches.extend(
                self._find_matches(text, self.EMAIL_PATTERN, "email")
            )

        if self.detect_cep:
            all_matches.extend(
                self._find_matches(text, self.CEP_PATTERN, "cep")
            )

        if self.detect_rg:
            all_matches.extend(
                self._find_matches(text, self.RG_PATTERN, "rg")
            )

        if self.detect_credit_card:
            all_matches.extend(
                self._find_matches(text, self.CREDIT_CARD_PATTERN, "credit_card")
            )

        if self.detect_ip:
            all_matches.extend(
                self._find_matches(text, self.IP_PATTERN, "ip")
            )

        if self.detect_address:
            for pattern in self.ADDRESS_PATTERNS:
                all_matches.extend(
                    self._find_matches(text, pattern, "address")
                )

        if self.detect_names:
            all_matches.extend(
                self._find_matches(text, self.NAME_PATTERN, "name")
            )

        # Sort by position (descending) for replacement
        all_matches.sort(key=lambda m: m.start, reverse=True)

        # Create cleaned text
        cleaned = text
        for match in all_matches:
            cleaned = (
                cleaned[: match.start]
                + match.replacement
                + cleaned[match.end:]
            )

        types_found = list({m.type for m in all_matches})

        if all_matches:
            logger.debug(
                "PII detected",
                count=len(all_matches),
                types=types_found,
            )

        return PIIResult(
            has_pii=len(all_matches) > 0,
            matches=all_matches,
            cleaned_text=cleaned,
            types_found=types_found,
        )

    def remove_pii(self, text: str) -> str:
        """Remove all PII and return cleaned text."""
        return self.detect(text).cleaned_text

    def has_pii(self, text: str) -> bool:
        """Quick check if text contains PII."""
        return self.detect(text).has_pii

    def get_stats(self, texts: list[str]) -> dict[str, Any]:
        """Get PII detection statistics for batch."""
        with_pii = 0
        without_pii = 0
        type_counts: dict[str, int] = {}
        total_matches = 0

        for text in texts:
            result = self.detect(text)
            if result.has_pii:
                with_pii += 1
                total_matches += len(result.matches)
                for pii_type in result.types_found:
                    type_counts[pii_type] = type_counts.get(pii_type, 0) + 1
            else:
                without_pii += 1

        return {
            "total": len(texts),
            "with_pii": with_pii,
            "without_pii": without_pii,
            "pii_rate": with_pii / max(1, len(texts)),
            "total_matches": total_matches,
            "type_breakdown": type_counts,
        }
