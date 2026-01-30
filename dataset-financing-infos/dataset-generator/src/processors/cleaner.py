"""
Text cleaning and normalization utilities.
"""

import html
import re
import unicodedata
from typing import Any

from ..utils.logger import get_logger

logger = get_logger(__name__)


class TextCleaner:
    """
    Advanced text cleaning and normalization.
    """

    # Patterns for cleaning
    HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
    URL_PATTERN = re.compile(
        r"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[^\s]*",
        re.IGNORECASE,
    )
    EMAIL_PATTERN = re.compile(
        r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        re.IGNORECASE,
    )
    MULTIPLE_SPACES = re.compile(r" {2,}")
    MULTIPLE_NEWLINES = re.compile(r"\n{3,}")
    LEADING_TRAILING_WHITESPACE = re.compile(r"^[ \t]+|[ \t]+$", re.MULTILINE)

    # Boilerplate patterns
    BOILERPLATE_PATTERNS = [
        re.compile(r"^\s*compartilh(e|ar).*$", re.IGNORECASE | re.MULTILINE),
        re.compile(r"^\s*siga-nos.*$", re.IGNORECASE | re.MULTILINE),
        re.compile(r"^\s*inscreva-se.*$", re.IGNORECASE | re.MULTILINE),
        re.compile(r"^\s*assine.*newsletter.*$", re.IGNORECASE | re.MULTILINE),
        re.compile(r"^\s*publicidade.*$", re.IGNORECASE | re.MULTILINE),
        re.compile(r"^\s*leia (também|mais).*$", re.IGNORECASE | re.MULTILINE),
        re.compile(r"^\s*veja (também|mais).*$", re.IGNORECASE | re.MULTILINE),
        re.compile(r"^\s*copyright.*$", re.IGNORECASE | re.MULTILINE),
        re.compile(r"^\s*todos os direitos reservados.*$", re.IGNORECASE | re.MULTILINE),
        re.compile(r"^\s*\d+ comentários?.*$", re.IGNORECASE | re.MULTILINE),
    ]

    # Emoji pattern
    EMOJI_PATTERN = re.compile(
        "["
        "\U0001f600-\U0001f64f"  # Emoticons
        "\U0001f300-\U0001f5ff"  # Symbols & pictographs
        "\U0001f680-\U0001f6ff"  # Transport & map symbols
        "\U0001f700-\U0001f77f"  # Alchemical symbols
        "\U0001f780-\U0001f7ff"  # Geometric shapes extended
        "\U0001f800-\U0001f8ff"  # Supplemental arrows-C
        "\U0001f900-\U0001f9ff"  # Supplemental symbols and pictographs
        "\U0001fa00-\U0001fa6f"  # Chess symbols
        "\U0001fa70-\U0001faff"  # Symbols and pictographs extended-A
        "\U00002702-\U000027b0"  # Dingbats
        "\U000024c2-\U0001f251"
        "]+"
    )

    def __init__(
        self,
        remove_urls: bool = True,
        remove_emails: bool = True,
        remove_emojis: bool = False,
        remove_boilerplate: bool = True,
        normalize_unicode: bool = True,
        normalize_whitespace: bool = True,
        fix_encoding: bool = True,
    ) -> None:
        """
        Initialize text cleaner.

        Args:
            remove_urls: Whether to remove URLs
            remove_emails: Whether to remove email addresses
            remove_emojis: Whether to remove emojis
            remove_boilerplate: Whether to remove boilerplate text
            normalize_unicode: Whether to normalize Unicode (NFKC)
            normalize_whitespace: Whether to normalize whitespace
            fix_encoding: Whether to fix encoding issues
        """
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.remove_emojis = remove_emojis
        self.remove_boilerplate = remove_boilerplate
        self.normalize_unicode = normalize_unicode
        self.normalize_whitespace = normalize_whitespace
        self.fix_encoding = fix_encoding

    def clean(self, text: str) -> str:
        """
        Clean text with all configured operations.

        Args:
            text: Text to clean

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        # Fix encoding issues first
        if self.fix_encoding:
            text = self._fix_encoding(text)

        # Normalize Unicode
        if self.normalize_unicode:
            text = self._normalize_unicode(text)

        # Remove HTML tags
        text = self._remove_html(text)

        # Decode HTML entities
        text = html.unescape(text)

        # Remove URLs
        if self.remove_urls:
            text = self._remove_urls(text)

        # Remove emails
        if self.remove_emails:
            text = self._remove_emails(text)

        # Remove emojis
        if self.remove_emojis:
            text = self._remove_emojis(text)

        # Remove boilerplate
        if self.remove_boilerplate:
            text = self._remove_boilerplate(text)

        # Normalize whitespace
        if self.normalize_whitespace:
            text = self._normalize_whitespace(text)

        # Normalize punctuation
        text = self._normalize_punctuation(text)

        return text.strip()

    def _fix_encoding(self, text: str) -> str:
        """Fix common encoding issues."""
        # Remove null bytes and BOM
        text = text.replace("\x00", "")
        text = text.replace("\ufeff", "")

        # Try to fix mojibake (UTF-8 decoded as Latin-1)
        try:
            # Check if text might be double-encoded
            if any(c in text for c in ["\xc3", "\xe2", "\xc2"]):
                try:
                    fixed = text.encode("latin-1").decode("utf-8")
                    # Only use fix if it produces valid text
                    if fixed.isprintable() or "\n" in fixed:
                        text = fixed
                except (UnicodeDecodeError, UnicodeEncodeError):
                    pass
        except Exception:
            pass

        return text

    def _normalize_unicode(self, text: str) -> str:
        """Normalize Unicode to NFKC form."""
        return unicodedata.normalize("NFKC", text)

    def _remove_html(self, text: str) -> str:
        """Remove HTML tags."""
        return self.HTML_TAG_PATTERN.sub("", text)

    def _remove_urls(self, text: str) -> str:
        """Remove URLs."""
        return self.URL_PATTERN.sub("", text)

    def _remove_emails(self, text: str) -> str:
        """Remove email addresses."""
        return self.EMAIL_PATTERN.sub("[EMAIL]", text)

    def _remove_emojis(self, text: str) -> str:
        """Remove emojis."""
        return self.EMOJI_PATTERN.sub("", text)

    def _remove_boilerplate(self, text: str) -> str:
        """Remove boilerplate text."""
        for pattern in self.BOILERPLATE_PATTERNS:
            text = pattern.sub("", text)
        return text

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace."""
        # Remove leading/trailing whitespace per line
        text = self.LEADING_TRAILING_WHITESPACE.sub("", text)
        # Collapse multiple spaces
        text = self.MULTIPLE_SPACES.sub(" ", text)
        # Collapse multiple newlines
        text = self.MULTIPLE_NEWLINES.sub("\n\n", text)
        return text

    def _normalize_punctuation(self, text: str) -> str:
        """Normalize punctuation."""
        # Fix common punctuation issues
        text = re.sub(r"\.{4,}", "...", text)  # Multiple dots
        text = re.sub(r"!{2,}", "!", text)  # Multiple exclamation
        text = re.sub(r"\?{2,}", "?", text)  # Multiple question
        text = re.sub(r"[""„]", '"', text)  # Quote normalization
        text = re.sub(r"[''‚]", "'", text)  # Apostrophe normalization
        text = re.sub(r"[—–]", "-", text)  # Dash normalization
        return text

    def get_stats(self, original: str, cleaned: str) -> dict[str, Any]:
        """Get cleaning statistics."""
        return {
            "original_length": len(original),
            "cleaned_length": len(cleaned),
            "reduction_ratio": 1 - (len(cleaned) / max(1, len(original))),
            "original_lines": original.count("\n") + 1,
            "cleaned_lines": cleaned.count("\n") + 1,
        }


def clean_text(
    text: str,
    remove_urls: bool = True,
    remove_emails: bool = True,
    remove_emojis: bool = False,
) -> str:
    """
    Convenience function for text cleaning.

    Args:
        text: Text to clean
        remove_urls: Whether to remove URLs
        remove_emails: Whether to remove emails
        remove_emojis: Whether to remove emojis

    Returns:
        Cleaned text
    """
    cleaner = TextCleaner(
        remove_urls=remove_urls,
        remove_emails=remove_emails,
        remove_emojis=remove_emojis,
    )
    return cleaner.clean(text)
