"""
Tokenizer validation and token counting.
"""

from typing import Any

from ..utils.logger import get_logger

logger = get_logger(__name__)


class TokenizerChecker:
    """
    Token counting and validation using tiktoken.
    """

    def __init__(
        self,
        model: str = "cl100k_base",
        min_tokens: int = 50,
        max_tokens: int = 4096,
    ) -> None:
        """
        Initialize tokenizer checker.

        Args:
            model: Tiktoken encoding model
            min_tokens: Minimum tokens
            max_tokens: Maximum tokens
        """
        self.model = model
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self._encoding = None

    @property
    def encoding(self):
        """Lazy load tiktoken encoding."""
        if self._encoding is None:
            try:
                import tiktoken

                self._encoding = tiktoken.get_encoding(self.model)
            except ImportError:
                logger.warning(
                    "tiktoken not available, using approximate counting"
                )
                self._encoding = None
        return self._encoding

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Text to count

        Returns:
            Token count
        """
        if not text:
            return 0

        if self.encoding is not None:
            return len(self.encoding.encode(text))

        # Fallback: approximate counting
        # Portuguese tends to have ~4 chars per token
        return len(text) // 4

    def validate(self, text: str) -> tuple[bool, dict[str, Any]]:
        """
        Validate text token count.

        Args:
            text: Text to validate

        Returns:
            Tuple of (valid, info)
        """
        token_count = self.count_tokens(text)

        info = {
            "token_count": token_count,
            "min_tokens": self.min_tokens,
            "max_tokens": self.max_tokens,
            "encoding": self.model,
        }

        if token_count < self.min_tokens:
            info["reason"] = "too_few_tokens"
            return False, info

        if token_count > self.max_tokens:
            info["reason"] = "too_many_tokens"
            return False, info

        return True, info

    def truncate(
        self,
        text: str,
        max_tokens: int | None = None,
        add_ellipsis: bool = True,
    ) -> str:
        """
        Truncate text to max tokens.

        Args:
            text: Text to truncate
            max_tokens: Maximum tokens (uses default if None)
            add_ellipsis: Add "..." at end if truncated

        Returns:
            Truncated text
        """
        max_tokens = max_tokens or self.max_tokens

        if self.encoding is None:
            # Approximate truncation
            char_limit = max_tokens * 4
            if len(text) <= char_limit:
                return text
            truncated = text[: char_limit - 3] if add_ellipsis else text[:char_limit]
            return truncated + "..." if add_ellipsis else truncated

        tokens = self.encoding.encode(text)

        if len(tokens) <= max_tokens:
            return text

        # Truncate tokens
        truncated_tokens = tokens[:max_tokens]
        truncated_text = self.encoding.decode(truncated_tokens)

        if add_ellipsis:
            truncated_text = truncated_text.rstrip() + "..."

        return truncated_text

    def split_into_chunks(
        self,
        text: str,
        chunk_size: int | None = None,
        overlap: int = 0,
    ) -> list[str]:
        """
        Split text into token-bounded chunks.

        Args:
            text: Text to split
            chunk_size: Tokens per chunk
            overlap: Token overlap between chunks

        Returns:
            List of text chunks
        """
        chunk_size = chunk_size or self.max_tokens

        if self.encoding is None:
            # Approximate splitting by characters
            char_size = chunk_size * 4
            char_overlap = overlap * 4
            chunks = []
            start = 0

            while start < len(text):
                end = start + char_size
                chunk = text[start:end]

                # Try to break at sentence boundary
                if end < len(text):
                    last_period = chunk.rfind(".")
                    if last_period > char_size // 2:
                        chunk = chunk[: last_period + 1]
                        end = start + last_period + 1

                chunks.append(chunk.strip())
                start = end - char_overlap

            return chunks

        tokens = self.encoding.encode(text)
        chunks = []
        start = 0

        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoding.decode(chunk_tokens)
            chunks.append(chunk_text.strip())
            start = end - overlap

        return chunks

    def get_stats(self, texts: list[str]) -> dict[str, Any]:
        """
        Get token statistics for a batch of texts.

        Args:
            texts: List of texts

        Returns:
            Statistics dictionary
        """
        token_counts = [self.count_tokens(t) for t in texts]
        valid_counts = [
            c
            for c in token_counts
            if self.min_tokens <= c <= self.max_tokens
        ]

        return {
            "total_texts": len(texts),
            "total_tokens": sum(token_counts),
            "avg_tokens": sum(token_counts) / max(1, len(token_counts)),
            "min_tokens": min(token_counts) if token_counts else 0,
            "max_tokens_found": max(token_counts) if token_counts else 0,
            "valid_count": len(valid_counts),
            "too_short": sum(1 for c in token_counts if c < self.min_tokens),
            "too_long": sum(1 for c in token_counts if c > self.max_tokens),
        }
