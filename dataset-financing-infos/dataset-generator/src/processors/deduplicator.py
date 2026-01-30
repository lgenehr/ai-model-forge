"""
Deduplication using MinHash for similarity detection.
"""

import hashlib
import re
from collections.abc import Iterator
from typing import Any

from datasketch import MinHash, MinHashLSH

from ..schemas.dataset import RawCollectedData
from ..utils.logger import get_logger

logger = get_logger(__name__)


class Deduplicator:
    """
    Deduplicator using MinHash LSH for approximate similarity detection.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        num_perm: int = 128,
        ngram_size: int = 5,
    ) -> None:
        """
        Initialize deduplicator.

        Args:
            similarity_threshold: Jaccard similarity threshold for duplicates
            num_perm: Number of permutations for MinHash
            ngram_size: Size of n-grams for shingling
        """
        self.similarity_threshold = similarity_threshold
        self.num_perm = num_perm
        self.ngram_size = ngram_size

        self._lsh = MinHashLSH(threshold=similarity_threshold, num_perm=num_perm)
        self._minhashes: dict[str, MinHash] = {}
        self._seen_hashes: set[str] = set()

    def _get_shingles(self, text: str) -> set[str]:
        """Get character-level n-gram shingles from text."""
        # Normalize text
        text = text.lower()
        text = re.sub(r"\s+", " ", text)

        shingles = set()
        for i in range(len(text) - self.ngram_size + 1):
            shingles.add(text[i : i + self.ngram_size])

        return shingles

    def _create_minhash(self, text: str) -> MinHash:
        """Create MinHash from text."""
        minhash = MinHash(num_perm=self.num_perm)
        shingles = self._get_shingles(text)

        for shingle in shingles:
            minhash.update(shingle.encode("utf-8"))

        return minhash

    def _get_exact_hash(self, text: str) -> str:
        """Get MD5 hash for exact duplicate detection."""
        normalized = re.sub(r"\s+", " ", text.lower().strip())
        return hashlib.md5(normalized.encode("utf-8")).hexdigest()

    def is_duplicate(self, item_id: str, text: str) -> bool:
        """
        Check if text is a duplicate.

        Args:
            item_id: Unique identifier for the item
            text: Text to check

        Returns:
            True if duplicate, False otherwise
        """
        if len(text) < 50:
            return False

        # Check exact hash first (fast path)
        exact_hash = self._get_exact_hash(text)
        if exact_hash in self._seen_hashes:
            return True

        # Check MinHash LSH for near-duplicates
        minhash = self._create_minhash(text)

        # Query LSH for similar items
        similar = self._lsh.query(minhash)
        if similar:
            # Verify similarity
            for similar_id in similar:
                if similar_id in self._minhashes:
                    similarity = minhash.jaccard(self._minhashes[similar_id])
                    if similarity >= self.similarity_threshold:
                        logger.debug(
                            "Duplicate found",
                            item_id=item_id,
                            similar_to=similar_id,
                            similarity=similarity,
                        )
                        return True

        return False

    def add(self, item_id: str, text: str) -> None:
        """
        Add item to deduplication index.

        Args:
            item_id: Unique identifier
            text: Text to index
        """
        if len(text) < 50:
            return

        # Add exact hash
        exact_hash = self._get_exact_hash(text)
        self._seen_hashes.add(exact_hash)

        # Add to MinHash LSH
        minhash = self._create_minhash(text)
        self._minhashes[item_id] = minhash

        try:
            self._lsh.insert(item_id, minhash)
        except ValueError:
            # Item already exists
            pass

    def check_and_add(self, item_id: str, text: str) -> bool:
        """
        Check if duplicate and add if not.

        Args:
            item_id: Unique identifier
            text: Text to check

        Returns:
            True if duplicate, False otherwise (and added)
        """
        if self.is_duplicate(item_id, text):
            return True

        self.add(item_id, text)
        return False

    def get_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate Jaccard similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Jaccard similarity (0-1)
        """
        mh1 = self._create_minhash(text1)
        mh2 = self._create_minhash(text2)
        return mh1.jaccard(mh2)

    def deduplicate_batch(
        self,
        items: list[RawCollectedData],
    ) -> Iterator[RawCollectedData]:
        """
        Deduplicate a batch of items.

        Args:
            items: List of items to deduplicate

        Yields:
            Non-duplicate items
        """
        duplicates = 0

        for item in items:
            if self.check_and_add(item.id, item.text):
                duplicates += 1
                continue
            yield item

        logger.info(
            "Batch deduplication complete",
            total=len(items),
            duplicates=duplicates,
            unique=len(items) - duplicates,
        )

    @property
    def size(self) -> int:
        """Get number of indexed items."""
        return len(self._minhashes)

    def clear(self) -> None:
        """Clear all indexed items."""
        self._lsh = MinHashLSH(
            threshold=self.similarity_threshold, num_perm=self.num_perm
        )
        self._minhashes.clear()
        self._seen_hashes.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get deduplication statistics."""
        return {
            "indexed_items": self.size,
            "exact_hashes": len(self._seen_hashes),
            "similarity_threshold": self.similarity_threshold,
            "num_perm": self.num_perm,
            "ngram_size": self.ngram_size,
        }


class ExactDeduplicator:
    """
    Simple exact deduplicator using content hashing.
    Faster than MinHash for exact duplicates only.
    """

    def __init__(self) -> None:
        self._hashes: set[str] = set()

    def _hash_text(self, text: str) -> str:
        """Create hash of normalized text."""
        normalized = re.sub(r"\s+", " ", text.lower().strip())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def is_duplicate(self, text: str) -> bool:
        """Check if text is an exact duplicate."""
        return self._hash_text(text) in self._hashes

    def add(self, text: str) -> None:
        """Add text to index."""
        self._hashes.add(self._hash_text(text))

    def check_and_add(self, text: str) -> bool:
        """Check and add if not duplicate."""
        text_hash = self._hash_text(text)
        if text_hash in self._hashes:
            return True
        self._hashes.add(text_hash)
        return False

    @property
    def size(self) -> int:
        """Get number of indexed items."""
        return len(self._hashes)

    def clear(self) -> None:
        """Clear index."""
        self._hashes.clear()
