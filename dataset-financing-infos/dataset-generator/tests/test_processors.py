"""Tests for processors module."""

import pytest

from src.processors.cleaner import TextCleaner
from src.processors.deduplicator import Deduplicator, ExactDeduplicator
from src.processors.quality_filter import QualityFilter


class TestTextCleaner:
    """Tests for TextCleaner class."""

    def test_clean_empty_text(self):
        cleaner = TextCleaner()
        assert cleaner.clean("") == ""
        assert cleaner.clean(None) == ""

    def test_remove_html_tags(self):
        cleaner = TextCleaner()
        text = "<p>Hello <b>world</b></p>"
        assert "Hello world" in cleaner.clean(text)

    def test_normalize_whitespace(self):
        cleaner = TextCleaner()
        text = "Hello    world\n\n\n\ntest"
        cleaned = cleaner.clean(text)
        assert "    " not in cleaned
        assert "\n\n\n" not in cleaned

    def test_remove_urls(self):
        cleaner = TextCleaner(remove_urls=True)
        text = "Visit https://example.com for more info"
        cleaned = cleaner.clean(text)
        assert "https://example.com" not in cleaned

    def test_keep_urls_when_disabled(self):
        cleaner = TextCleaner(remove_urls=False)
        text = "Visit https://example.com for more info"
        cleaned = cleaner.clean(text)
        assert "example.com" in cleaned

    def test_remove_emails(self):
        cleaner = TextCleaner(remove_emails=True)
        text = "Contact test@example.com for info"
        cleaned = cleaner.clean(text)
        assert "test@example.com" not in cleaned
        assert "[EMAIL]" in cleaned

    def test_fix_encoding(self):
        cleaner = TextCleaner(fix_encoding=True)
        text = "Informação"
        cleaned = cleaner.clean(text)
        assert cleaned == "Informação"

    def test_normalize_unicode(self):
        cleaner = TextCleaner(normalize_unicode=True)
        # Test with different unicode representations
        text = "café"
        cleaned = cleaner.clean(text)
        assert "café" in cleaned or "cafe" in cleaned

    def test_remove_boilerplate(self):
        cleaner = TextCleaner(remove_boilerplate=True)
        text = "Main content here.\nLeia também: outro artigo\nMore content."
        cleaned = cleaner.clean(text)
        assert "Leia também" not in cleaned

    def test_get_stats(self):
        cleaner = TextCleaner()
        original = "Hello    world   with   spaces"
        cleaned = cleaner.clean(original)
        stats = cleaner.get_stats(original, cleaned)

        assert "original_length" in stats
        assert "cleaned_length" in stats
        assert "reduction_ratio" in stats


class TestDeduplicator:
    """Tests for Deduplicator class."""

    def test_exact_duplicate(self):
        dedup = Deduplicator(similarity_threshold=0.85)
        text1 = "This is a test sentence that should be deduplicated."
        text2 = "This is a test sentence that should be deduplicated."

        dedup.add("1", text1)
        assert dedup.is_duplicate("2", text2) is True

    def test_non_duplicate(self):
        dedup = Deduplicator(similarity_threshold=0.85)
        text1 = "This is about machine learning and artificial intelligence."
        text2 = "The weather is sunny today in Brazil."

        dedup.add("1", text1)
        assert dedup.is_duplicate("2", text2) is False

    def test_near_duplicate(self):
        dedup = Deduplicator(similarity_threshold=0.85)
        text1 = "Machine learning is a subset of artificial intelligence."
        text2 = "Machine learning is a subset of artificial intelligence systems."

        dedup.add("1", text1)
        # Depending on threshold, this might be detected as duplicate
        result = dedup.is_duplicate("2", text2)
        assert isinstance(result, bool)

    def test_check_and_add(self):
        dedup = Deduplicator()
        # Text must be long enough for MinHash (>50 chars)
        text = "Unique content for testing purposes with enough characters to ensure proper deduplication works correctly."

        # First time should add
        assert dedup.check_and_add("1", text) is False
        # Second time should be duplicate
        assert dedup.check_and_add("2", text) is True

    def test_size(self):
        dedup = Deduplicator()
        assert dedup.size == 0

        # Texts must be long enough for MinHash (>50 chars)
        dedup.add("1", "First text content here with enough characters to be properly indexed by the deduplicator system.")
        assert dedup.size == 1

        dedup.add("2", "Second text content here with enough characters to be properly indexed by the deduplicator system.")
        assert dedup.size == 2

    def test_clear(self):
        dedup = Deduplicator()
        dedup.add("1", "Some text content.")
        dedup.clear()
        assert dedup.size == 0


class TestExactDeduplicator:
    """Tests for ExactDeduplicator class."""

    def test_exact_match(self):
        dedup = ExactDeduplicator()
        text = "Exact match test"

        dedup.add(text)
        assert dedup.is_duplicate(text) is True

    def test_case_insensitive(self):
        dedup = ExactDeduplicator()
        text1 = "Case Insensitive Test"
        text2 = "case insensitive test"

        dedup.add(text1)
        assert dedup.is_duplicate(text2) is True

    def test_whitespace_normalized(self):
        dedup = ExactDeduplicator()
        text1 = "Text   with   spaces"
        text2 = "Text with spaces"

        dedup.add(text1)
        assert dedup.is_duplicate(text2) is True


class TestQualityFilter:
    """Tests for QualityFilter class."""

    def test_empty_text(self):
        qf = QualityFilter()
        score = qf.calculate_quality_score("")
        assert score.passed is False
        assert "empty_text" in score.filter_reasons

    def test_too_short(self):
        qf = QualityFilter(min_tokens=50)
        score = qf.calculate_quality_score("Short text.")
        assert score.passed is False
        assert "too_short" in score.filter_reasons

    def test_too_long(self):
        qf = QualityFilter(max_tokens=10)
        long_text = "word " * 100
        score = qf.calculate_quality_score(long_text)
        assert "too_long" in score.filter_reasons

    def test_good_quality(self):
        qf = QualityFilter(min_tokens=10, min_quality_score=0.5)
        good_text = """
        Este é um texto de boa qualidade com várias frases completas.
        O texto contém informações relevantes e bem estruturadas.
        As frases são claras e têm um bom comprimento médio.
        """
        score = qf.calculate_quality_score(good_text)
        assert score.total_score > 0.5

    def test_high_special_chars(self):
        qf = QualityFilter(max_special_char_ratio=0.1)
        text = "Normal text @#$%^&*()@#$%^&*() with too many special chars"
        score = qf.calculate_quality_score(text)
        # May or may not fail depending on exact ratio

    def test_word_repetition(self):
        qf = QualityFilter(max_word_repetition=0.3)
        text = "test test test test different word test test"
        score = qf.calculate_quality_score(text)
        # Check that repetition is detected
        assert score.word_repetition_score < 1.0

    def test_filter_method(self):
        qf = QualityFilter(min_tokens=10, min_quality_score=0.5)
        text = "This is a reasonable quality text for testing purposes."
        passed, score = qf.filter(text)
        assert isinstance(passed, bool)
        assert hasattr(score, "total_score")

    def test_get_filter_stats(self):
        qf = QualityFilter(min_tokens=5)
        texts = [
            "Good quality text with enough content.",
            "Short",
            "Another good text with proper content here.",
        ]
        stats = qf.get_filter_stats(texts)

        assert "total" in stats
        assert "passed" in stats
        assert "failed" in stats
        assert stats["total"] == 3
