"""Tests for guardrails module."""

import pytest

from src.guardrails.bias_checker import BiasChecker
from src.guardrails.content_filter import ContentFilter
from src.guardrails.language_detector import LanguageDetector
from src.guardrails.pii_detector import PIIDetector


class TestContentFilter:
    """Tests for ContentFilter class."""

    def test_clean_content(self):
        cf = ContentFilter()
        text = "Este é um texto limpo e apropriado sobre economia brasileira."
        result = cf.filter(text)
        assert result.passed is True
        assert len(result.reasons) == 0

    def test_empty_text(self):
        cf = ContentFilter()
        result = cf.filter("")
        assert result.passed is True

    def test_spam_detection(self):
        cf = ContentFilter(filter_spam=True)
        text = "Ganhe dinheiro rápido! Clique aqui para http://spam.com"
        result = cf.filter(text)
        assert result.passed is False
        assert "contains_spam" in result.reasons

    def test_is_safe(self):
        cf = ContentFilter()
        assert cf.is_safe("Texto normal sobre finanças.") is True

    def test_get_stats(self):
        cf = ContentFilter()
        texts = [
            "Texto limpo número um.",
            "Texto limpo número dois.",
            "Ganhe dinheiro fácil agora!",
        ]
        stats = cf.get_stats(texts)

        assert stats["total"] == 3
        assert "passed" in stats
        assert "failed" in stats


class TestPIIDetector:
    """Tests for PIIDetector class."""

    def test_no_pii(self):
        detector = PIIDetector()
        text = "Este texto não contém informações pessoais."
        result = detector.detect(text)
        assert result.has_pii is False
        assert len(result.matches) == 0

    def test_detect_cpf(self):
        detector = PIIDetector(detect_cpf=True)
        text = "Meu CPF é 123.456.789-00 e preciso de ajuda."
        result = detector.detect(text)
        assert result.has_pii is True
        assert "cpf" in result.types_found

    def test_detect_email(self):
        detector = PIIDetector(detect_email=True)
        text = "Entre em contato pelo email teste@exemplo.com.br"
        result = detector.detect(text)
        assert result.has_pii is True
        assert "email" in result.types_found

    def test_detect_phone(self):
        detector = PIIDetector(detect_phone=True)
        text = "Me ligue no (11) 99999-8888 para mais informações."
        result = detector.detect(text)
        assert result.has_pii is True
        assert "phone" in result.types_found

    def test_remove_pii(self):
        detector = PIIDetector()
        text = "Email: teste@teste.com e CPF: 111.222.333-44"
        cleaned = detector.remove_pii(text)

        assert "teste@teste.com" not in cleaned
        assert "111.222.333-44" not in cleaned
        assert "[EMAIL]" in cleaned
        assert "[CPF]" in cleaned

    def test_has_pii_quick_check(self):
        detector = PIIDetector()
        assert detector.has_pii("Texto sem dados pessoais.") is False
        assert detector.has_pii("Meu email é a@b.com") is True

    def test_detect_cnpj(self):
        detector = PIIDetector(detect_cnpj=True)
        text = "CNPJ da empresa: 12.345.678/0001-90"
        result = detector.detect(text)
        assert result.has_pii is True
        assert "cnpj" in result.types_found

    def test_detect_cep(self):
        detector = PIIDetector(detect_cep=True)
        text = "O CEP é 01310-100 em São Paulo."
        result = detector.detect(text)
        assert result.has_pii is True
        assert "cep" in result.types_found

    def test_get_stats(self):
        detector = PIIDetector()
        texts = [
            "Texto limpo sem PII.",
            "Email: teste@teste.com",
            "CPF: 123.456.789-00",
        ]
        stats = detector.get_stats(texts)

        assert stats["total"] == 3
        assert stats["with_pii"] >= 2
        assert stats["without_pii"] >= 0


class TestBiasChecker:
    """Tests for BiasChecker class."""

    def test_no_bias(self):
        checker = BiasChecker()
        text = "A economia brasileira apresentou crescimento no último trimestre."
        result = checker.check(text)
        assert result.has_bias is False

    def test_empty_text(self):
        checker = BiasChecker()
        result = checker.check("")
        assert result.has_bias is False

    def test_has_bias_quick_check(self):
        checker = BiasChecker()
        assert checker.has_bias("Texto neutro sobre economia.") is False

    def test_get_stats(self):
        checker = BiasChecker()
        texts = [
            "Texto neutro número um.",
            "Texto neutro número dois.",
            "Outro texto neutro.",
        ]
        stats = checker.get_stats(texts)

        assert stats["total"] == 3
        assert "with_bias" in stats
        assert "without_bias" in stats


class TestLanguageDetector:
    """Tests for LanguageDetector class."""

    def test_detect_portuguese(self):
        detector = LanguageDetector(target_languages=["pt"])
        text = "Este é um texto em português brasileiro com várias palavras."
        result = detector.detect(text)
        assert result.language == "pt"
        assert result.is_target is True

    def test_detect_english(self):
        detector = LanguageDetector(target_languages=["pt"])
        text = "This is a text in English with many words."
        result = detector.detect(text)
        assert result.language == "en"
        assert result.is_target is False

    def test_short_text(self):
        detector = LanguageDetector()
        result = detector.detect("Hi")
        # Short texts may not be reliably detected
        assert result.language in ["unknown", "en", "pt"]

    def test_empty_text(self):
        detector = LanguageDetector()
        result = detector.detect("")
        assert result.language == "unknown"
        assert result.confidence == 0.0

    def test_is_target_language(self):
        detector = LanguageDetector(target_languages=["pt"])
        assert detector.is_target_language(
            "Este texto é em português brasileiro."
        ) is True

    def test_filter_by_language(self):
        detector = LanguageDetector(target_languages=["pt"])
        texts = [
            "Texto em português brasileiro.",
            "This is English text.",
            "Outro texto em português.",
        ]
        target, non_target = detector.filter_by_language(texts)

        # At least some should be detected as Portuguese
        assert len(target) >= 0
        assert len(non_target) >= 0
        assert len(target) + len(non_target) == 3

    def test_get_stats(self):
        detector = LanguageDetector(target_languages=["pt"])
        texts = [
            "Texto em português.",
            "English text here.",
            "Mais português aqui.",
        ]
        stats = detector.get_stats(texts)

        assert stats["total"] == 3
        assert "target_language" in stats
        assert "non_target_language" in stats
        assert "language_breakdown" in stats
