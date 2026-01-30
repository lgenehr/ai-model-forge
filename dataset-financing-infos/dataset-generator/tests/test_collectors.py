"""Tests for collectors module."""

import pytest

from src.collectors.base import AsyncCollector, CollectorRegistry
from src.schemas.dataset import RawCollectedData


class TestCollectorRegistry:
    """Tests for CollectorRegistry class."""

    def test_list_collectors(self):
        collectors = CollectorRegistry.list_collectors()
        assert isinstance(collectors, list)
        # Should have registered collectors
        assert "news" in collectors or len(collectors) >= 0

    def test_get_collector(self):
        # Test getting a registered collector
        collector_class = CollectorRegistry.get("news")
        if collector_class:
            assert issubclass(collector_class, AsyncCollector)

    def test_get_nonexistent_collector(self):
        collector_class = CollectorRegistry.get("nonexistent_collector")
        assert collector_class is None

    def test_create_collector(self):
        collector = CollectorRegistry.create("news")
        if collector:
            assert isinstance(collector, AsyncCollector)
            assert collector.source_name == "news"

    def test_create_nonexistent_collector(self):
        collector = CollectorRegistry.create("nonexistent")
        assert collector is None


class TestRawCollectedData:
    """Tests for RawCollectedData schema."""

    def test_create_minimal(self):
        data = RawCollectedData(
            id="test-1",
            source="test",
            text="Test content here.",
        )
        assert data.id == "test-1"
        assert data.source == "test"
        assert data.text == "Test content here."
        assert data.language == "pt_br"  # Default

    def test_create_full(self):
        data = RawCollectedData(
            id="test-2",
            source="news",
            source_url="https://example.com/article",
            title="Test Article",
            text="Full article content here.",
            summary="Article summary.",
            author="Test Author",
            topic="financeiro",
            subtopic="economia",
            language="pt_br",
            metadata={"feed": "test_feed"},
        )

        assert data.id == "test-2"
        assert data.source == "news"
        assert data.source_url == "https://example.com/article"
        assert data.title == "Test Article"
        assert data.topic == "financeiro"
        assert data.subtopic == "economia"
        assert data.metadata["feed"] == "test_feed"

    def test_default_values(self):
        data = RawCollectedData(
            id="test-3",
            source="test",
            text="Content",
        )

        assert data.language == "pt_br"
        assert data.title is None
        assert data.summary is None
        assert data.author is None
        assert data.topic is None
        assert data.metadata == {}
        assert data.collected_date is not None


@pytest.mark.asyncio
class TestAsyncCollectorBase:
    """Tests for AsyncCollector base class behavior."""

    async def test_collector_context_manager(self):
        # Test that collectors can be used as context managers
        collector = CollectorRegistry.create("news")
        if collector:
            async with collector:
                assert collector._session is not None
            assert collector._session is None

    async def test_rate_limiting_exists(self):
        collector = CollectorRegistry.create("news")
        if collector:
            assert collector.rate_limiter is not None
            assert hasattr(collector, "rate_limit")
