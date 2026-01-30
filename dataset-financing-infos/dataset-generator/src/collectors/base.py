"""
Base collector class and registry.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

import aiohttp

from ..config.settings import Settings, get_settings
from ..schemas.dataset import RawCollectedData
from ..storage.checkpoint import CheckpointManager
from ..utils.logger import get_logger
from ..utils.rate_limiter import RateLimiter, get_rate_limiter

logger = get_logger(__name__)


class AsyncCollector(ABC):
    """
    Abstract base class for async data collectors.

    All collectors should inherit from this class and implement
    the collect method.
    """

    source_name: str = "unknown"
    rate_limit_name: str = "web_scraping"

    def __init__(
        self,
        settings: Settings | None = None,
        checkpoint_manager: CheckpointManager | None = None,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        """
        Initialize collector.

        Args:
            settings: Application settings
            checkpoint_manager: Checkpoint manager for resumable collection
            rate_limiter: Rate limiter for API calls
        """
        self.settings = settings or get_settings()
        self.checkpoint_manager = checkpoint_manager
        self.rate_limiter = rate_limiter or get_rate_limiter()
        self.logger = get_logger(self.__class__.__name__)
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> "AsyncCollector":
        """Async context manager entry."""
        await self.setup()
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        await self.cleanup()

    async def setup(self) -> None:
        """Setup resources (e.g., HTTP session)."""
        if self._session is None:
            timeout = aiohttp.ClientTimeout(total=self.settings.request_timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)

    async def cleanup(self) -> None:
        """Cleanup resources."""
        if self._session is not None:
            await self._session.close()
            self._session = None

    @property
    def session(self) -> aiohttp.ClientSession:
        """Get the HTTP session."""
        if self._session is None:
            raise RuntimeError("Collector not initialized. Use 'async with' context.")
        return self._session

    async def rate_limit(self) -> None:
        """Apply rate limiting before making a request."""
        await self.rate_limiter.acquire(self.rate_limit_name)

    def is_collected(self, topic: str, item_id: str) -> bool:
        """Check if an item has already been collected."""
        if self.checkpoint_manager is None:
            return False
        return self.checkpoint_manager.is_collected(self.source_name, topic, item_id)

    def mark_collected(self, topic: str, item_id: str, page: int | None = None) -> None:
        """Mark an item as collected."""
        if self.checkpoint_manager is not None:
            self.checkpoint_manager.mark_collected(
                self.source_name, topic, item_id, page=page
            )

    @abstractmethod
    async def collect(
        self,
        topic: str,
        subtopic: str | None = None,
        max_items: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[RawCollectedData]:
        """
        Collect data for a topic.

        Args:
            topic: Main topic to collect
            subtopic: Optional subtopic
            max_items: Maximum items to collect
            **kwargs: Additional collector-specific arguments

        Yields:
            RawCollectedData items
        """
        yield  # type: ignore

    async def collect_all(
        self,
        topics: list[str],
        max_items_per_topic: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[RawCollectedData]:
        """
        Collect data for multiple topics.

        Args:
            topics: List of topics to collect
            max_items_per_topic: Maximum items per topic
            **kwargs: Additional arguments

        Yields:
            RawCollectedData items
        """
        for topic in topics:
            self.logger.info(
                "Starting collection",
                source=self.source_name,
                topic=topic,
                max_items=max_items_per_topic,
            )
            try:
                async for item in self.collect(
                    topic, max_items=max_items_per_topic, **kwargs
                ):
                    yield item
            except Exception as e:
                self.logger.error(
                    "Collection error",
                    source=self.source_name,
                    topic=topic,
                    error=str(e),
                )
                continue

    async def fetch_url(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        timeout: aiohttp.ClientTimeout | None = None,
    ) -> str:
        """
        Fetch URL content with rate limiting.

        Args:
            url: URL to fetch
            headers: Optional headers
            params: Optional query parameters
            timeout: Optional request timeout override

        Returns:
            Response text
        """
        await self.rate_limit()
        async with self.session.get(
            url, headers=headers, params=params, timeout=timeout
        ) as response:
            response.raise_for_status()
            return await response.text()

    async def fetch_json(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        timeout: aiohttp.ClientTimeout | None = None,
    ) -> dict[str, Any]:
        """
        Fetch URL and parse as JSON.

        Args:
            url: URL to fetch
            headers: Optional headers
            params: Optional query parameters
            timeout: Optional request timeout override

        Returns:
            Parsed JSON response
        """
        await self.rate_limit()
        async with self.session.get(
            url, headers=headers, params=params, timeout=timeout
        ) as response:
            response.raise_for_status()
            return await response.json()


class CollectorRegistry:
    """
    Registry for collector classes.
    """

    _collectors: dict[str, type[AsyncCollector]] = {}

    @classmethod
    def register(cls, name: str) -> callable:
        """
        Decorator to register a collector class.

        Args:
            name: Source name for the collector
        """

        def decorator(collector_class: type[AsyncCollector]) -> type[AsyncCollector]:
            collector_class.source_name = name
            cls._collectors[name] = collector_class
            return collector_class

        return decorator

    @classmethod
    def get(cls, name: str) -> type[AsyncCollector] | None:
        """Get a collector class by name."""
        return cls._collectors.get(name)

    @classmethod
    def list_collectors(cls) -> list[str]:
        """List all registered collectors."""
        return list(cls._collectors.keys())

    @classmethod
    def create(
        cls,
        name: str,
        settings: Settings | None = None,
        checkpoint_manager: CheckpointManager | None = None,
        rate_limiter: RateLimiter | None = None,
    ) -> AsyncCollector | None:
        """
        Create a collector instance by name.

        Args:
            name: Source name
            settings: Application settings
            checkpoint_manager: Checkpoint manager
            rate_limiter: Rate limiter

        Returns:
            Collector instance or None
        """
        collector_class = cls.get(name)
        if collector_class is None:
            return None
        return collector_class(
            settings=settings,
            checkpoint_manager=checkpoint_manager,
            rate_limiter=rate_limiter,
        )

    @classmethod
    def create_all(
        cls,
        names: list[str] | None = None,
        settings: Settings | None = None,
        checkpoint_manager: CheckpointManager | None = None,
        rate_limiter: RateLimiter | None = None,
    ) -> dict[str, AsyncCollector]:
        """
        Create multiple collector instances.

        Args:
            names: List of source names (or all if None)
            settings: Application settings
            checkpoint_manager: Checkpoint manager
            rate_limiter: Rate limiter

        Returns:
            Dictionary of name -> collector
        """
        names = names or cls.list_collectors()
        collectors = {}
        for name in names:
            collector = cls.create(
                name,
                settings=settings,
                checkpoint_manager=checkpoint_manager,
                rate_limiter=rate_limiter,
            )
            if collector is not None:
                collectors[name] = collector
        return collectors
