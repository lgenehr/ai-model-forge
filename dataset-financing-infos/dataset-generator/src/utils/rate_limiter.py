"""
Rate limiting utilities for API access.
"""

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from .logger import get_logger

logger = get_logger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for a rate limit."""

    requests: int
    period_seconds: float
    name: str = ""

    @property
    def requests_per_second(self) -> float:
        return self.requests / self.period_seconds


class RateLimiter:
    """
    Async rate limiter using token bucket algorithm.
    Supports multiple named rate limits for different APIs.
    """

    def __init__(self) -> None:
        self._limits: dict[str, RateLimitConfig] = {}
        self._tokens: dict[str, float] = {}
        self._last_update: dict[str, float] = {}
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    def add_limit(
        self,
        name: str,
        requests: int,
        period_seconds: float,
    ) -> None:
        """
        Add a rate limit configuration.

        Args:
            name: Name/identifier for this limit
            requests: Number of requests allowed
            period_seconds: Time period in seconds
        """
        self._limits[name] = RateLimitConfig(
            requests=requests,
            period_seconds=period_seconds,
            name=name,
        )
        self._tokens[name] = float(requests)
        self._last_update[name] = time.monotonic()
        logger.info(
            "Rate limit configured",
            name=name,
            requests=requests,
            period_seconds=period_seconds,
        )

    def _refill_tokens(self, name: str) -> None:
        """Refill tokens based on elapsed time."""
        if name not in self._limits:
            return

        config = self._limits[name]
        now = time.monotonic()
        elapsed = now - self._last_update[name]

        # Add tokens based on elapsed time
        tokens_to_add = elapsed * config.requests_per_second
        self._tokens[name] = min(
            float(config.requests),
            self._tokens[name] + tokens_to_add,
        )
        self._last_update[name] = now

    async def acquire(self, name: str, tokens: int = 1) -> None:
        """
        Acquire rate limit tokens, waiting if necessary.

        Args:
            name: Rate limit name
            tokens: Number of tokens to acquire
        """
        if name not in self._limits:
            # No limit configured, proceed immediately
            return

        async with self._locks[name]:
            while True:
                self._refill_tokens(name)

                if self._tokens[name] >= tokens:
                    self._tokens[name] -= tokens
                    return

                # Calculate wait time
                config = self._limits[name]
                tokens_needed = tokens - self._tokens[name]
                wait_time = tokens_needed / config.requests_per_second

                logger.debug(
                    "Rate limit wait",
                    name=name,
                    tokens_needed=tokens_needed,
                    wait_seconds=wait_time,
                )

                await asyncio.sleep(wait_time)

    def remaining_tokens(self, name: str) -> float:
        """Get remaining tokens for a rate limit."""
        if name not in self._limits:
            return float("inf")
        self._refill_tokens(name)
        return self._tokens[name]

    async def __aenter__(self) -> "RateLimiter":
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass


class RateLimitedSession:
    """
    A rate-limited wrapper for async HTTP sessions.
    """

    def __init__(
        self,
        rate_limiter: RateLimiter,
        limit_name: str,
    ) -> None:
        self.rate_limiter = rate_limiter
        self.limit_name = limit_name

    async def request(
        self,
        session: Any,  # aiohttp.ClientSession
        method: str,
        url: str,
        **kwargs: Any,
    ) -> Any:
        """
        Make a rate-limited request.

        Args:
            session: aiohttp ClientSession
            method: HTTP method
            url: Request URL
            **kwargs: Additional request arguments

        Returns:
            Response object
        """
        await self.rate_limiter.acquire(self.limit_name)
        return await session.request(method, url, **kwargs)

    async def get(self, session: Any, url: str, **kwargs: Any) -> Any:
        """Rate-limited GET request."""
        return await self.request(session, "GET", url, **kwargs)

    async def post(self, session: Any, url: str, **kwargs: Any) -> Any:
        """Rate-limited POST request."""
        return await self.request(session, "POST", url, **kwargs)


# Pre-configured rate limiters for common APIs
def create_default_rate_limiter() -> RateLimiter:
    """Create a rate limiter with default API configurations."""
    limiter = RateLimiter()

    # NewsAPI - 100 requests/day
    limiter.add_limit("newsapi", requests=100, period_seconds=86400)

    # YouTube - 10000 requests/day
    limiter.add_limit("youtube", requests=10000, period_seconds=86400)

    # Reddit - 60 requests/minute
    limiter.add_limit("reddit", requests=60, period_seconds=60)

    # Semantic Scholar - 100 requests/5 minutes
    limiter.add_limit("semantic_scholar", requests=100, period_seconds=300)

    # Generic web scraping - 30 requests/minute
    limiter.add_limit("web_scraping", requests=30, period_seconds=60)

    # Wikipedia - 200 requests/minute
    limiter.add_limit("wikipedia", requests=200, period_seconds=60)

    # arXiv - 3 requests/second (conservative)
    limiter.add_limit("arxiv", requests=3, period_seconds=1)

    # RSS feeds - 60 requests/minute
    limiter.add_limit("rss", requests=60, period_seconds=60)

    return limiter


# Global rate limiter instance
_global_rate_limiter: RateLimiter | None = None


def get_rate_limiter() -> RateLimiter:
    """Get or create the global rate limiter."""
    global _global_rate_limiter
    if _global_rate_limiter is None:
        _global_rate_limiter = create_default_rate_limiter()
    return _global_rate_limiter
