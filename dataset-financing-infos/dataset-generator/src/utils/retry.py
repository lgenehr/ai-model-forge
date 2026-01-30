"""
Retry decorators and utilities using tenacity.
"""

import asyncio
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import Any, ParamSpec, TypeVar

from tenacity import (
    AsyncRetrying,
    RetryError,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_random_exponential,
)

from .logger import get_logger

logger = get_logger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


# Default exceptions to retry on
RETRYABLE_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
    OSError,
)


def sync_retry(
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 60.0,
    exceptions: tuple[type[Exception], ...] = RETRYABLE_EXCEPTIONS,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator for synchronous functions with retry logic.

    Args:
        max_attempts: Maximum number of retry attempts
        min_wait: Minimum wait time between retries (seconds)
        max_wait: Maximum wait time between retries (seconds)
        exceptions: Tuple of exception types to retry on

    Returns:
        Decorated function with retry logic
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            retryer = Retrying(
                stop=stop_after_attempt(max_attempts),
                wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
                retry=retry_if_exception_type(exceptions),
                reraise=True,
            )

            for attempt in retryer:
                with attempt:
                    try:
                        return func(*args, **kwargs)
                    except exceptions as e:
                        logger.warning(
                            "Retry attempt",
                            function=func.__name__,
                            attempt=attempt.retry_state.attempt_number,
                            max_attempts=max_attempts,
                            error=str(e),
                        )
                        raise

            # This should never be reached due to reraise=True
            raise RuntimeError("Unexpected retry state")

        return wrapper

    return decorator


def async_retry(
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 60.0,
    exceptions: tuple[type[Exception], ...] = RETRYABLE_EXCEPTIONS,
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """
    Decorator for async functions with retry logic.

    Args:
        max_attempts: Maximum number of retry attempts
        min_wait: Minimum wait time between retries (seconds)
        max_wait: Maximum wait time between retries (seconds)
        exceptions: Tuple of exception types to retry on

    Returns:
        Decorated async function with retry logic
    """

    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            retryer = AsyncRetrying(
                stop=stop_after_attempt(max_attempts),
                wait=wait_random_exponential(multiplier=1, min=min_wait, max=max_wait),
                retry=retry_if_exception_type(exceptions),
                reraise=True,
            )

            async for attempt in retryer:
                with attempt:
                    try:
                        return await func(*args, **kwargs)
                    except exceptions as e:
                        logger.warning(
                            "Async retry attempt",
                            function=func.__name__,
                            attempt=attempt.retry_state.attempt_number,
                            max_attempts=max_attempts,
                            error=str(e),
                        )
                        raise

            raise RuntimeError("Unexpected retry state")

        return wrapper

    return decorator


async def retry_with_fallback(
    primary_coro: Awaitable[T],
    fallback_coros: list[Awaitable[T]],
    exceptions: tuple[type[Exception], ...] = RETRYABLE_EXCEPTIONS,
) -> T:
    """
    Try primary coroutine, then fallbacks on failure.

    Args:
        primary_coro: Primary coroutine to try first
        fallback_coros: List of fallback coroutines
        exceptions: Exceptions that trigger fallback

    Returns:
        Result from first successful coroutine

    Raises:
        Last exception if all attempts fail
    """
    all_coros = [primary_coro] + fallback_coros
    last_error: Exception | None = None

    for i, coro in enumerate(all_coros):
        try:
            return await coro
        except exceptions as e:
            last_error = e
            logger.warning(
                "Fallback triggered",
                attempt=i + 1,
                total=len(all_coros),
                error=str(e),
            )
            continue

    if last_error:
        raise last_error
    raise RuntimeError("No coroutines provided")


class RetryBudget:
    """
    Manages a budget of retries across multiple operations.
    Useful for limiting total retries in a batch operation.
    """

    def __init__(self, max_retries: int = 10, window_seconds: float = 60.0):
        """
        Initialize retry budget.

        Args:
            max_retries: Maximum retries allowed in the window
            window_seconds: Time window for retry counting
        """
        self.max_retries = max_retries
        self.window_seconds = window_seconds
        self._retries: list[float] = []
        self._lock = asyncio.Lock()

    async def can_retry(self) -> bool:
        """Check if a retry is allowed within budget."""
        async with self._lock:
            now = asyncio.get_event_loop().time()
            # Clean up old retries outside window
            self._retries = [t for t in self._retries if now - t < self.window_seconds]
            return len(self._retries) < self.max_retries

    async def record_retry(self) -> None:
        """Record a retry attempt."""
        async with self._lock:
            self._retries.append(asyncio.get_event_loop().time())

    @property
    def remaining(self) -> int:
        """Get remaining retry budget."""
        now = asyncio.get_event_loop().time()
        recent = [t for t in self._retries if now - t < self.window_seconds]
        return max(0, self.max_retries - len(recent))
