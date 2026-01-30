"""Collectors package for data collection from various sources."""

from .base import AsyncCollector, CollectorRegistry
from .news import NewsCollector
from .wikipedia import WikipediaCollector
from .books import BooksCollector
from .academic import AcademicCollector
from .legal import LegalCollector
from .social_media import SocialMediaCollector
from .videos import VideosCollector

__all__ = [
    "AsyncCollector",
    "CollectorRegistry",
    "NewsCollector",
    "WikipediaCollector",
    "BooksCollector",
    "AcademicCollector",
    "LegalCollector",
    "SocialMediaCollector",
    "VideosCollector",
]
