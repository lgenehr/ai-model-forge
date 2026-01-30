"""
News collector for RSS feeds and news APIs.
"""

import hashlib
import re
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any
from xml.etree import ElementTree

from bs4 import BeautifulSoup

from ..schemas.dataset import RawCollectedData
from ..utils.retry import async_retry
from .base import AsyncCollector, CollectorRegistry


@CollectorRegistry.register("news")
class NewsCollector(AsyncCollector):
    """
    Collector for news articles from RSS feeds and APIs.
    """

    source_name = "news"
    rate_limit_name = "rss"

    # Brazilian news RSS feeds by topic
    RSS_FEEDS: dict[str, list[dict[str, str]]] = {
        "financeiro": [
            {"name": "G1 Economia", "url": "https://g1.globo.com/rss/g1/economia/"},
            {"name": "InfoMoney", "url": "https://www.infomoney.com.br/feed/"},
            {"name": "Valor Econômico", "url": "https://valor.globo.com/rss/"},
        ],
        "tecnologia": [
            {"name": "G1 Tecnologia", "url": "https://g1.globo.com/rss/g1/tecnologia/"},
            {"name": "Tecnoblog", "url": "https://tecnoblog.net/feed/"},
            {"name": "Canaltech", "url": "https://canaltech.com.br/rss/"},
        ],
        "ciencias": [
            {"name": "G1 Ciência", "url": "https://g1.globo.com/rss/g1/ciencia-e-saude/"},
        ],
        "saude": [
            {"name": "G1 Saúde", "url": "https://g1.globo.com/rss/g1/ciencia-e-saude/"},
        ],
        "juridico": [
            {"name": "ConJur", "url": "https://www.conjur.com.br/rss.xml"},
        ],
        "negocios": [
            {"name": "Exame", "url": "https://exame.com/feed/"},
        ],
        "educacao": [
            {"name": "G1 Educação", "url": "https://g1.globo.com/rss/g1/educacao/"},
        ],
        "meio_ambiente": [
            {"name": "G1 Natureza", "url": "https://g1.globo.com/rss/g1/natureza/"},
        ],
        "humanidades": [
            {"name": "G1 Política", "url": "https://g1.globo.com/rss/g1/politica/"},
        ],
        "cultura": [
            {"name": "G1 Pop Arte", "url": "https://g1.globo.com/rss/g1/pop-arte/"},
        ],
    }

    @async_retry(max_attempts=3, min_wait=2.0)
    async def _fetch_rss(self, url: str) -> str:
        """Fetch RSS feed with retry."""
        return await self.fetch_url(url)

    def _parse_rss(self, xml_content: str) -> list[dict[str, Any]]:
        """Parse RSS XML into list of articles."""
        articles = []
        try:
            root = ElementTree.fromstring(xml_content)

            # Handle both RSS 2.0 and Atom feeds
            items = root.findall(".//item") or root.findall(
                ".//{http://www.w3.org/2005/Atom}entry"
            )

            for item in items:
                article = self._parse_rss_item(item)
                if article:
                    articles.append(article)

        except ElementTree.ParseError as e:
            self.logger.warning("Failed to parse RSS", error=str(e))

        return articles

    def _parse_rss_item(self, item: ElementTree.Element) -> dict[str, Any] | None:
        """Parse a single RSS item."""
        # RSS 2.0 format
        title = item.findtext("title")
        link = item.findtext("link")
        description = item.findtext("description")
        pub_date = item.findtext("pubDate")
        content = item.findtext("{http://purl.org/rss/1.0/modules/content/}encoded")

        # Atom format fallback
        if not title:
            title = item.findtext("{http://www.w3.org/2005/Atom}title")
        if not link:
            link_elem = item.find("{http://www.w3.org/2005/Atom}link")
            link = link_elem.get("href") if link_elem is not None else None
        if not description:
            description = item.findtext("{http://www.w3.org/2005/Atom}summary")

        if not title or not (description or content):
            return None

        # Clean HTML from description
        text = content or description or ""
        if text:
            soup = BeautifulSoup(text, "html.parser")
            text = soup.get_text(separator=" ", strip=True)

        return {
            "title": title.strip() if title else None,
            "url": link.strip() if link else None,
            "text": text,
            "published_date": self._parse_date(pub_date) if pub_date else None,
        }

    def _parse_date(self, date_str: str) -> datetime | None:
        """Parse various date formats."""
        formats = [
            "%a, %d %b %Y %H:%M:%S %z",
            "%a, %d %b %Y %H:%M:%S %Z",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%d",
        ]
        for fmt in formats:
            try:
                return datetime.strptime(date_str.strip(), fmt)
            except ValueError:
                continue
        return None

    def _generate_id(self, url: str | None, title: str | None) -> str:
        """Generate unique ID for an article."""
        content = f"{url or ''}{title or ''}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    async def collect(
        self,
        topic: str,
        subtopic: str | None = None,
        max_items: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[RawCollectedData]:
        """
        Collect news articles for a topic.

        Args:
            topic: Topic to collect
            subtopic: Optional subtopic (unused for news)
            max_items: Maximum items to collect

        Yields:
            RawCollectedData items
        """
        feeds = self.RSS_FEEDS.get(topic, [])
        if not feeds:
            self.logger.warning("No RSS feeds configured for topic", topic=topic)
            return

        collected = 0
        for feed in feeds:
            if max_items and collected >= max_items:
                break

            self.logger.info(
                "Fetching RSS feed",
                feed=feed["name"],
                url=feed["url"],
            )

            try:
                xml_content = await self._fetch_rss(feed["url"])
                articles = self._parse_rss(xml_content)

                for article in articles:
                    if max_items and collected >= max_items:
                        break

                    item_id = self._generate_id(article["url"], article["title"])

                    # Skip if already collected
                    if self.is_collected(topic, item_id):
                        continue

                    # Skip if text is too short
                    if not article["text"] or len(article["text"]) < 100:
                        continue

                    yield RawCollectedData(
                        id=item_id,
                        source=self.source_name,
                        source_url=article["url"],
                        title=article["title"],
                        text=article["text"],
                        published_date=article["published_date"],
                        topic=topic,
                        subtopic=subtopic,
                        metadata={
                            "feed_name": feed["name"],
                            "feed_url": feed["url"],
                        },
                    )

                    self.mark_collected(topic, item_id)
                    collected += 1

            except Exception as e:
                self.logger.error(
                    "Failed to fetch feed",
                    feed=feed["name"],
                    error=str(e),
                )
                continue

        self.logger.info(
            "News collection completed",
            topic=topic,
            collected=collected,
        )

    async def collect_from_newsapi(
        self,
        topic: str,
        query: str,
        max_items: int = 100,
    ) -> AsyncIterator[RawCollectedData]:
        """
        Collect from NewsAPI (requires API key).

        Args:
            topic: Topic category
            query: Search query
            max_items: Maximum items

        Yields:
            RawCollectedData items
        """
        if not self.settings.news_api_key:
            self.logger.warning("NewsAPI key not configured")
            return

        # Use different rate limit for API
        self.rate_limit_name = "newsapi"

        base_url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "language": "pt",
            "sortBy": "publishedAt",
            "pageSize": min(100, max_items),
            "apiKey": self.settings.news_api_key,
        }

        try:
            data = await self.fetch_json(base_url, params=params)

            for article in data.get("articles", []):
                content = article.get("content") or article.get("description") or ""
                if not content:
                    continue

                item_id = self._generate_id(article.get("url"), article.get("title"))

                if self.is_collected(topic, item_id):
                    continue

                yield RawCollectedData(
                    id=item_id,
                    source=self.source_name,
                    source_url=article.get("url"),
                    title=article.get("title"),
                    text=content,
                    author=article.get("author"),
                    published_date=self._parse_date(article.get("publishedAt", "")),
                    topic=topic,
                    metadata={
                        "source_name": article.get("source", {}).get("name"),
                        "api": "newsapi",
                    },
                )

                self.mark_collected(topic, item_id)

        except Exception as e:
            self.logger.error("NewsAPI error", error=str(e))

        finally:
            # Reset rate limit name
            self.rate_limit_name = "rss"
