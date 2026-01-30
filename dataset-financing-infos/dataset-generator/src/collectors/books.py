"""
Books collector for Project Gutenberg and Open Library.
"""

import hashlib
import re
from collections.abc import AsyncIterator
from typing import Any

from ..schemas.dataset import RawCollectedData
from ..utils.retry import async_retry
from .base import AsyncCollector, CollectorRegistry


@CollectorRegistry.register("books")
class BooksCollector(AsyncCollector):
    """
    Collector for books from Project Gutenberg and Open Library.
    """

    source_name = "books"
    rate_limit_name = "web_scraping"

    GUTENBERG_API = "https://gutendex.com/books"
    OPEN_LIBRARY_API = "https://openlibrary.org"

    # Search terms by topic
    SEARCH_TERMS: dict[str, list[str]] = {
        "financeiro": ["economia", "dinheiro", "comercio", "mercado"],
        "tecnologia": ["ciencia", "tecnica", "invencao"],
        "ciencias": ["fisica", "quimica", "biologia", "matematica", "ciencia"],
        "saude": ["medicina", "saude", "doenca"],
        "juridico": ["direito", "lei", "justica", "constituicao"],
        "humanidades": ["historia", "filosofia", "politica", "sociedade"],
        "cultura": ["literatura", "poesia", "romance", "teatro", "arte"],
        "negocios": ["comercio", "industria", "empresa"],
        "educacao": ["educacao", "ensino", "pedagogia"],
        "meio_ambiente": ["natureza", "ambiente", "ecologia"],
    }

    @async_retry(max_attempts=3, min_wait=2.0)
    async def _search_gutenberg(
        self,
        query: str,
        language: str = "pt",
        page: int = 1,
    ) -> dict[str, Any]:
        """Search Project Gutenberg via Gutendex API."""
        params = {
            "search": query,
            "languages": language,
            "page": page,
        }
        return await self.fetch_json(self.GUTENBERG_API, params=params)

    async def _get_book_text(self, book: dict[str, Any]) -> str | None:
        """Get book text content."""
        formats = book.get("formats", {})

        # Prefer plain text
        text_url = formats.get("text/plain; charset=utf-8") or formats.get(
            "text/plain"
        )

        if not text_url:
            return None

        try:
            text = await self.fetch_url(text_url)
            return self._clean_gutenberg_text(text)
        except Exception as e:
            self.logger.warning(
                "Failed to fetch book text",
                book_id=book.get("id"),
                error=str(e),
            )
            return None

    def _clean_gutenberg_text(self, text: str) -> str:
        """Clean Project Gutenberg text."""
        # Remove header
        start_markers = [
            "*** START OF THIS PROJECT GUTENBERG",
            "*** START OF THE PROJECT GUTENBERG",
            "*END*THE SMALL PRINT",
        ]
        for marker in start_markers:
            if marker in text:
                text = text.split(marker, 1)[-1]

        # Remove footer
        end_markers = [
            "*** END OF THIS PROJECT GUTENBERG",
            "*** END OF THE PROJECT GUTENBERG",
            "End of Project Gutenberg",
        ]
        for marker in end_markers:
            if marker in text:
                text = text.split(marker, 1)[0]

        # Clean up whitespace
        text = re.sub(r"\r\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    def _generate_id(self, source: str, book_id: str) -> str:
        """Generate unique ID."""
        content = f"{source}:{book_id}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    async def collect(
        self,
        topic: str,
        subtopic: str | None = None,
        max_items: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[RawCollectedData]:
        """
        Collect books for a topic.

        Args:
            topic: Topic to collect
            subtopic: Optional subtopic
            max_items: Maximum items to collect

        Yields:
            RawCollectedData items
        """
        search_terms = self.SEARCH_TERMS.get(topic, [topic])
        max_items = max_items or 50
        collected = 0

        for term in search_terms:
            if collected >= max_items:
                break

            self.logger.info(
                "Searching Gutenberg",
                term=term,
                topic=topic,
            )

            try:
                data = await self._search_gutenberg(term)
                books = data.get("results", [])

                for book in books:
                    if collected >= max_items:
                        break

                    book_id = str(book.get("id", ""))
                    item_id = self._generate_id("gutenberg", book_id)

                    if self.is_collected(topic, item_id):
                        continue

                    # Get book text
                    text = await self._get_book_text(book)
                    if not text or len(text) < 1000:
                        continue

                    # Get metadata
                    title = book.get("title", "")
                    authors = book.get("authors", [])
                    author = authors[0].get("name") if authors else None

                    # Truncate very long texts
                    if len(text) > 50000:
                        text = text[:50000] + "..."

                    yield RawCollectedData(
                        id=item_id,
                        source=self.source_name,
                        source_url=f"https://www.gutenberg.org/ebooks/{book_id}",
                        title=title,
                        text=text,
                        author=author,
                        topic=topic,
                        subtopic=subtopic,
                        metadata={
                            "gutenberg_id": book_id,
                            "subjects": book.get("subjects", [])[:5],
                            "languages": book.get("languages", []),
                            "download_count": book.get("download_count"),
                        },
                    )

                    self.mark_collected(topic, item_id)
                    collected += 1

            except Exception as e:
                self.logger.error(
                    "Gutenberg search failed",
                    term=term,
                    error=str(e),
                )
                continue

        self.logger.info(
            "Books collection completed",
            topic=topic,
            collected=collected,
        )

    async def collect_from_open_library(
        self,
        topic: str,
        query: str,
        max_items: int = 20,
    ) -> AsyncIterator[RawCollectedData]:
        """
        Collect from Open Library API.

        Args:
            topic: Topic category
            query: Search query
            max_items: Maximum items

        Yields:
            RawCollectedData items
        """
        params = {
            "q": query,
            "language": "por",
            "limit": max_items,
        }

        try:
            url = f"{self.OPEN_LIBRARY_API}/search.json"
            data = await self.fetch_json(url, params=params)

            for doc in data.get("docs", []):
                work_key = doc.get("key", "")
                item_id = self._generate_id("openlibrary", work_key)

                if self.is_collected(topic, item_id):
                    continue

                title = doc.get("title", "")
                authors = doc.get("author_name", [])
                author = authors[0] if authors else None

                # Open Library doesn't provide full text, only metadata
                # We create a summary from available data
                first_sentence = doc.get("first_sentence", [""])[0] if doc.get("first_sentence") else ""
                description = first_sentence

                if not description or len(description) < 50:
                    continue

                yield RawCollectedData(
                    id=item_id,
                    source=self.source_name,
                    source_url=f"https://openlibrary.org{work_key}",
                    title=title,
                    text=description,
                    author=author,
                    topic=topic,
                    metadata={
                        "open_library_key": work_key,
                        "subjects": doc.get("subject", [])[:5],
                        "first_publish_year": doc.get("first_publish_year"),
                        "source_api": "open_library",
                    },
                )

                self.mark_collected(topic, item_id)

        except Exception as e:
            self.logger.error("Open Library search failed", error=str(e))
