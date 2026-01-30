"""
Academic papers collector for arXiv and Semantic Scholar.
"""

import hashlib
import re
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any
from xml.etree import ElementTree

from ..schemas.dataset import RawCollectedData
from ..utils.retry import async_retry
from .base import AsyncCollector, CollectorRegistry


@CollectorRegistry.register("academic")
class AcademicCollector(AsyncCollector):
    """
    Collector for academic papers from arXiv and Semantic Scholar.
    """

    source_name = "academic"
    rate_limit_name = "arxiv"

    ARXIV_API = "http://export.arxiv.org/api/query"
    SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1"

    # arXiv categories by topic
    ARXIV_CATEGORIES: dict[str, list[str]] = {
        "financeiro": ["q-fin", "econ", "stat.AP"],
        "tecnologia": ["cs.AI", "cs.CL", "cs.LG", "cs.SE", "cs.CR"],
        "ciencias": ["physics", "math", "q-bio", "cond-mat"],
        "saude": ["q-bio", "stat.ME"],
        "humanidades": ["cs.CY", "econ", "physics.soc-ph"],
        "negocios": ["econ", "q-fin", "stat.AP"],
    }

    # Search queries by topic (for papers in Portuguese or about Brazil)
    SEARCH_QUERIES: dict[str, list[str]] = {
        "financeiro": [
            "Brazil economy",
            "Brazilian market",
            "Latin America finance",
            "emerging markets",
        ],
        "tecnologia": [
            "natural language processing Portuguese",
            "machine learning Brazil",
            "artificial intelligence",
        ],
        "ciencias": ["Brazilian physics", "tropical biology", "Amazon"],
        "saude": ["public health Brazil", "tropical medicine", "epidemiology Brazil"],
        "juridico": ["Brazilian law", "legal NLP Portuguese"],
        "humanidades": ["Brazilian history", "Latin American politics"],
        "educacao": ["education Brazil", "Brazilian universities"],
        "meio_ambiente": ["Amazon deforestation", "Brazil climate", "sustainability Brazil"],
    }

    @async_retry(max_attempts=3, min_wait=2.0)
    async def _search_arxiv(
        self,
        query: str,
        categories: list[str] | None = None,
        max_results: int = 50,
        start: int = 0,
    ) -> list[dict[str, Any]]:
        """Search arXiv API."""
        # Build query
        search_query = f"all:{query}"
        if categories:
            cat_query = " OR ".join(f"cat:{cat}" for cat in categories)
            search_query = f"({search_query}) AND ({cat_query})"

        params = {
            "search_query": search_query,
            "start": start,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }

        xml_content = await self.fetch_url(self.ARXIV_API, params=params)
        return self._parse_arxiv_response(xml_content)

    def _parse_arxiv_response(self, xml_content: str) -> list[dict[str, Any]]:
        """Parse arXiv XML response."""
        papers = []
        ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

        try:
            root = ElementTree.fromstring(xml_content)

            for entry in root.findall("atom:entry", ns):
                paper = {
                    "id": entry.findtext("atom:id", "", ns).split("/abs/")[-1],
                    "title": entry.findtext("atom:title", "", ns).strip(),
                    "summary": entry.findtext("atom:summary", "", ns).strip(),
                    "authors": [
                        author.findtext("atom:name", "", ns)
                        for author in entry.findall("atom:author", ns)
                    ],
                    "published": entry.findtext("atom:published", "", ns),
                    "categories": [
                        cat.get("term", "")
                        for cat in entry.findall("atom:category", ns)
                    ],
                    "link": entry.findtext("atom:id", "", ns),
                }
                if paper["id"] and paper["title"]:
                    papers.append(paper)

        except ElementTree.ParseError as e:
            self.logger.warning("Failed to parse arXiv response", error=str(e))

        return papers

    def _clean_abstract(self, text: str) -> str:
        """Clean abstract text."""
        # Remove LaTeX commands
        text = re.sub(r"\$[^$]+\$", "", text)
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _generate_id(self, source: str, paper_id: str) -> str:
        """Generate unique ID."""
        content = f"{source}:{paper_id}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    async def collect(
        self,
        topic: str,
        subtopic: str | None = None,
        max_items: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[RawCollectedData]:
        """
        Collect academic papers for a topic.

        Args:
            topic: Topic to collect
            subtopic: Optional subtopic
            max_items: Maximum items to collect

        Yields:
            RawCollectedData items
        """
        categories = self.ARXIV_CATEGORIES.get(topic, [])
        queries = self.SEARCH_QUERIES.get(topic, [topic])
        max_items = max_items or 100
        collected = 0

        for query in queries:
            if collected >= max_items:
                break

            self.logger.info(
                "Searching arXiv",
                query=query,
                categories=categories,
                topic=topic,
            )

            try:
                papers = await self._search_arxiv(
                    query,
                    categories=categories if categories else None,
                    max_results=min(50, max_items - collected),
                )

                for paper in papers:
                    if collected >= max_items:
                        break

                    paper_id = paper["id"]
                    item_id = self._generate_id("arxiv", paper_id)

                    if self.is_collected(topic, item_id):
                        continue

                    title = paper["title"]
                    abstract = self._clean_abstract(paper["summary"])

                    if not abstract or len(abstract) < 100:
                        continue

                    # Combine title and abstract
                    text = f"{title}\n\n{abstract}"

                    # Parse date
                    published_date = None
                    if paper["published"]:
                        try:
                            published_date = datetime.fromisoformat(
                                paper["published"].replace("Z", "+00:00")
                            )
                        except ValueError:
                            pass

                    yield RawCollectedData(
                        id=item_id,
                        source=self.source_name,
                        source_url=paper["link"],
                        title=title,
                        text=text,
                        summary=abstract,
                        author=", ".join(paper["authors"][:3]),
                        published_date=published_date,
                        topic=topic,
                        subtopic=subtopic,
                        metadata={
                            "arxiv_id": paper_id,
                            "categories": paper["categories"][:5],
                            "all_authors": paper["authors"],
                        },
                    )

                    self.mark_collected(topic, item_id)
                    collected += 1

            except Exception as e:
                self.logger.error(
                    "arXiv search failed",
                    query=query,
                    error=str(e),
                )
                continue

        self.logger.info(
            "Academic collection completed",
            topic=topic,
            collected=collected,
        )

    async def collect_from_semantic_scholar(
        self,
        topic: str,
        query: str,
        max_items: int = 50,
    ) -> AsyncIterator[RawCollectedData]:
        """
        Collect from Semantic Scholar API.

        Args:
            topic: Topic category
            query: Search query
            max_items: Maximum items

        Yields:
            RawCollectedData items
        """
        if not self.settings.semantic_scholar_key:
            self.logger.warning("Semantic Scholar API key not configured")
            return

        self.rate_limit_name = "semantic_scholar"

        headers = {"x-api-key": self.settings.semantic_scholar_key}
        params = {
            "query": query,
            "limit": min(100, max_items),
            "fields": "paperId,title,abstract,authors,year,url,citationCount",
        }

        try:
            url = f"{self.SEMANTIC_SCHOLAR_API}/paper/search"
            data = await self.fetch_json(url, headers=headers, params=params)

            for paper in data.get("data", []):
                paper_id = paper.get("paperId", "")
                item_id = self._generate_id("semantic_scholar", paper_id)

                if self.is_collected(topic, item_id):
                    continue

                title = paper.get("title", "")
                abstract = paper.get("abstract", "")

                if not abstract or len(abstract) < 100:
                    continue

                text = f"{title}\n\n{abstract}"
                authors = paper.get("authors", [])
                author = ", ".join(a.get("name", "") for a in authors[:3])

                yield RawCollectedData(
                    id=item_id,
                    source=self.source_name,
                    source_url=paper.get("url"),
                    title=title,
                    text=text,
                    summary=abstract,
                    author=author,
                    topic=topic,
                    metadata={
                        "semantic_scholar_id": paper_id,
                        "year": paper.get("year"),
                        "citation_count": paper.get("citationCount"),
                        "source_api": "semantic_scholar",
                    },
                )

                self.mark_collected(topic, item_id)

        except Exception as e:
            self.logger.error("Semantic Scholar search failed", error=str(e))

        finally:
            self.rate_limit_name = "arxiv"
