"""
Wikipedia collector for encyclopedia content.
"""

import hashlib
import re
from collections.abc import AsyncIterator
from typing import Any

from ..schemas.dataset import RawCollectedData
from ..utils.retry import async_retry
from .base import AsyncCollector, CollectorRegistry


@CollectorRegistry.register("encyclopedia")
class WikipediaCollector(AsyncCollector):
    """
    Collector for Wikipedia articles in Portuguese.
    """

    source_name = "encyclopedia"
    rate_limit_name = "wikipedia"

    WIKIPEDIA_API = "https://pt.wikipedia.org/w/api.php"

    # Categories by topic
    CATEGORIES: dict[str, list[str]] = {
        "financeiro": [
            "Economia",
            "Economia_do_Brasil",
            "Mercado_financeiro",
            "Investimentos",
            "Bolsas_de_valores",
            "Criptomoedas",
            "Bancos_do_Brasil",
        ],
        "tecnologia": [
            "Ciência_da_computação",
            "Programação_de_computadores",
            "Inteligência_artificial",
            "Aprendizado_de_máquina",
            "Desenvolvimento_de_software",
            "Segurança_da_informação",
        ],
        "ciencias": [
            "Física",
            "Química",
            "Biologia",
            "Matemática",
            "Astronomia",
            "Ciências_naturais",
        ],
        "saude": [
            "Medicina",
            "Saúde",
            "Doenças",
            "Nutrição",
            "Psicologia",
            "Farmacologia",
        ],
        "juridico": [
            "Direito_do_Brasil",
            "Direito_civil",
            "Direito_penal",
            "Direito_constitucional",
            "Legislação_do_Brasil",
            "Tribunais_do_Brasil",
        ],
        "humanidades": [
            "História_do_Brasil",
            "Filosofia",
            "Sociologia",
            "Antropologia",
            "Ciência_política",
        ],
        "cultura": [
            "Literatura_do_Brasil",
            "Arte_do_Brasil",
            "Música_do_Brasil",
            "Cinema_do_Brasil",
            "Culinária_do_Brasil",
        ],
        "negocios": [
            "Administração",
            "Empreendedorismo",
            "Marketing",
            "Gestão_empresarial",
            "Empresas_do_Brasil",
        ],
        "educacao": [
            "Educação_no_Brasil",
            "Universidades_do_Brasil",
            "Pedagogia",
            "Ensino",
        ],
        "meio_ambiente": [
            "Meio_ambiente_do_Brasil",
            "Sustentabilidade",
            "Energia_renovável",
            "Conservação_da_natureza",
            "Biomas_do_Brasil",
        ],
    }

    @async_retry(max_attempts=3, min_wait=1.0)
    async def _api_request(self, params: dict[str, Any]) -> dict[str, Any]:
        """Make Wikipedia API request with retry."""
        params["format"] = "json"
        return await self.fetch_json(self.WIKIPEDIA_API, params=params)

    async def _get_category_members(
        self,
        category: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Get articles in a category."""
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Categoria:{category}",
            "cmlimit": limit,
            "cmtype": "page",
        }

        try:
            data = await self._api_request(params)
            return data.get("query", {}).get("categorymembers", [])
        except Exception as e:
            self.logger.warning(
                "Failed to get category members",
                category=category,
                error=str(e),
            )
            return []

    async def _get_random_articles(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get random articles."""
        params = {
            "action": "query",
            "list": "random",
            "rnnamespace": 0,
            "rnlimit": limit,
        }

        try:
            data = await self._api_request(params)
            return data.get("query", {}).get("random", [])
        except Exception as e:
            self.logger.warning("Failed to get random articles", error=str(e))
            return []

    async def _get_article_content(self, title: str) -> dict[str, Any] | None:
        """Get full article content."""
        params = {
            "action": "query",
            "titles": title,
            "prop": "extracts|info|categories",
            "explaintext": True,
            "exsectionformat": "plain",
            "inprop": "url",
        }

        try:
            data = await self._api_request(params)
            pages = data.get("query", {}).get("pages", {})

            for page_id, page in pages.items():
                if page_id == "-1":
                    return None

                return {
                    "title": page.get("title"),
                    "text": page.get("extract", ""),
                    "url": page.get("fullurl"),
                    "page_id": page_id,
                    "categories": [
                        c.get("title", "").replace("Categoria:", "")
                        for c in page.get("categories", [])
                    ],
                }

        except Exception as e:
            self.logger.warning(
                "Failed to get article",
                title=title,
                error=str(e),
            )

        return None

    def _clean_text(self, text: str) -> str:
        """Clean Wikipedia text."""
        # Remove section headers artifacts
        text = re.sub(r"={2,}[^=]+={2,}", "", text)
        # Remove multiple newlines
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Remove references markers
        text = re.sub(r"\[\d+\]", "", text)
        return text.strip()

    def _generate_id(self, page_id: str, title: str) -> str:
        """Generate unique ID."""
        content = f"wikipedia:{page_id}:{title}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    async def collect(
        self,
        topic: str,
        subtopic: str | None = None,
        max_items: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[RawCollectedData]:
        """
        Collect Wikipedia articles for a topic.

        Args:
            topic: Topic to collect
            subtopic: Optional subtopic
            max_items: Maximum items to collect

        Yields:
            RawCollectedData items
        """
        categories = self.CATEGORIES.get(topic, [])
        if not categories:
            self.logger.warning("No categories configured for topic", topic=topic)
            # Fall back to random articles
            categories = []

        collected = 0
        max_items = max_items or 100
        articles_per_category = max(10, max_items // max(1, len(categories)))

        # Collect from categories
        for category in categories:
            if collected >= max_items:
                break

            self.logger.info(
                "Fetching Wikipedia category",
                category=category,
                topic=topic,
            )

            members = await self._get_category_members(
                category, limit=articles_per_category
            )

            for member in members:
                if collected >= max_items:
                    break

                title = member.get("title")
                if not title:
                    continue

                page_id = str(member.get("pageid", ""))
                item_id = self._generate_id(page_id, title)

                if self.is_collected(topic, item_id):
                    continue

                article = await self._get_article_content(title)
                if not article or not article["text"]:
                    continue

                text = self._clean_text(article["text"])

                # Skip if too short
                if len(text) < 200:
                    continue

                yield RawCollectedData(
                    id=item_id,
                    source=self.source_name,
                    source_url=article["url"],
                    title=article["title"],
                    text=text,
                    topic=topic,
                    subtopic=subtopic,
                    metadata={
                        "page_id": article["page_id"],
                        "categories": article["categories"][:5],
                        "source_category": category,
                    },
                )

                self.mark_collected(topic, item_id)
                collected += 1

        # Add random articles if we haven't reached max
        if collected < max_items:
            remaining = max_items - collected
            self.logger.info(
                "Fetching random Wikipedia articles",
                count=remaining,
            )

            random_articles = await self._get_random_articles(limit=remaining)

            for member in random_articles:
                if collected >= max_items:
                    break

                title = member.get("title")
                if not title:
                    continue

                page_id = str(member.get("id", ""))
                item_id = self._generate_id(page_id, title)

                if self.is_collected(topic, item_id):
                    continue

                article = await self._get_article_content(title)
                if not article or not article["text"]:
                    continue

                text = self._clean_text(article["text"])

                if len(text) < 200:
                    continue

                yield RawCollectedData(
                    id=item_id,
                    source=self.source_name,
                    source_url=article["url"],
                    title=article["title"],
                    text=text,
                    topic=topic,
                    subtopic=subtopic,
                    metadata={
                        "page_id": article["page_id"],
                        "categories": article.get("categories", [])[:5],
                        "random": True,
                    },
                )

                self.mark_collected(topic, item_id)
                collected += 1

        self.logger.info(
            "Wikipedia collection completed",
            topic=topic,
            collected=collected,
        )
