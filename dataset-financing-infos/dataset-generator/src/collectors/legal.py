"""
Legal documents collector for Brazilian legislation and jurisprudence.
"""

import hashlib
import re
from collections.abc import AsyncIterator
from typing import Any

from bs4 import BeautifulSoup

from ..schemas.dataset import RawCollectedData
from ..utils.retry import async_retry
from .base import AsyncCollector, CollectorRegistry


@CollectorRegistry.register("legal")
class LegalCollector(AsyncCollector):
    """
    Collector for Brazilian legal documents from Planalto, STF, and STJ.
    """

    source_name = "legal"
    rate_limit_name = "web_scraping"

    # Planalto URLs for legislation
    PLANALTO_URLS = {
        "constituicao": "https://www.planalto.gov.br/ccivil_03/constituicao/constituicao.htm",
        "codigo_civil": "https://www.planalto.gov.br/ccivil_03/leis/2002/l10406compilada.htm",
        "codigo_penal": "https://www.planalto.gov.br/ccivil_03/decreto-lei/del2848compilado.htm",
        "clt": "https://www.planalto.gov.br/ccivil_03/decreto-lei/del5452.htm",
        "cdc": "https://www.planalto.gov.br/ccivil_03/leis/l8078compilado.htm",
        "cpc": "https://www.planalto.gov.br/ccivil_03/_ato2015-2018/2015/lei/l13105.htm",
    }

    # Keywords by legal subtopic
    LEGAL_KEYWORDS: dict[str, list[str]] = {
        "direito_civil": [
            "contrato",
            "propriedade",
            "obrigação",
            "responsabilidade civil",
            "família",
            "sucessão",
        ],
        "direito_penal": [
            "crime",
            "pena",
            "dolo",
            "culpa",
            "furto",
            "roubo",
            "homicídio",
        ],
        "direito_trabalhista": [
            "CLT",
            "trabalhador",
            "empregador",
            "férias",
            "FGTS",
            "rescisão",
        ],
        "direito_tributario": [
            "imposto",
            "tributo",
            "ICMS",
            "ISS",
            "contribuição",
            "fisco",
        ],
        "direito_constitucional": [
            "constituição",
            "direito fundamental",
            "separação de poderes",
            "federalismo",
        ],
    }

    def _generate_id(self, source: str, identifier: str) -> str:
        """Generate unique ID."""
        content = f"{source}:{identifier}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    @async_retry(max_attempts=3, min_wait=2.0)
    async def _fetch_legislation(self, url: str) -> str:
        """Fetch legislation page with retry."""
        return await self.fetch_url(url)

    def _parse_legislation(self, html: str) -> dict[str, Any]:
        """Parse legislation HTML."""
        soup = BeautifulSoup(html, "html.parser")

        # Try to find the main content
        content = soup.find("div", class_="textoLei") or soup.find("body")

        if not content:
            return {"text": "", "title": ""}

        # Get title
        title_elem = soup.find("h1") or soup.find("title")
        title = title_elem.get_text(strip=True) if title_elem else ""

        # Get text
        text = content.get_text(separator="\n", strip=True)

        # Clean text
        text = self._clean_legal_text(text)

        return {"text": text, "title": title}

    def _clean_legal_text(self, text: str) -> str:
        """Clean legal text."""
        # Remove excessive whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)

        # Remove navigation elements
        text = re.sub(r"Ir para o conteúdo.*?\n", "", text)
        text = re.sub(r"Pular para o conteúdo.*?\n", "", text)

        return text.strip()

    def _extract_articles(self, text: str, max_articles: int = 50) -> list[dict[str, str]]:
        """Extract individual articles from legislation."""
        articles = []

        # Pattern for articles
        pattern = r"(Art\.\s*\d+[º°]?[\s\S]*?)(?=Art\.\s*\d+[º°]?|$)"
        matches = re.findall(pattern, text, re.IGNORECASE)

        for i, match in enumerate(matches[:max_articles]):
            article_text = match.strip()
            if len(article_text) > 50:  # Skip very short fragments
                articles.append(
                    {
                        "number": i + 1,
                        "text": article_text,
                    }
                )

        return articles

    async def collect(
        self,
        topic: str,
        subtopic: str | None = None,
        max_items: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[RawCollectedData]:
        """
        Collect legal documents for a topic.

        Args:
            topic: Topic to collect (should be 'juridico')
            subtopic: Optional legal subtopic
            max_items: Maximum items to collect

        Yields:
            RawCollectedData items
        """
        max_items = max_items or 100
        collected = 0

        # Map subtopics to legislation
        subtopic_to_law = {
            "direito_civil": ["codigo_civil"],
            "direito_penal": ["codigo_penal"],
            "direito_trabalhista": ["clt"],
            "direito_constitucional": ["constituicao"],
        }

        laws_to_collect = subtopic_to_law.get(subtopic, list(self.PLANALTO_URLS.keys()))

        for law_name in laws_to_collect:
            if collected >= max_items:
                break

            url = self.PLANALTO_URLS.get(law_name)
            if not url:
                continue

            self.logger.info(
                "Fetching legislation",
                law=law_name,
                url=url,
            )

            try:
                html = await self._fetch_legislation(url)
                parsed = self._parse_legislation(html)

                if not parsed["text"]:
                    continue

                # Extract individual articles
                articles = self._extract_articles(
                    parsed["text"],
                    max_articles=max_items - collected,
                )

                for article in articles:
                    if collected >= max_items:
                        break

                    item_id = self._generate_id(
                        law_name, f"art_{article['number']}"
                    )

                    if self.is_collected(topic, item_id):
                        continue

                    yield RawCollectedData(
                        id=item_id,
                        source=self.source_name,
                        source_url=url,
                        title=f"{parsed['title']} - Artigo {article['number']}",
                        text=article["text"],
                        topic=topic,
                        subtopic=subtopic or law_name,
                        metadata={
                            "law_name": law_name,
                            "article_number": article["number"],
                            "source_type": "legislation",
                        },
                    )

                    self.mark_collected(topic, item_id)
                    collected += 1

            except Exception as e:
                self.logger.error(
                    "Failed to fetch legislation",
                    law=law_name,
                    error=str(e),
                )
                continue

        self.logger.info(
            "Legal collection completed",
            topic=topic,
            subtopic=subtopic,
            collected=collected,
        )

    async def collect_full_legislation(
        self,
        law_name: str,
        topic: str = "juridico",
    ) -> AsyncIterator[RawCollectedData]:
        """
        Collect full legislation text as a single document.

        Args:
            law_name: Name of the law (e.g., 'constituicao')
            topic: Topic category

        Yields:
            Single RawCollectedData with full text
        """
        url = self.PLANALTO_URLS.get(law_name)
        if not url:
            self.logger.warning("Unknown law", law_name=law_name)
            return

        try:
            html = await self._fetch_legislation(url)
            parsed = self._parse_legislation(html)

            if not parsed["text"]:
                return

            item_id = self._generate_id(law_name, "full")

            if self.is_collected(topic, item_id):
                return

            # Truncate if too long
            text = parsed["text"]
            if len(text) > 100000:
                text = text[:100000] + "..."

            yield RawCollectedData(
                id=item_id,
                source=self.source_name,
                source_url=url,
                title=parsed["title"],
                text=text,
                topic=topic,
                subtopic=law_name,
                metadata={
                    "law_name": law_name,
                    "source_type": "full_legislation",
                    "char_count": len(parsed["text"]),
                },
            )

            self.mark_collected(topic, item_id)

        except Exception as e:
            self.logger.error(
                "Failed to fetch full legislation",
                law=law_name,
                error=str(e),
            )
