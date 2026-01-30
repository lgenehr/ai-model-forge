"""
Social media collector for Reddit and other platforms.
"""

import hashlib
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any

from ..schemas.dataset import RawCollectedData
from ..utils.retry import async_retry
from .base import AsyncCollector, CollectorRegistry


@CollectorRegistry.register("social_media")
class SocialMediaCollector(AsyncCollector):
    """
    Collector for social media content from Reddit.
    """

    source_name = "social_media"
    rate_limit_name = "reddit"

    # Subreddits by topic
    SUBREDDITS: dict[str, list[str]] = {
        "financeiro": [
            "investimentos",
            "brasil",
            "economia",
            "farialimabets",
            "financaspessoaispt",
        ],
        "tecnologia": [
            "brdev",
            "programacao",
            "tecnologia",
            "linuxbrasil",
            "gamedev",
        ],
        "ciencias": [
            "ciencia",
            "brasil",
        ],
        "saude": [
            "brasil",
            "desabafos",
        ],
        "juridico": [
            "ConsselhosLegais",
            "direito",
            "brasil",
        ],
        "humanidades": [
            "brasil",
            "historia",
            "filosofia_pt",
        ],
        "cultura": [
            "brasil",
            "futebol",
            "livros",
            "filmes",
        ],
        "negocios": [
            "empreendedorismo",
            "brasil",
        ],
        "educacao": [
            "brasil",
            "enem",
            "faculdade",
        ],
        "meio_ambiente": [
            "brasil",
            "meioambiente",
        ],
    }

    REDDIT_API = "https://oauth.reddit.com"
    REDDIT_AUTH = "https://www.reddit.com/api/v1/access_token"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._access_token: str | None = None

    async def setup(self) -> None:
        """Setup with Reddit authentication."""
        await super().setup()
        await self._authenticate()

    async def _authenticate(self) -> None:
        """Authenticate with Reddit API."""
        if not self.settings.reddit_client_id or not self.settings.reddit_client_secret:
            self.logger.warning("Reddit credentials not configured")
            return

        auth = (self.settings.reddit_client_id, self.settings.reddit_client_secret)

        data = {
            "grant_type": "client_credentials",
        }

        headers = {
            "User-Agent": self.settings.reddit_user_agent,
        }

        try:
            async with self.session.post(
                self.REDDIT_AUTH,
                auth=auth,
                data=data,
                headers=headers,
            ) as response:
                response.raise_for_status()
                result = await response.json()
                self._access_token = result.get("access_token")
                self.logger.info("Reddit authentication successful")
        except Exception as e:
            self.logger.error("Reddit authentication failed", error=str(e))

    def _get_headers(self) -> dict[str, str]:
        """Get headers for Reddit API requests."""
        headers = {
            "User-Agent": self.settings.reddit_user_agent,
        }
        if self._access_token:
            headers["Authorization"] = f"Bearer {self._access_token}"
        return headers

    @async_retry(max_attempts=3, min_wait=2.0)
    async def _fetch_subreddit(
        self,
        subreddit: str,
        sort: str = "hot",
        limit: int = 25,
        after: str | None = None,
    ) -> dict[str, Any]:
        """Fetch posts from a subreddit."""
        await self.rate_limiter.acquire(self.rate_limit_name)

        url = f"{self.REDDIT_API}/r/{subreddit}/{sort}"
        params = {"limit": limit}
        if after:
            params["after"] = after

        async with self.session.get(
            url,
            headers=self._get_headers(),
            params=params,
        ) as response:
            response.raise_for_status()
            return await response.json()

    def _generate_id(self, post_id: str) -> str:
        """Generate unique ID."""
        return hashlib.md5(f"reddit:{post_id}".encode()).hexdigest()[:16]

    def _extract_post_data(self, post: dict[str, Any]) -> dict[str, Any] | None:
        """Extract relevant data from a Reddit post."""
        data = post.get("data", {})

        # Skip removed/deleted posts
        if data.get("removed_by_category") or data.get("selftext") == "[deleted]":
            return None

        # Skip posts with very little content
        selftext = data.get("selftext", "")
        if len(selftext) < 50:
            return None

        return {
            "id": data.get("id"),
            "title": data.get("title"),
            "text": selftext,
            "author": data.get("author"),
            "subreddit": data.get("subreddit"),
            "score": data.get("score", 0),
            "num_comments": data.get("num_comments", 0),
            "created_utc": data.get("created_utc"),
            "url": f"https://reddit.com{data.get('permalink', '')}",
        }

    async def collect(
        self,
        topic: str,
        subtopic: str | None = None,
        max_items: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[RawCollectedData]:
        """
        Collect Reddit posts for a topic.

        Args:
            topic: Topic to collect
            subtopic: Optional subtopic
            max_items: Maximum items to collect

        Yields:
            RawCollectedData items
        """
        if not self._access_token:
            self.logger.warning("Reddit not authenticated, skipping")
            return

        subreddits = self.SUBREDDITS.get(topic, ["brasil"])
        max_items = max_items or 100
        collected = 0
        min_score = kwargs.get("min_score", 10)

        for subreddit in subreddits:
            if collected >= max_items:
                break

            self.logger.info(
                "Fetching subreddit",
                subreddit=subreddit,
                topic=topic,
            )

            try:
                # Fetch from different sorts
                for sort in ["hot", "top"]:
                    if collected >= max_items:
                        break

                    data = await self._fetch_subreddit(
                        subreddit,
                        sort=sort,
                        limit=min(25, max_items - collected),
                    )

                    posts = data.get("data", {}).get("children", [])

                    for post in posts:
                        if collected >= max_items:
                            break

                        post_data = self._extract_post_data(post)
                        if not post_data:
                            continue

                        # Filter by score
                        if post_data["score"] < min_score:
                            continue

                        item_id = self._generate_id(post_data["id"])

                        if self.is_collected(topic, item_id):
                            continue

                        # Combine title and text
                        full_text = f"{post_data['title']}\n\n{post_data['text']}"

                        # Parse date
                        published_date = None
                        if post_data["created_utc"]:
                            published_date = datetime.fromtimestamp(
                                post_data["created_utc"]
                            )

                        yield RawCollectedData(
                            id=item_id,
                            source=self.source_name,
                            source_url=post_data["url"],
                            title=post_data["title"],
                            text=full_text,
                            author=post_data["author"],
                            published_date=published_date,
                            topic=topic,
                            subtopic=subtopic,
                            metadata={
                                "subreddit": post_data["subreddit"],
                                "score": post_data["score"],
                                "num_comments": post_data["num_comments"],
                                "reddit_id": post_data["id"],
                            },
                        )

                        self.mark_collected(topic, item_id)
                        collected += 1

            except Exception as e:
                self.logger.error(
                    "Failed to fetch subreddit",
                    subreddit=subreddit,
                    error=str(e),
                )
                continue

        self.logger.info(
            "Social media collection completed",
            topic=topic,
            collected=collected,
        )

    async def collect_comments(
        self,
        post_id: str,
        topic: str,
    ) -> AsyncIterator[RawCollectedData]:
        """
        Collect comments from a specific post.

        Args:
            post_id: Reddit post ID
            topic: Topic category

        Yields:
            RawCollectedData items for each comment
        """
        if not self._access_token:
            return

        try:
            url = f"{self.REDDIT_API}/comments/{post_id}"
            await self.rate_limiter.acquire(self.rate_limit_name)

            async with self.session.get(
                url, headers=self._get_headers()
            ) as response:
                response.raise_for_status()
                data = await response.json()

            # Comments are in the second element
            if len(data) < 2:
                return

            comments = data[1].get("data", {}).get("children", [])

            for comment in comments:
                comment_data = comment.get("data", {})
                body = comment_data.get("body", "")

                if len(body) < 50 or body == "[deleted]":
                    continue

                item_id = self._generate_id(comment_data.get("id", ""))

                if self.is_collected(topic, item_id):
                    continue

                yield RawCollectedData(
                    id=item_id,
                    source=self.source_name,
                    source_url=f"https://reddit.com{comment_data.get('permalink', '')}",
                    title=None,
                    text=body,
                    author=comment_data.get("author"),
                    topic=topic,
                    metadata={
                        "parent_id": post_id,
                        "score": comment_data.get("score", 0),
                        "is_comment": True,
                    },
                )

                self.mark_collected(topic, item_id)

        except Exception as e:
            self.logger.error(
                "Failed to fetch comments",
                post_id=post_id,
                error=str(e),
            )
