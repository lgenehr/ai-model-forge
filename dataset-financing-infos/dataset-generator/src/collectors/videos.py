"""
Videos collector for YouTube transcriptions.
"""

import hashlib
import re
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any

from ..schemas.dataset import RawCollectedData
from ..utils.retry import async_retry
from .base import AsyncCollector, CollectorRegistry


@CollectorRegistry.register("videos")
class VideosCollector(AsyncCollector):
    """
    Collector for video transcriptions from YouTube.
    """

    source_name = "videos"
    rate_limit_name = "youtube"

    YOUTUBE_API = "https://www.googleapis.com/youtube/v3"

    # Channels by topic (Brazilian educational content)
    CHANNELS: dict[str, list[dict[str, str]]] = {
        "financeiro": [
            {"id": "UC_mSfchV-fgpPy-vuwML8_A", "name": "Me Poupe!"},
            {"id": "UCT4nDeU5pv1XIGySbSK-GgA", "name": "Primo Rico"},
            {"id": "UC8mDF5mWNGE-Kpfcvnn0bUg", "name": "Investidor Sardinha"},
        ],
        "tecnologia": [
            {"id": "UCVHFbqXqoYvEWM1Ddxl0QDg", "name": "Código Fonte TV"},
            {"id": "UCSSLqiuGHK-RQYvU1jUZzaw", "name": "Fabio Akita"},
            {"id": "UCzR2u5RWXWjUh7CwLSvbiA", "name": "Filipe Deschamps"},
        ],
        "ciencias": [
            {"id": "UCKHhA5hN2UohhFDfNXB_cvQ", "name": "Manual do Mundo"},
            {"id": "UCyhnYIvPcwt-MGPeEStFSsg", "name": "Ciência Todo Dia"},
        ],
        "educacao": [
            {"id": "UCg-YSXL8xJRdSbsWZxhUmHw", "name": "Nerdologia"},
            {"id": "UCR3-O2Ar_UL5YMnSPKdcqNw", "name": "Kurzgesagt Português"},
        ],
    }

    # Search queries by topic
    SEARCH_QUERIES: dict[str, list[str]] = {
        "financeiro": [
            "investimentos para iniciantes",
            "como investir dinheiro",
            "educação financeira",
            "renda fixa",
            "bolsa de valores brasil",
        ],
        "tecnologia": [
            "programação python",
            "inteligência artificial tutorial",
            "desenvolvimento web",
            "machine learning brasil",
        ],
        "ciencias": [
            "física explicada",
            "ciência documentário",
            "biologia molecular",
        ],
        "saude": [
            "medicina preventiva",
            "nutrição saudável",
            "saúde mental",
        ],
        "juridico": [
            "direito explicado",
            "leis brasileiras",
            "concurso público direito",
        ],
    }

    @async_retry(max_attempts=3, min_wait=2.0)
    async def _youtube_request(
        self,
        endpoint: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Make YouTube API request with retry."""
        if not self.settings.youtube_api_key:
            raise ValueError("YouTube API key not configured")

        params["key"] = self.settings.youtube_api_key
        url = f"{self.YOUTUBE_API}/{endpoint}"
        return await self.fetch_json(url, params=params)

    async def _search_videos(
        self,
        query: str,
        max_results: int = 25,
        published_after: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search for videos."""
        params = {
            "part": "snippet",
            "q": query,
            "type": "video",
            "maxResults": max_results,
            "relevanceLanguage": "pt",
            "regionCode": "BR",
            "videoCaption": "closedCaption",  # Only videos with captions
        }

        if published_after:
            params["publishedAfter"] = published_after

        try:
            data = await self._youtube_request("search", params)
            return data.get("items", [])
        except Exception as e:
            self.logger.warning("YouTube search failed", error=str(e))
            return []

    async def _get_channel_videos(
        self,
        channel_id: str,
        max_results: int = 25,
    ) -> list[dict[str, Any]]:
        """Get videos from a channel."""
        # First get the uploads playlist
        params = {
            "part": "contentDetails",
            "id": channel_id,
        }

        try:
            data = await self._youtube_request("channels", params)
            items = data.get("items", [])
            if not items:
                return []

            uploads_playlist = (
                items[0]
                .get("contentDetails", {})
                .get("relatedPlaylists", {})
                .get("uploads")
            )

            if not uploads_playlist:
                return []

            # Get videos from playlist
            params = {
                "part": "snippet",
                "playlistId": uploads_playlist,
                "maxResults": max_results,
            }

            data = await self._youtube_request("playlistItems", params)
            return data.get("items", [])

        except Exception as e:
            self.logger.warning(
                "Failed to get channel videos",
                channel_id=channel_id,
                error=str(e),
            )
            return []

    async def _get_video_details(self, video_id: str) -> dict[str, Any] | None:
        """Get video details including description."""
        params = {
            "part": "snippet,contentDetails,statistics",
            "id": video_id,
        }

        try:
            data = await self._youtube_request("videos", params)
            items = data.get("items", [])
            return items[0] if items else None
        except Exception as e:
            self.logger.warning(
                "Failed to get video details",
                video_id=video_id,
                error=str(e),
            )
            return None

    async def _get_captions(self, video_id: str) -> str | None:
        """
        Get video captions/subtitles.

        Note: YouTube Data API doesn't provide caption text directly.
        This method returns the video description as a fallback.
        For actual captions, you would need to use youtube-transcript-api
        or similar library.
        """
        # The YouTube API requires OAuth for caption download
        # For now, we use the video description + metadata
        # In production, use youtube_transcript_api library
        self.logger.debug(
            "Caption extraction requires external library",
            video_id=video_id,
        )
        return None

    def _generate_id(self, video_id: str) -> str:
        """Generate unique ID."""
        return hashlib.md5(f"youtube:{video_id}".encode()).hexdigest()[:16]

    def _parse_duration(self, duration: str) -> int:
        """Parse ISO 8601 duration to seconds."""
        # PT#H#M#S format
        pattern = r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?"
        match = re.match(pattern, duration)
        if not match:
            return 0

        hours = int(match.group(1) or 0)
        minutes = int(match.group(2) or 0)
        seconds = int(match.group(3) or 0)

        return hours * 3600 + minutes * 60 + seconds

    async def collect(
        self,
        topic: str,
        subtopic: str | None = None,
        max_items: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[RawCollectedData]:
        """
        Collect video metadata and descriptions for a topic.

        Args:
            topic: Topic to collect
            subtopic: Optional subtopic
            max_items: Maximum items to collect

        Yields:
            RawCollectedData items
        """
        if not self.settings.youtube_api_key:
            self.logger.warning("YouTube API key not configured, skipping")
            return

        max_items = max_items or 50
        collected = 0

        # First collect from configured channels
        channels = self.CHANNELS.get(topic, [])

        for channel in channels:
            if collected >= max_items:
                break

            self.logger.info(
                "Fetching channel videos",
                channel=channel["name"],
                topic=topic,
            )

            videos = await self._get_channel_videos(
                channel["id"],
                max_results=min(25, max_items - collected),
            )

            for video_item in videos:
                if collected >= max_items:
                    break

                snippet = video_item.get("snippet", {})
                resource_id = snippet.get("resourceId", {})
                video_id = resource_id.get("videoId")

                if not video_id:
                    continue

                item_id = self._generate_id(video_id)

                if self.is_collected(topic, item_id):
                    continue

                # Get full video details
                details = await self._get_video_details(video_id)
                if not details:
                    continue

                full_snippet = details.get("snippet", {})
                title = full_snippet.get("title", "")
                description = full_snippet.get("description", "")

                # Skip if description is too short
                if len(description) < 100:
                    continue

                # Combine title and description
                text = f"{title}\n\n{description}"

                # Parse date
                published_date = None
                pub_str = full_snippet.get("publishedAt")
                if pub_str:
                    try:
                        published_date = datetime.fromisoformat(
                            pub_str.replace("Z", "+00:00")
                        )
                    except ValueError:
                        pass

                # Get statistics
                stats = details.get("statistics", {})
                content_details = details.get("contentDetails", {})

                yield RawCollectedData(
                    id=item_id,
                    source=self.source_name,
                    source_url=f"https://www.youtube.com/watch?v={video_id}",
                    title=title,
                    text=text,
                    author=full_snippet.get("channelTitle"),
                    published_date=published_date,
                    topic=topic,
                    subtopic=subtopic,
                    metadata={
                        "video_id": video_id,
                        "channel_id": channel["id"],
                        "channel_name": channel["name"],
                        "view_count": int(stats.get("viewCount", 0)),
                        "like_count": int(stats.get("likeCount", 0)),
                        "comment_count": int(stats.get("commentCount", 0)),
                        "duration_seconds": self._parse_duration(
                            content_details.get("duration", "")
                        ),
                        "tags": full_snippet.get("tags", [])[:10],
                    },
                )

                self.mark_collected(topic, item_id)
                collected += 1

        # Search for additional videos if needed
        if collected < max_items:
            queries = self.SEARCH_QUERIES.get(topic, [topic])

            for query in queries:
                if collected >= max_items:
                    break

                self.logger.info(
                    "Searching YouTube",
                    query=query,
                    topic=topic,
                )

                videos = await self._search_videos(
                    query,
                    max_results=min(25, max_items - collected),
                )

                for video_item in videos:
                    if collected >= max_items:
                        break

                    video_id = video_item.get("id", {}).get("videoId")
                    if not video_id:
                        continue

                    item_id = self._generate_id(video_id)

                    if self.is_collected(topic, item_id):
                        continue

                    details = await self._get_video_details(video_id)
                    if not details:
                        continue

                    full_snippet = details.get("snippet", {})
                    title = full_snippet.get("title", "")
                    description = full_snippet.get("description", "")

                    if len(description) < 100:
                        continue

                    text = f"{title}\n\n{description}"

                    published_date = None
                    pub_str = full_snippet.get("publishedAt")
                    if pub_str:
                        try:
                            published_date = datetime.fromisoformat(
                                pub_str.replace("Z", "+00:00")
                            )
                        except ValueError:
                            pass

                    stats = details.get("statistics", {})

                    yield RawCollectedData(
                        id=item_id,
                        source=self.source_name,
                        source_url=f"https://www.youtube.com/watch?v={video_id}",
                        title=title,
                        text=text,
                        author=full_snippet.get("channelTitle"),
                        published_date=published_date,
                        topic=topic,
                        subtopic=subtopic,
                        metadata={
                            "video_id": video_id,
                            "search_query": query,
                            "view_count": int(stats.get("viewCount", 0)),
                            "like_count": int(stats.get("likeCount", 0)),
                        },
                    )

                    self.mark_collected(topic, item_id)
                    collected += 1

        self.logger.info(
            "Videos collection completed",
            topic=topic,
            collected=collected,
        )
