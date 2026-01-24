import feedparser
from typing import Generator, Dict, Any
from dateutil import parser as date_parser
from datetime import datetime
from .base import BaseConnector, logger
import re

class RSSConnector(BaseConnector):
    def _parse_pt_date(self, date_str: str) -> datetime:
        """
        Custom parser for Portuguese dates often found in Brazilian RSS feeds.
        Example: "Sex, 23 Jan 2026 19:57:23 -0300"
        """
        # Map PT months/days to EN
        replacements = {
            'jan': 'Jan', 'fev': 'Feb', 'mar': 'Mar', 'abr': 'Apr', 'mai': 'May', 'jun': 'Jun',
            'jul': 'Jul', 'ago': 'Aug', 'set': 'Sep', 'out': 'Oct', 'nov': 'Nov', 'dez': 'Dec',
            'seg': 'Mon', 'ter': 'Tue', 'qua': 'Wed', 'qui': 'Thu', 'sex': 'Fri', 'sab': 'Sat', 'dom': 'Sun',
            'janeiro': 'January', 'fevereiro': 'February', 'março': 'March', 'abril': 'April',
            'maio': 'May', 'junho': 'June', 'julho': 'July', 'agosto': 'August', 'setembro': 'September',
            'outubro': 'October', 'novembro': 'November', 'dezembro': 'December'
        }
        
        date_str_lower = date_str.lower()
        for pt, en in replacements.items():
            # Replace whole words only to avoid replacing inside other words (though usually dates are structured)
            date_str_lower = date_str_lower.replace(pt, en)
            
        try:
            return date_parser.parse(date_str_lower)
        except Exception:
            raise ValueError(f"Could not parse date: {date_str}")

    def fetch(self, year_start: int, year_end: int) -> Generator[Dict[str, Any], None, None]:
        url = self.config.get("url")
        if not url:
            logger.error(f"No URL for RSS source {self.source_id}")
            return

        logger.info(f"Fetching RSS: {url}")
        try:
            # Use self.client to fetch content (handles User-Agent, retries, etc.)
            response = self.client.get(url)
            content = response.content
            
            # Parse the content
            feed = feedparser.parse(content)
            
            if feed.bozo:
                logger.warning(f"RSS Feed {self.source_id} has malformed XML (bozo exception): {feed.bozo_exception}")
                # We continue anyway, as feedparser often salvages items from broken feeds

            for entry in feed.entries:
                published = entry.get("published") or entry.get("updated")
                dt = None
                
                # Try parsing date
                if published:
                    try:
                        dt = date_parser.parse(published)
                    except Exception:
                        # Try PT fallback
                        try:
                            dt = self._parse_pt_date(published)
                        except Exception as e:
                            logger.debug(f"Date parse failed for {self.source_id} '{published}': {e}")
                
                # If no date, we can optionally use current date if we want "fresh" info, 
                # but for this specific "2025-2026" filter, we must be strict or rely on 'mock' data.
                # If the item has no date, we can't verify it's 2025-2026.
                # HOWEVER, since we are doing a future dataset, if we find NO date, maybe we skip.
                if not dt:
                    continue
                    
                # Check year
                if not (year_start <= dt.year <= year_end):
                    continue

                yield {
                    "content": entry.get("summary", "") or entry.get("description", ""),
                    "title": entry.get("title", ""),
                    "url": entry.get("link", ""),
                    "date": dt,
                    "source_id": self.source_id,
                    "topics": [self.config.get("category", "general")], # Pass source category
                    "language": self.config.get("language", "pt"), # Assume PT for these sources unless specified
                    "raw": entry
                }

        except Exception as e:
            logger.error(f"Failed to fetch RSS {url}: {e}")
