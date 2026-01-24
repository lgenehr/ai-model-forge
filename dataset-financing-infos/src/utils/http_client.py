import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
from typing import Optional, Dict, Any

class HttpClient:
    def __init__(self, retries: int = 3, backoff_factor: float = 0.3):
        self.session = requests.Session()
        
        retry = Retry(
            total=retries,
            read=retries,
            connect=retries,
            backoff_factor=backoff_factor,
            status_forcelist=(403, 429, 500, 502, 503, 504), # Added 403/429 for aggressive blocking
        )
        
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        
        # Mimic a real browser to avoid 403 Forbidden on RSS feeds
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/rss+xml, application/xml, text/xml, application/atom+xml, */*",
            "Accept-Language": "en-US,en;q=0.9,pt-BR;q=0.8,pt;q=0.7"
        }

    def get(self, url: str, params: Optional[Dict[str, Any]] = None, timeout: int = 15) -> requests.Response:
        try:
            # Allow insecure SSL if absolutely necessary (e.g. some misconfigured gov sites), but prefer verify=True
            # For this dataset project, we'll keep verify=True but catch errors in the connector.
            response = self.session.get(url, params=params, headers=self.headers, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            raise e
