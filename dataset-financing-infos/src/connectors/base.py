from abc import ABC, abstractmethod
from typing import List, Dict, Any, Generator
from ..utils.http_client import HttpClient
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class BaseConnector(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client = HttpClient()
        self.source_id = config.get("id", "unknown")

    @abstractmethod
    def fetch(self, year_start: int, year_end: int) -> Generator[Dict[str, Any], None, None]:
        """
        Yields raw data items.
        Each item should at least have:
        - content (str)
        - url (str)
        - date (datetime or str)
        - title (str)
        """
        pass
