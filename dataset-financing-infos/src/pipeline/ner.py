import yaml
import re
from typing import List, Dict, Set
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class MetadataExtractor:
    def __init__(self, config_path: str = "configs/topics.yaml"):
        self.keyword_map = {}
        self.entities_map = {} # Can be extended
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                self.keyword_map = config.get("keywords", {})
        except Exception as e:
            logger.warning(f"Could not load topics config: {e}")

    def extract_topics(self, text: str) -> List[str]:
        """
        Assigns topics based on keyword occurrence.
        """
        found_topics = set()
        text_lower = text.lower()
        
        for topic, keywords in self.keyword_map.items():
            for kw in keywords:
                # Use regex word boundary for cleaner matching
                # Escape kw to be safe
                pattern = r'\b' + re.escape(kw.lower()) + r'\b'
                if re.search(pattern, text_lower):
                    found_topics.add(topic)
                    break 
        
        return list(found_topics)

    def extract_entities(self, text: str) -> List[str]:
        """
        Simple NER. 
        1. Uses capitalized phrases (heuristic).
        2. Can be improved with spacy or others, but 'simple' requested.
        """
        # Very basic heuristic: Sequences of capitalized words.
        # Excluding common sentence starters is hard without NLP lib, 
        # but we can try to skip the first word of sentences if needed.
        # For this dataset, we'll look for Title Case words inside the sentence.
        
        entities = set()
        
        # Regex for Capitalized Words appearing together
        # Matches "United States", "Apple Inc", "Bitcoin"
        matches = re.findall(r'(?<!^)(?<!\. )[A-Z][a-z]+(?: [A-Z][a-z]+)*', text)
        
        for m in matches:
            if len(m) > 2: # Filter out short noise
                entities.add(m)
                
        return list(entities)
