import re
from typing import List, Dict, Any
import yaml
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class Guardrails:
    def __init__(self, config_path: str = "configs/topics.yaml"):
        self.financial_blacklist = []
        self.harmful_blacklist = []
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                self.financial_blacklist = config.get("blacklisted_terms", {}).get("financial_advice", [])
                self.harmful_blacklist = config.get("blacklisted_terms", {}).get("harmful", [])
        except Exception as e:
            logger.warning(f"Could not load topics config: {e}. Using defaults.")
            self.financial_blacklist = ["strong buy", "strong sell", "price target", "moon", "pump"]
            self.harmful_blacklist = ["hate", "kill"]

    def check(self, text: str) -> bool:
        """
        Returns True if the content is safe/compliant, False otherwise.
        """
        text_lower = text.lower()
        
        # 1. Financial Advice Check
        for term in self.financial_blacklist:
            # Use word boundaries for stricter matching
            # Escape the term to handle regex special chars if any
            pattern = r'\b' + re.escape(term) + r'\b'
            if re.search(pattern, text_lower):
                logger.debug(f"Guardrail tripped (financial): {term}")
                return False
                
        # 2. Harmful Content Check
        for term in self.harmful_blacklist:
            pattern = r'\b' + re.escape(term) + r'\b'
            if re.search(pattern, text_lower):
                logger.debug(f"Guardrail tripped (harmful): {term}")
                return False
                
        # 3. Specific Regex for price predictions (e.g., "$500 target", "hit $1000")
        if re.search(r'price target of \$\d+', text_lower):
            return False
            
        return True

    def sanitize(self, text: str) -> str:
        return text
