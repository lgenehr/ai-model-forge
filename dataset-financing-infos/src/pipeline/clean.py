import re
import unicodedata
from bs4 import BeautifulSoup

def clean_text(text: str) -> str:
    if not text:
        return ""
    
    # 1. Use BeautifulSoup to strip HTML/XML
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text(separator=" ")
    
    # 2. Unicode normalization (NFKC)
    text = unicodedata.normalize('NFKC', text)
    
    # 3. Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 4. Remove boilerplate
    boilerplate_patterns = [
        r'Copyright © \d{4}',
        r'All rights reserved',
        r'Subscribe to our newsletter',
        r'Read more at'
    ]
    for pattern in boilerplate_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
    return text.strip()
