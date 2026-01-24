import requests
from typing import Generator, Dict, Any, List
from .base import BaseConnector, logger
import datetime
import time

class WikipediaConnector(BaseConnector):
    def fetch(self, year_start: int, year_end: int) -> Generator[Dict[str, Any], None, None]:
        lang = self.config.get("language", "en")
        base_url = f"https://{lang}.wikipedia.org/w/api.php"
        
        # Expanded search terms to get more results
        topics_en = ["Business", "Economy", "Technology", "Science", "Politics", "Election", "Sports", "Climate", "Energy", "Crypto", "Finance", "Health"]
        topics_pt = ["Negócios", "Economia", "Tecnologia", "Ciência", "Política", "Eleição", "Esporte", "Clima", "Energia", "Cripto", "Finanças", "Saúde"]
        
        topics = topics_pt if lang == 'pt' else topics_en
        
        years = range(year_start, year_end + 1)
        
        # Create a product of years and topics + generic "Year events"
        search_queries = []
        for year in years:
            search_queries.append(f"{year}") # Generic year search
            if lang == 'pt':
                search_queries.append(f"eventos de {year}")
            else:
                search_queries.append(f"{year} events")
                
            for topic in topics:
                search_queries.append(f"{topic} {year}")
        
        for term in search_queries:
            params = {
                "action": "query",
                "format": "json",
                "list": "search",
                "srsearch": term,
                "srlimit": 50, # Increased limit
            }
            
            try:
                logger.info(f"Searching Wikipedia ({lang}) for: {term}")
                resp = self.client.get(base_url, params=params)
                data = resp.json()
                
                search_results = data.get("query", {}).get("search", [])
                logger.info(f"Found {len(search_results)} results for {term}")

                for item in search_results:
                    title = item["title"]
                    # Fetch content for this page
                    content_params = {
                        "action": "query",
                        "format": "json",
                        "prop": "extracts",
                        "titles": title,
                        "exintro": True,
                        "explaintext": True
                    }
                    try:
                        c_resp = self.client.get(base_url, params=content_params)
                        c_data = c_resp.json()
                        pages = c_data.get("query", {}).get("pages", {})
                        
                        for page_id, page_info in pages.items():
                            if page_id == "-1": continue
                            
                            extract = page_info.get("extract", "")
                            if not extract:
                                continue

                            # Try to deduce a date. If we searched for "2025", assume 2025.
                            # We can try to parse a date from the title or just default to Jan 1 of the searched year.
                            # Find which year triggered this.
                            found_year = year_start
                            for y in years:
                                if str(y) in term:
                                    found_year = y
                                    break
                            
                            yield {
                                "content": extract,
                                "title": title,
                                "url": f"https://{lang}.wikipedia.org/wiki/{title.replace(' ', '_')}",
                                "date": datetime.datetime(found_year, 1, 1),
                                "source_id": self.source_id,
                                "raw": page_info
                            }
                    except Exception as e:
                        logger.warning(f"Error fetching wiki page {title}: {e}")
                        
                    # Be nice to the API
                    time.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Wiki fetch error for {term}: {e}")
