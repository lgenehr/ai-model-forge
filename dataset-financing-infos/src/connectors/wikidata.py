import requests
import random
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
            offset = 0
            limit = 50
            max_pages = 20  # Fetch up to 1000 items per term (20 * 50)

            for page in range(max_pages):
                params = {
                    "action": "query",
                    "format": "json",
                    "list": "search",
                    "srsearch": term,
                    "srlimit": limit,
                    "sroffset": offset
                }
                
                try:
                    logger.info(f"Searching Wikipedia ({lang}) for: {term} (offset={offset})")
                    resp = self.client.get(base_url, params=params)
                    data = resp.json()

                    search_results = data.get("query", {}).get("search", [])
                    if not search_results:
                        break

                    logger.info(f"Found {len(search_results)} results for {term} (page {page+1})")

                    for item in search_results:
                        title = item["title"]
                        # Fetch content for this page
                        content_params = {
                            "action": "query",
                            "format": "json",
                            "prop": "extracts",
                            "titles": title,
                            "explaintext": True
                            # Removed exintro=True to get full content
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

                                # Generate a random date within that year to avoid all items being Jan 1st
                                start_date = datetime.datetime(found_year, 1, 1)
                                end_date = datetime.datetime(found_year, 12, 31)
                                delta_days = (end_date - start_date).days
                                random_days = random.randint(0, delta_days)
                                final_date = start_date + datetime.timedelta(days=random_days)

                                yield {
                                    "content": extract,
                                    "title": title,
                                    "url": f"https://{lang}.wikipedia.org/wiki/{title.replace(' ', '_')}",
                                    "date": final_date,
                                    "source_id": self.source_id,
                                    "raw": page_info
                                }
                        except Exception as e:
                            logger.warning(f"Error fetching wiki page {title}: {e}")
                            
                        # Be nice to the API
                        time.sleep(0.05)
                    
                    offset += limit
                    time.sleep(0.5)

                except Exception as e:
                    logger.error(f"Wiki fetch error for {term}: {e}")
                    break
