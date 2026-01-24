import yaml
import json
import os
from typing import List, Dict
from datetime import datetime
from ..connectors.rss import RSSConnector
from ..connectors.wikidata import WikipediaConnector
from ..connectors.mock import MockConnector
from .clean import clean_text
from .guardrails import Guardrails
from .dedupe import Deduplicator
from .ner import MetadataExtractor
from .schema import DatasetRow, DatasetMeta
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class DatasetBuilder:
    def __init__(self, sources_config: str = "configs/sources.yaml", topics_config: str = "configs/topics.yaml"):
        self.sources_config = sources_config
        self.guardrails = Guardrails(topics_config)
        self.metadata_extractor = MetadataExtractor(topics_config)
        self.deduplicator = Deduplicator()
        self.sources = self._load_sources()
        
    def _load_sources(self) -> List[Dict]:
        with open(self.sources_config, 'r') as f:
            config = yaml.safe_load(f)
        return config.get("sources", [])

    def _get_connector(self, source_conf: Dict):
        stype = source_conf.get("type")
        if stype == "rss":
            return RSSConnector(source_conf)
        elif stype == "api_custom" or stype == "wikipedia":
            return WikipediaConnector(source_conf)
        elif stype == "mock":
            return MockConnector(source_conf)
        else:
            return None

    def build(self, years: List[int], output_path: str, max_items: int = 1000):
        logger.info(f"Starting build for years {years}")
        
        count = 0
        stats = {"total_fetched": 0, "cleaned_removed": 0, "guardrail_removed": 0, "duplicate": 0, "saved": 0}
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f_out:
            for source_conf in self.sources:
                if not source_conf.get("enabled", True):
                    continue
                
                connector = self._get_connector(source_conf)
                if not connector:
                    logger.warning(f"No connector for type {source_conf.get('type')}")
                    continue
                
                # Determine time range
                year_start = min(years)
                year_end = max(years)
                
                for item in connector.fetch(year_start, year_end):
                    stats["total_fetched"] += 1
                    
                    # 1. Cleaning
                    clean_content = clean_text(item.get("content", ""))
                    if not clean_content or len(clean_content) < 50: # arbitrary min length
                        stats["cleaned_removed"] += 1
                        continue
                        
                    # 2. Guardrails
                    if not self.guardrails.check(clean_content) or not self.guardrails.check(item.get("title", "")):
                        stats["guardrail_removed"] += 1
                        continue
                        
                    # 3. Deduplication
                    if self.deduplicator.is_duplicate(clean_content):
                        stats["duplicate"] += 1
                        continue
                        
                    # 4. Format
                    # Heuristic for Instruction/Input/Output
                    title = clean_text(item.get("title", "Info"))
                    date_str = item.get("date").strftime("%Y-%m-%d") if isinstance(item.get("date"), datetime) else str(item.get("date"))
                    
                    # Extract meta
                    extracted_topics = self.metadata_extractor.extract_topics(clean_content)
                    extracted_entities = self.metadata_extractor.extract_entities(clean_content)
                    
                    # Merge source provided topics with extracted ones
                    final_topics = list(set(item.get("topics", []) + extracted_topics))
                    
                    # Determine language from source config or item
                    lang = item.get("language") or source_conf.get("language", "en")

                    row = DatasetRow(
                        instruction=f"Explain the event '{title}' from {date_str}.",
                        input="", # For factual Q&A, input can be empty or provide context. Here we treat it as a direct question.
                        output=clean_content,
                        meta=DatasetMeta(
                            source=item.get("source_id", "unknown"),
                            url=item.get("url"),
                            date=date_str,
                            language=lang,
                            topics=final_topics,
                            entities=extracted_entities
                        )
                    )
                    
                    f_out.write(row.model_dump_json() + "\n")
                    stats["saved"] += 1
                    count += 1
                    
                    if count >= max_items:
                        logger.info("Max items reached.")
                        break
                
                if count >= max_items:
                    break

        logger.info(f"Build complete. Stats: {stats}")
        
        # Write report
        report_path = output_path.replace(".jsonl", "_report.json")
        with open(report_path, 'w') as f_rep:
            json.dump(stats, f_rep, indent=2)
