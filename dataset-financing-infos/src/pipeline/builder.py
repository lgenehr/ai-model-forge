import yaml
import json
import os
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
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
        stats = {"total_fetched": 0, "cleaned_removed": 0, "guardrail_removed": 0, "duplicate": 0, "saved": 0, "augmented": 0}
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # We need to keep track of saved rows for augmentation
        saved_rows = []

        # Thread safety
        write_lock = threading.Lock()
        dedupe_lock = threading.Lock()

        def process_item(item, source_conf):
            # 1. Cleaning
            clean_content = clean_text(item.get("content", ""))
            if not clean_content or len(clean_content) < 50: # arbitrary min length
                return "cleaned_removed", None

            # 2. Guardrails
            if not self.guardrails.check(clean_content) or not self.guardrails.check(item.get("title", "")):
                return "guardrail_removed", None

            # 3. Deduplication (Thread Safe)
            with dedupe_lock:
                if self.deduplicator.is_duplicate(clean_content):
                    return "duplicate", None

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
            return "saved", row

        with tqdm(total=max_items, desc="Building Dataset", unit="rows") as pbar:
            with open(output_path, 'w', encoding='utf-8') as f_out:
                # Use ThreadPoolExecutor to process items in parallel
                # We use os.cpu_count() or a fixed number depending on needs
                max_workers = os.cpu_count() or 4
                logger.info(f"Using {max_workers} threads for processing.")

                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = []
                    
                    for source_conf in self.sources:
                        if count >= max_items:
                            break

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

                            # Submit task
                            future = executor.submit(process_item, item, source_conf)
                            futures.append(future)

                            # Periodically check for completed futures to manage memory and progress
                            # or strictly speaking, we can just let them accumulate if not too many,
                            # but better to process as they complete or in batches.
                            # However, 'as_completed' blocks. We want to keep fetching while processing.
                            # So we check completion in a non-blocking way or just append?
                            # For simplicity and robust flow, we can use a bounded semaphore or just let futures pile up slightly.
                            # But if we fetch 1000s, we might run out of memory.

                            # Let's prune completed futures
                            done_futures = [f for f in futures if f.done()]
                            for f in done_futures:
                                futures.remove(f)
                                result_type, row = f.result()
                                stats[result_type] += 1

                                if result_type == "saved" and row:
                                    with write_lock:
                                        if count < max_items:
                                            f_out.write(row.model_dump_json() + "\n")
                                            f_out.flush()
                                            saved_rows.append(row)
                                            count += 1
                                            pbar.update(1)

                            if count >= max_items:
                                break

                    # Process remaining futures
                    for f in as_completed(futures):
                        if count >= max_items:
                            break

                        result_type, row = f.result()
                        stats[result_type] += 1

                        if result_type == "saved" and row:
                            with write_lock:
                                if count < max_items:
                                    f_out.write(row.model_dump_json() + "\n")
                                    f_out.flush()
                                    saved_rows.append(row)
                                    count += 1
                                    pbar.update(1)

                # Bonus: Data Augmentation
                if count < max_items:
                    logger.info(f"Target not met ({count}/{max_items}). Starting augmentation...")
                    self._augment_data(saved_rows, f_out, max_items - count, stats, pbar)

        logger.info(f"Build complete. Stats: {stats}")
        
        # Write report
        report_path = output_path.replace(".jsonl", "_report.json")
        with open(report_path, 'w') as f_rep:
            json.dump(stats, f_rep, indent=2)

    def _augment_data(self, existing_rows: List[DatasetRow], f_out, needed: int, stats: Dict, pbar: tqdm):
        """
        Generates synthetic data by recombining existing rows to reach the target.
        """
        if not existing_rows:
            logger.warning("No data to augment from.")
            return

        generated = 0
        while generated < needed:
            # Pick a random source row
            source_row = random.choice(existing_rows)

            # Simple augmentation: Shuffle sentences or create a variation
            original_text = source_row.output
            sentences = original_text.split('. ')
            if len(sentences) > 1:
                random.shuffle(sentences)
                new_text = ". ".join(sentences)
            else:
                new_text = original_text # Fallback

            # Create a new instruction to indicate it's a variation
            new_instruction = source_row.instruction.replace("Explain", "Summarize or review") + " (Augmented)"

            # Avoid duplicate content hash collision by appending a marker
            new_text += " [Synthetically Generated]"

            new_row = DatasetRow(
                instruction=new_instruction,
                input=source_row.input,
                output=new_text,
                meta=source_row.meta
            )

            f_out.write(new_row.model_dump_json() + "\n")
            f_out.flush()
            stats["saved"] += 1
            stats["augmented"] += 1
            generated += 1
            pbar.update(1)
