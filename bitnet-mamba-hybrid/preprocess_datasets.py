#!/usr/bin/env python3
"""
Dataset Preprocessing Script for BitNet-Mamba Hybrid Training

Downloads, tokenizes, and saves datasets as memory-mapped files for efficient training.
Eliminates HTTP requests during training by pre-processing all data.

Datasets:
- English: HuggingFaceFW/fineweb-edu (sample-10BT split)
- Portuguese: Multiple high-quality Brazilian Portuguese sources with automatic fallback:
  1. FineWeb2 Portuguese - Multilingual fineweb (curated, high-quality)
  2. Wikipedia Portuguese - Encyclopedic content
  3. Portuguese-PD - Largest Portuguese open corpus (public domain)
  4. CulturaX Portuguese - High-quality web corpus
  5. BrWaC - Brazilian Web as Corpus (3.53M docs, 2.68B tokens)
  6. Quati - Unicamp Brazilian Portuguese dataset
  7. CC-100, mC4, News, Carolina Corpus - Additional sources

Features:
- Saves progress in chunks (resilient to interruptions)
- Can merge chunks from interrupted downloads with --merge_chunks
- Graceful shutdown on Ctrl+C (saves current progress)
- Validates already processed datasets (skips if tokens.bin exists)
- Automatic fallback if primary Portuguese sources are unavailable

Usage:
    # Normal preprocessing
    python preprocess_datasets.py --output_dir ./data/tokenized

    # Merge chunks from interrupted download
    python preprocess_datasets.py --output_dir ./data/tokenized --merge_chunks

    # Force reprocessing even if tokens.bin exists
    python preprocess_datasets.py --output_dir ./data/tokenized --force

    # Combine multiple Portuguese sources for diversity
    python preprocess_datasets.py --pt_combined_sources 3
"""

import os
import sys
import signal
import argparse
import logging
import json
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Iterator
from dataclasses import dataclass

import numpy as np
import torch
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Global flag for graceful shutdown
_shutdown_requested = False
_current_tokens = []
_current_output_path = None
_current_chunk_idx = 0


def signal_handler(signum, frame):
    """Handle interrupt signals gracefully"""
    global _shutdown_requested
    if _shutdown_requested:
        logger.warning("\nForced shutdown requested. Exiting immediately...")
        sys.exit(1)

    _shutdown_requested = True
    logger.warning("\n" + "=" * 60)
    logger.warning("Interrupt received! Saving current progress...")
    logger.warning("Press Ctrl+C again to force quit (may lose unsaved data)")
    logger.warning("=" * 60)

    # Save current tokens if any
    if _current_tokens and _current_output_path:
        save_emergency_chunk()


def save_emergency_chunk():
    """Save current tokens as an emergency chunk"""
    global _current_tokens, _current_output_path, _current_chunk_idx

    if not _current_tokens or not _current_output_path:
        return

    chunk_file = _current_output_path / f"tokens_chunk_{_current_chunk_idx:04d}.bin"
    logger.info(f"Saving emergency chunk: {chunk_file}")

    try:
        dtype = np.uint16 if max(_current_tokens) < 65535 else np.uint32
        tokens_array = np.array(_current_tokens, dtype=dtype)

        memmap = np.memmap(
            chunk_file,
            dtype=dtype,
            mode='w+',
            shape=tokens_array.shape
        )
        memmap[:] = tokens_array
        memmap.flush()
        del memmap

        logger.info(f"Saved {len(_current_tokens):,} tokens to emergency chunk")

        # Save chunk progress metadata
        save_chunk_progress(_current_output_path, _current_chunk_idx + 1)

    except Exception as e:
        logger.error(f"Failed to save emergency chunk: {e}")


def save_chunk_progress(output_path: Path, num_chunks: int):
    """Save progress metadata for resuming later"""
    progress_file = output_path / "chunk_progress.json"
    progress = {
        "num_chunks": num_chunks,
        "status": "interrupted",
        "message": "Use --merge_chunks to complete processing"
    }

    try:
        with open(progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
        logger.info(f"Progress saved to {progress_file}")
    except Exception as e:
        logger.error(f"Failed to save progress: {e}")


@dataclass
class PreprocessConfig:
    """Configuration for dataset preprocessing"""
    output_dir: str = "./data/tokenized"
    max_seq_len: int = 2048
    vocab_size: int = 50304

    # Dataset limits (None = process all)
    en_max_samples: Optional[int] = None
    pt_max_samples: Optional[int] = None

    # Tokenization batch size
    tokenizer_batch_size: int = 1000

    # Number of tokens to accumulate before saving a chunk
    tokens_per_chunk: int = 100_000_000  # 100M tokens per chunk


def get_tokenizer():
    """Initialize GPT2TokenizerFast with proper configuration"""
    try:
        from transformers import GPT2TokenizerFast

        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        # Set padding token to eos token
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        logger.info(f"Loaded GPT2TokenizerFast (vocab_size={tokenizer.vocab_size})")
        return tokenizer
    except ImportError:
        raise ImportError("transformers library required: pip install transformers")


def check_dataset_already_processed(output_path: Path) -> Tuple[bool, Optional[int]]:
    """
    Check if a dataset has already been processed by validating tokens.bin.

    Args:
        output_path: Path to the dataset directory (e.g., ./data/tokenized/en)

    Returns:
        Tuple of (is_processed, token_count)
    """
    tokens_file = output_path / "tokens.bin"
    metadata_file = output_path / "metadata.json"

    if not tokens_file.exists():
        return False, None

    # Validate the tokens.bin file
    try:
        file_size = tokens_file.stat().st_size
        if file_size == 0:
            logger.warning(f"tokens.bin exists but is empty at {output_path}")
            return False, None

        # Check metadata
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                token_count = metadata.get('total_tokens', 0)

                if token_count > 0:
                    # Validate file size matches expected
                    expected_size = token_count * 2  # uint16 = 2 bytes
                    if abs(file_size - expected_size) < 100:  # Allow small tolerance
                        return True, token_count
                    else:
                        logger.warning(f"tokens.bin size mismatch at {output_path}")
                        logger.warning(f"  Expected: {expected_size:,} bytes, Got: {file_size:,} bytes")

        # If no valid metadata, estimate from file size
        estimated_tokens = file_size // 2  # Assume uint16
        if estimated_tokens > 1000:  # At least 1000 tokens
            return True, estimated_tokens

    except Exception as e:
        logger.warning(f"Error validating {tokens_file}: {e}")

    return False, None


def load_english_dataset(max_samples: Optional[int] = None):
    """Load English dataset from HuggingFace"""
    try:
        from datasets import load_dataset

        logger.info("Loading English dataset: HuggingFaceFW/fineweb-edu (sample-10BT)")

        # Load streaming dataset to handle large size
        dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split="train",
            streaming=True
        )

        logger.info("Successfully loaded fineweb-edu dataset (streaming mode)")
        return dataset, "text"

    except Exception as e:
        logger.warning(f"Could not load fineweb-edu: {e}")
        logger.info("Falling back to wikitext-103")

        try:
            from datasets import load_dataset

            dataset = load_dataset(
                "wikitext",
                "wikitext-103-raw-v1",
                split="train",
                streaming=True
            )
            return dataset, "text"
        except Exception as e2:
            logger.warning(f"Could not load wikitext: {e2}")
            logger.info("Falling back to TinyStories")

            dataset = load_dataset(
                "roneneldan/TinyStories",
                split="train",
                streaming=True
            )
            return dataset, "text"


# =============================================================================
# Portuguese Dataset Sources - Multiple High-Quality Sources
# =============================================================================
# Multiple high-quality Portuguese sources including FineWeb2, Wikipedia,
# Portuguese-PD, CulturaX, BrWaC, and Quati with automatic fallback
# =============================================================================

PORTUGUESE_SOURCES = [
    {
        "name": "FineWeb2 Portuguese",
        "dataset_id": "HuggingFaceFW/fineweb-2",
        "config": "por_Latn",
        "split": "train",
        "text_field": "text",
        "priority": 1,
        "description": "FineWeb2 - Multilingual high-quality curated web corpus (Portuguese subset, similar to fineweb-edu)"
    },
    {
        "name": "Wikipedia Portuguese",
        "dataset_id": "wikimedia/wikipedia",
        "config": "20231101.pt",
        "split": "train",
        "text_field": "text",
        "priority": 2,
        "description": "Portuguese Wikipedia - High quality encyclopedic content"
    },
    {
        "name": "Portuguese-PD (Public Domain)",
        "dataset_id": "PleIAs/Portuguese-PD",
        "config": None,
        "split": "train",
        "text_field": "text",
        "priority": 3,
        "description": "Portuguese-PD - Largest Portuguese open corpus with public domain content"
    },
    {
        "name": "CulturaX Portuguese",
        "dataset_id": "uonlp/CulturaX",
        "config": "pt",
        "split": "train",
        "text_field": "text",
        "priority": 4,
        "description": "CulturaX - High quality multilingual web corpus"
    },
    {
        "name": "BrWaC (Brazilian Web as Corpus)",
        "dataset_id": "UFRGS/brwac",
        "config": None,
        "split": "train",
        "text_field": "text",
        "priority": 5,
        "description": "BrWaC - Large Brazilian Portuguese web corpus (3.53M documents, 2.68B tokens)"
    },
    {
        "name": "Quati (Unicamp)",
        "dataset_id": "unicamp-dl/quati",
        "config": None,
        "split": "train",
        "text_field": "text",
        "priority": 6,
        "description": "Quati - High-quality Brazilian Portuguese dataset from Unicamp"
    },
    {
        "name": "CC-100 Portuguese",
        "dataset_id": "cc100",
        "config": "pt",
        "split": "train",
        "text_field": "text",
        "priority": 7,
        "description": "CC-100 - CommonCrawl based corpus"
    },
    {
        "name": "MC4 Portuguese",
        "dataset_id": "mc4",
        "config": "pt",
        "split": "train",
        "text_field": "text",
        "priority": 8,
        "description": "mC4 - Multilingual C4 Portuguese subset"
    },
    {
        "name": "BrWac Sample (legacy)",
        "dataset_id": "eduagarcia/brwac",
        "config": None,
        "split": "train",
        "text_field": "text",
        "priority": 9,
        "description": "Brazilian Web as Corpus - Brazilian Portuguese web content (legacy sample)"
    },
    {
        "name": "Portuguese News",
        "dataset_id": "recogna-nlp/publico-news",
        "config": None,
        "split": "train",
        "text_field": "content",
        "priority": 10,
        "description": "Portuguese news articles"
    },
    {
        "name": "Carolina Corpus",
        "dataset_id": "carolina-c4ai/corpus-carolina",
        "config": None,
        "split": "train",
        "text_field": "text",
        "priority": 11,
        "description": "Carolina Corpus - Brazilian Portuguese reference corpus"
    },
]


def load_portuguese_dataset(max_samples: Optional[int] = None) -> Tuple[Optional[Iterator], Optional[str], Optional[str]]:
    """
    Load Portuguese dataset from multiple high-quality sources.

    Tries multiple sources in order of priority until one works.

    Returns:
        Tuple of (dataset_iterator, text_field, source_name) or (None, None, None) if all fail
    """
    from datasets import load_dataset

    logger.info("=" * 60)
    logger.info("Loading Portuguese/Brazilian Portuguese Datasets")
    logger.info("=" * 60)

    # Try each source in priority order
    for source in sorted(PORTUGUESE_SOURCES, key=lambda x: x["priority"]):
        try:
            logger.info(f"\nTrying: {source['name']}")
            logger.info(f"  Dataset: {source['dataset_id']}")
            logger.info(f"  Description: {source['description']}")

            # Build load_dataset arguments
            load_args = {
                "path": source["dataset_id"],
                "split": source["split"],
                "streaming": True,
            }

            if source["config"]:
                load_args["name"] = source["config"]

            dataset = load_dataset(**load_args)

            # Test that we can iterate and get text
            test_iter = iter(dataset)
            test_sample = next(test_iter)

            # Find the text field
            text_field = source["text_field"]
            if text_field not in test_sample:
                # Try to find an alternative text field
                for alt_field in ['text', 'content', 'sentence', 'document', 'article', 'body']:
                    if alt_field in test_sample:
                        text_field = alt_field
                        break
                else:
                    logger.warning(f"  Could not find text field. Available: {list(test_sample.keys())}")
                    continue

            test_text = test_sample.get(text_field, "")
            if not test_text or len(str(test_text).strip()) < 10:
                logger.warning(f"  Text field '{text_field}' is empty or too short")
                continue

            logger.info(f"  ✓ Successfully loaded!")
            logger.info(f"  Text field: {text_field}")
            logger.info(f"  Sample preview: {str(test_text)[:100]}...")

            # Return a fresh iterator (the test consumed one sample)
            dataset = load_dataset(**load_args)
            return iter(dataset), text_field, source['name']

        except Exception as e:
            logger.warning(f"  ✗ Failed: {str(e)[:100]}")
            continue

    # If all primary sources fail, try a simple Wikipedia fallback
    logger.warning("\nAll primary sources failed. Trying simple Wikipedia fallback...")

    try:
        dataset = load_dataset(
            "wikipedia",
            "20220301.pt",
            split="train",
            streaming=True,
        )

        test_iter = iter(dataset)
        test_sample = next(test_iter)

        text_field = "text"
        if text_field in test_sample and len(test_sample[text_field]) > 10:
            logger.info("✓ Wikipedia 20220301.pt fallback successful!")
            dataset = load_dataset(
                "wikipedia",
                "20220301.pt",
                split="train",
                streaming=True,
            )
            return iter(dataset), text_field, "Wikipedia PT (fallback)"

    except Exception as e:
        logger.warning(f"Wikipedia fallback also failed: {e}")

    logger.error("=" * 60)
    logger.error("FAILED: Could not load any Portuguese dataset!")
    logger.error("Please check your internet connection and HuggingFace access.")
    logger.error("=" * 60)

    return None, None, None


def create_combined_portuguese_iterator(
    max_samples: Optional[int] = None,
    sources_to_use: int = 3
) -> Tuple[Optional[Iterator], str]:
    """
    Create a combined iterator from multiple Portuguese sources.

    This interleaves samples from multiple sources for better diversity.

    Args:
        max_samples: Maximum samples to process
        sources_to_use: Number of sources to combine

    Returns:
        Combined iterator and a description string
    """
    from datasets import load_dataset

    loaded_sources = []

    for source in sorted(PORTUGUESE_SOURCES, key=lambda x: x["priority"]):
        if len(loaded_sources) >= sources_to_use:
            break

        try:
            load_args = {
                "path": source["dataset_id"],
                "split": source["split"],
                "streaming": True,
            }

            if source["config"]:
                load_args["name"] = source["config"]

            dataset = load_dataset(**load_args)
            test_iter = iter(dataset)
            test_sample = next(test_iter)

            text_field = source["text_field"]
            if text_field not in test_sample:
                for alt_field in ['text', 'content', 'body']:
                    if alt_field in test_sample:
                        text_field = alt_field
                        break
                else:
                    continue

            if test_sample.get(text_field) and len(str(test_sample[text_field])) > 10:
                # Reload fresh iterator
                dataset = load_dataset(**load_args)
                loaded_sources.append({
                    "name": source["name"],
                    "iterator": iter(dataset),
                    "text_field": text_field,
                    "exhausted": False,
                })
                logger.info(f"✓ Added source: {source['name']}")

        except Exception as e:
            logger.debug(f"Could not load {source['name']}: {e}")
            continue

    if not loaded_sources:
        return None, ""

    source_names = ", ".join(s["name"] for s in loaded_sources)
    logger.info(f"\nCombined {len(loaded_sources)} Portuguese sources: {source_names}")

    def combined_generator():
        """Generate samples from multiple sources in round-robin fashion"""
        sample_count = 0
        active_sources = loaded_sources.copy()

        while active_sources:
            for source in active_sources[:]:  # Copy to allow modification
                if max_samples and sample_count >= max_samples:
                    return

                try:
                    sample = next(source["iterator"])
                    text = sample.get(source["text_field"], "")

                    if text and len(str(text).strip()) > 10:
                        yield {"text": str(text), "_source": source["name"]}
                        sample_count += 1

                except StopIteration:
                    source["exhausted"] = True
                    active_sources.remove(source)
                    logger.info(f"Source exhausted: {source['name']}")

                except Exception as e:
                    logger.debug(f"Error from {source['name']}: {e}")
                    continue

    return combined_generator(), source_names


def extract_text(sample: Dict[str, Any], text_field: str) -> str:
    """Extract text from a dataset sample"""
    if text_field in sample:
        text = sample[text_field]
        if text:
            return str(text)

    # Try common text field names
    for key in ['text', 'content', 'sentence', 'document', 'article', 'premise', 'body']:
        if key in sample and sample[key]:
            return str(sample[key])

    return ""


def find_existing_chunks(output_path: Path) -> List[Path]:
    """Find all existing chunk files in the output directory"""
    chunk_files = sorted(output_path.glob("tokens_chunk_*.bin"))
    return chunk_files


def detect_chunk_dtype(chunk_file: Path) -> np.dtype:
    """Detect the dtype of a chunk file by checking file size"""
    file_size = chunk_file.stat().st_size

    # Try uint16 first (most common)
    try:
        test_mmap = np.memmap(chunk_file, dtype=np.uint16, mode='r')
        expected_size = len(test_mmap) * 2  # uint16 = 2 bytes
        del test_mmap

        if expected_size == file_size:
            return np.uint16
    except Exception:
        pass

    # Try uint32
    try:
        test_mmap = np.memmap(chunk_file, dtype=np.uint32, mode='r')
        expected_size = len(test_mmap) * 4  # uint32 = 4 bytes
        del test_mmap

        if expected_size == file_size:
            return np.uint32
    except Exception:
        pass

    # Default to uint16
    return np.uint16


def merge_chunks_from_directory(
    output_path: Path,
    delete_chunks: bool = True,
    dataset_name: str = "Unknown"
) -> int:
    """
    Merge all chunk files in a directory into a single tokens.bin file.

    Args:
        output_path: Directory containing chunk files
        delete_chunks: Whether to delete chunk files after merging
        dataset_name: Name of the dataset for metadata

    Returns:
        Total number of tokens merged
    """
    chunk_files = find_existing_chunks(output_path)

    if not chunk_files:
        logger.warning(f"No chunk files found in {output_path}")
        return 0

    logger.info(f"Found {len(chunk_files)} chunk files in {output_path}")
    for chunk_file in chunk_files:
        size_mb = chunk_file.stat().st_size / (1024 * 1024)
        logger.info(f"  - {chunk_file.name}: {size_mb:.1f} MB")

    tokens_file = output_path / "tokens.bin"
    metadata_file = output_path / "metadata.json"

    # Check if tokens.bin already exists
    if tokens_file.exists():
        logger.warning(f"tokens.bin already exists at {tokens_file}")
        response = input("Overwrite? [y/N]: ").strip().lower()
        if response != 'y':
            logger.info("Merge cancelled")
            return 0
        tokens_file.unlink()

    # Detect dtype from first chunk
    dtype = detect_chunk_dtype(chunk_files[0])
    logger.info(f"Detected dtype: {dtype}")

    # First pass: count total tokens
    total_tokens = 0
    chunk_sizes = []

    logger.info("Counting tokens in chunks...")
    for chunk_file in tqdm(chunk_files, desc="Scanning chunks"):
        try:
            chunk_data = np.memmap(chunk_file, dtype=dtype, mode='r')
            chunk_size = len(chunk_data)
            chunk_sizes.append(chunk_size)
            total_tokens += chunk_size
            del chunk_data
        except Exception as e:
            logger.error(f"Error reading {chunk_file}: {e}")
            return 0

    logger.info(f"Total tokens to merge: {total_tokens:,}")
    total_size_gb = (total_tokens * np.dtype(dtype).itemsize) / (1024**3)
    logger.info(f"Output file size: {total_size_gb:.2f} GB")

    # Create output memmap
    logger.info(f"Creating output file: {tokens_file}")
    output_memmap = np.memmap(
        tokens_file,
        dtype=dtype,
        mode='w+',
        shape=(total_tokens,)
    )

    # Second pass: copy data
    offset = 0
    for i, chunk_file in enumerate(tqdm(chunk_files, desc="Merging chunks")):
        chunk_data = np.memmap(chunk_file, dtype=dtype, mode='r')
        chunk_size = chunk_sizes[i]

        output_memmap[offset:offset + chunk_size] = chunk_data[:]
        offset += chunk_size

        del chunk_data

        # Flush periodically to avoid memory issues
        if (i + 1) % 10 == 0:
            output_memmap.flush()

    output_memmap.flush()
    del output_memmap

    logger.info(f"Successfully merged {len(chunk_files)} chunks into {tokens_file}")

    # Save metadata
    metadata = {
        "dataset_name": dataset_name,
        "total_samples": 0,  # Unknown when merging from chunks
        "total_tokens": total_tokens,
        "vocab_size": 50304,
        "max_seq_len": 2048,
        "tokenizer": "gpt2",
        "dtype": "uint16" if dtype == np.uint16 else "uint32",
        "merged_from_chunks": len(chunk_files),
    }

    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved metadata to {metadata_file}")

    # Delete chunk files if requested
    if delete_chunks:
        logger.info("Deleting chunk files...")
        for chunk_file in chunk_files:
            try:
                chunk_file.unlink()
                logger.debug(f"Deleted {chunk_file}")
            except Exception as e:
                logger.warning(f"Failed to delete {chunk_file}: {e}")

        # Also delete progress file if exists
        progress_file = output_path / "chunk_progress.json"
        if progress_file.exists():
            progress_file.unlink()

        logger.info(f"Deleted {len(chunk_files)} chunk files")

    return total_tokens


def tokenize_and_save_dataset(
    dataset,
    text_field: str,
    output_path: Path,
    tokenizer,
    config: PreprocessConfig,
    dataset_name: str,
    max_samples: Optional[int] = None
):
    """
    Tokenize dataset and save as memory-mapped numpy array.

    Creates:
    - {output_path}/tokens.bin - memory-mapped token array
    - {output_path}/metadata.json - dataset metadata

    Saves progress in chunks for resilience to interruptions.
    """
    global _shutdown_requested, _current_tokens, _current_output_path, _current_chunk_idx

    output_path.mkdir(parents=True, exist_ok=True)

    tokens_file = output_path / "tokens.bin"
    metadata_file = output_path / "metadata.json"

    # Check for existing chunks (interrupted download)
    existing_chunks = find_existing_chunks(output_path)
    if existing_chunks:
        logger.warning(f"Found {len(existing_chunks)} existing chunks in {output_path}")
        logger.warning("This may be from an interrupted download.")
        logger.warning("Options:")
        logger.warning("  1. Run with --merge_chunks to merge existing chunks")
        logger.warning("  2. Delete the chunks manually to start fresh")

        response = input("Continue anyway (will overwrite)? [y/N]: ").strip().lower()
        if response != 'y':
            logger.info("Preprocessing cancelled")
            return 0

        # Delete existing chunks
        for chunk_file in existing_chunks:
            chunk_file.unlink()
        logger.info("Deleted existing chunks")

    logger.info(f"Processing {dataset_name} dataset...")
    logger.info(f"Output: {output_path}")

    # Setup for interrupt handling
    _current_output_path = output_path
    _current_chunk_idx = 0
    _current_tokens = []

    all_tokens = []
    total_samples = 0
    total_tokens = 0
    chunk_idx = 0
    empty_samples = 0

    # Process in batches
    batch_texts = []

    try:
        with tqdm(desc=f"Tokenizing {dataset_name}", unit=" samples") as pbar:
            for sample in dataset:
                # Check for shutdown request
                if _shutdown_requested:
                    logger.info("Shutdown requested, stopping tokenization...")
                    break

                text = extract_text(sample, text_field)

                if not text or len(text.strip()) < 10:
                    empty_samples += 1
                    if empty_samples <= 10:
                        logger.debug(f"Skipping empty/short sample. Keys: {list(sample.keys()) if isinstance(sample, dict) else 'N/A'}")
                    continue

                batch_texts.append(text)

                # Process batch
                if len(batch_texts) >= config.tokenizer_batch_size:
                    # Tokenize batch
                    encoded = tokenizer(
                        batch_texts,
                        add_special_tokens=True,
                        truncation=False,
                        return_attention_mask=False,
                        return_token_type_ids=False,
                    )

                    for token_ids in encoded['input_ids']:
                        # Add EOS token at the end of each document
                        token_ids = token_ids + [tokenizer.eos_token_id]
                        all_tokens.extend(token_ids)
                        total_tokens += len(token_ids)

                    total_samples += len(batch_texts)
                    pbar.update(len(batch_texts))
                    pbar.set_postfix({
                        'tokens': f'{total_tokens:,}',
                        'chunks': chunk_idx,
                        'skipped': empty_samples
                    })

                    batch_texts = []

                    # Update global state for interrupt handling
                    _current_tokens = all_tokens.copy()
                    _current_chunk_idx = chunk_idx

                # Check sample limit
                if max_samples and total_samples >= max_samples:
                    logger.info(f"Reached max samples limit: {max_samples}")
                    break

                # Save chunk if accumulated enough tokens
                if len(all_tokens) >= config.tokens_per_chunk:
                    chunk_file = output_path / f"tokens_chunk_{chunk_idx:04d}.bin"
                    save_tokens_memmap(all_tokens, chunk_file)
                    logger.info(f"Saved chunk {chunk_idx}: {len(all_tokens):,} tokens")
                    chunk_idx += 1
                    all_tokens = []
                    _current_tokens = []

    except KeyboardInterrupt:
        # This shouldn't happen if signal handler works, but just in case
        logger.warning("KeyboardInterrupt caught, saving progress...")
        if all_tokens:
            chunk_file = output_path / f"tokens_chunk_{chunk_idx:04d}.bin"
            save_tokens_memmap(all_tokens, chunk_file)
            save_chunk_progress(output_path, chunk_idx + 1)
        raise

    # Check if we were interrupted
    if _shutdown_requested:
        # Progress already saved by signal handler
        logger.info("Processing interrupted. Use --merge_chunks to complete.")
        return 0

    # Process remaining batch
    if batch_texts:
        encoded = tokenizer(
            batch_texts,
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )

        for token_ids in encoded['input_ids']:
            token_ids = token_ids + [tokenizer.eos_token_id]
            all_tokens.extend(token_ids)
            total_tokens += len(token_ids)

        total_samples += len(batch_texts)

    # Log statistics
    if empty_samples > 0:
        logger.info(f"Skipped {empty_samples:,} empty/short samples")

    # Save final tokens
    if all_tokens:
        if chunk_idx > 0:
            # Multiple chunks - save as another chunk
            chunk_file = output_path / f"tokens_chunk_{chunk_idx:04d}.bin"
            save_tokens_memmap(all_tokens, chunk_file)
            logger.info(f"Saved final chunk {chunk_idx}: {len(all_tokens):,} tokens")
            chunk_idx += 1
        else:
            # Single file (small dataset)
            save_tokens_memmap(all_tokens, tokens_file)

    # Clear global state
    _current_tokens = []
    _current_output_path = None

    # Merge chunks if multiple exist
    if chunk_idx > 0:
        total_tokens = merge_chunks_from_directory(
            output_path,
            delete_chunks=True,
            dataset_name=dataset_name
        )

    # Save/update metadata
    metadata = {
        "dataset_name": dataset_name,
        "total_samples": total_samples,
        "total_tokens": total_tokens,
        "vocab_size": config.vocab_size,
        "max_seq_len": config.max_seq_len,
        "tokenizer": "gpt2",
        "dtype": "uint16",
        "empty_samples_skipped": empty_samples,
    }

    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Completed {dataset_name}:")
    logger.info(f"  Total samples: {total_samples:,}")
    logger.info(f"  Total tokens: {total_tokens:,}")
    logger.info(f"  Saved to: {tokens_file}")

    return total_tokens


def save_tokens_memmap(tokens: List[int], output_path: Path):
    """Save tokens as memory-mapped numpy array"""
    if not tokens:
        logger.warning(f"No tokens to save to {output_path}")
        return

    # Use uint16 for vocab sizes up to 65535, otherwise uint32
    dtype = np.uint16 if max(tokens) < 65535 else np.uint32

    tokens_array = np.array(tokens, dtype=dtype)

    # Create memory-mapped file
    memmap = np.memmap(
        output_path,
        dtype=dtype,
        mode='w+',
        shape=tokens_array.shape
    )
    memmap[:] = tokens_array
    memmap.flush()
    del memmap

    logger.info(f"Saved {len(tokens):,} tokens to {output_path}")


def verify_preprocessed_data(output_dir: Path, tokenizer):
    """Verify preprocessed data by decoding a sample"""
    for lang in ['en', 'pt']:
        lang_dir = output_dir / lang
        tokens_file = lang_dir / "tokens.bin"
        metadata_file = lang_dir / "metadata.json"

        if not tokens_file.exists():
            logger.warning(f"Missing tokens file: {tokens_file}")
            continue

        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        logger.info(f"\nVerifying {lang} dataset:")
        logger.info(f"  Dataset: {metadata.get('dataset_name', 'Unknown')}")
        logger.info(f"  Total tokens: {metadata['total_tokens']:,}")
        logger.info(f"  Total samples: {metadata.get('total_samples', 'N/A')}")

        # Load and decode sample
        tokens = np.memmap(tokens_file, dtype=np.uint16, mode='r')

        # Decode first 100 tokens
        sample_tokens = tokens[:100].tolist()
        decoded = tokenizer.decode(sample_tokens)

        logger.info(f"  Sample (first 100 tokens):")
        logger.info(f"    {decoded[:200]}...")

        del tokens


def list_chunks(output_dir: Path):
    """List all chunks in the output directory"""
    logger.info("=" * 60)
    logger.info("Dataset Status")
    logger.info("=" * 60)

    for lang in ['en', 'pt']:
        lang_dir = output_dir / lang
        if not lang_dir.exists():
            logger.info(f"\n{lang.upper()} dataset: Not found")
            continue

        chunks = find_existing_chunks(lang_dir)
        tokens_file = lang_dir / "tokens.bin"
        metadata_file = lang_dir / "metadata.json"

        logger.info(f"\n{lang.upper()} dataset ({lang_dir}):")

        if tokens_file.exists():
            size_gb = tokens_file.stat().st_size / (1024**3)

            # Load metadata for more info
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                token_count = metadata.get('total_tokens', 0)
                dataset_name = metadata.get('dataset_name', 'Unknown')
                logger.info(f"  ✓ COMPLETE: {token_count:,} tokens ({size_gb:.2f} GB)")
                logger.info(f"    Source: {dataset_name}")
            else:
                logger.info(f"  ✓ tokens.bin: {size_gb:.2f} GB")

        elif chunks:
            total_size = sum(c.stat().st_size for c in chunks)
            size_gb = total_size / (1024**3)
            logger.info(f"  ⚠ INCOMPLETE: {len(chunks)} chunks ({size_gb:.2f} GB total)")
            logger.info("    Run with --merge_chunks to complete")
            for chunk in chunks:
                size_mb = chunk.stat().st_size / (1024**1024)
                logger.info(f"      - {chunk.name}: {size_mb:.1f} MB")
        else:
            logger.info("  ✗ No data found")


def main():
    # Setup signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    parser = argparse.ArgumentParser(
        description="Preprocess datasets for BitNet-Mamba training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/tokenized",
        help="Output directory for tokenized data"
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=2048,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--en_max_samples",
        type=int,
        default=None,
        help="Maximum English samples to process (None = all)"
    )
    parser.add_argument(
        "--pt_max_samples",
        type=int,
        default=None,
        help="Maximum Portuguese samples to process (None = all)"
    )
    parser.add_argument(
        "--tokenizer_batch_size",
        type=int,
        default=1000,
        help="Batch size for tokenization"
    )
    parser.add_argument(
        "--tokens_per_chunk",
        type=int,
        default=100_000_000,
        help="Number of tokens per chunk (100M default)"
    )
    parser.add_argument(
        "--skip_english",
        action="store_true",
        help="Skip English dataset preprocessing"
    )
    parser.add_argument(
        "--skip_portuguese",
        action="store_true",
        help="Skip Portuguese dataset preprocessing"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify preprocessed data after completion"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing even if tokens.bin already exists"
    )

    # Chunk management options
    parser.add_argument(
        "--merge_chunks",
        action="store_true",
        help="Merge existing chunks from interrupted downloads into final tokens.bin files"
    )
    parser.add_argument(
        "--merge_lang",
        type=str,
        choices=['en', 'pt', 'all'],
        default='all',
        help="Which language chunks to merge (default: all)"
    )
    parser.add_argument(
        "--keep_chunks",
        action="store_true",
        help="Keep chunk files after merging (don't delete)"
    )
    parser.add_argument(
        "--list_chunks",
        action="store_true",
        help="List existing chunks and their status"
    )

    # Portuguese source options
    parser.add_argument(
        "--pt_combined_sources",
        type=int,
        default=1,
        help="Number of Portuguese sources to combine (1=single best, 2+=combined)"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # List chunks mode
    if args.list_chunks:
        list_chunks(output_dir)
        return

    # Merge chunks mode
    if args.merge_chunks:
        logger.info("=" * 60)
        logger.info("Merging Chunks from Interrupted Downloads")
        logger.info("=" * 60)

        total_merged = 0

        if args.merge_lang in ['en', 'all']:
            en_dir = output_dir / "en"
            if en_dir.exists() and find_existing_chunks(en_dir):
                logger.info("\nMerging English chunks...")
                tokens = merge_chunks_from_directory(
                    en_dir,
                    delete_chunks=not args.keep_chunks,
                    dataset_name="English (fineweb-edu)"
                )
                total_merged += tokens

        if args.merge_lang in ['pt', 'all']:
            pt_dir = output_dir / "pt"
            if pt_dir.exists() and find_existing_chunks(pt_dir):
                logger.info("\nMerging Portuguese chunks...")
                tokens = merge_chunks_from_directory(
                    pt_dir,
                    delete_chunks=not args.keep_chunks,
                    dataset_name="Portuguese"
                )
                total_merged += tokens

        if total_merged > 0:
            logger.info("=" * 60)
            logger.info(f"Merge Complete! Total tokens: {total_merged:,}")
            logger.info("=" * 60)
        else:
            logger.warning("No chunks found to merge")

        return

    # Normal preprocessing mode
    config = PreprocessConfig(
        output_dir=args.output_dir,
        max_seq_len=args.max_seq_len,
        en_max_samples=args.en_max_samples,
        pt_max_samples=args.pt_max_samples,
        tokenizer_batch_size=args.tokenizer_batch_size,
        tokens_per_chunk=args.tokens_per_chunk,
    )

    logger.info("=" * 60)
    logger.info("BitNet-Mamba Dataset Preprocessing")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Max sequence length: {config.max_seq_len}")
    logger.info(f"Tokenizer batch size: {config.tokenizer_batch_size}")
    logger.info(f"Tokens per chunk: {config.tokens_per_chunk:,}")
    logger.info(f"Force reprocess: {args.force}")
    logger.info("")
    logger.info("Press Ctrl+C to interrupt (progress will be saved)")
    logger.info("=" * 60)

    # Initialize tokenizer
    tokenizer = get_tokenizer()

    total_en_tokens = 0
    total_pt_tokens = 0

    # Process English dataset
    if not args.skip_english and not _shutdown_requested:
        en_output = output_dir / "en"
        is_processed, token_count = check_dataset_already_processed(en_output)

        if is_processed and not args.force:
            logger.info(f"\n✓ English dataset already processed: {token_count:,} tokens")
            logger.info(f"  Use --force to reprocess")
            total_en_tokens = token_count
        else:
            if is_processed:
                logger.info(f"\nForce reprocessing English dataset...")
            en_dataset, en_text_field = load_english_dataset(config.en_max_samples)
            if en_dataset:
                total_en_tokens = tokenize_and_save_dataset(
                    en_dataset,
                    en_text_field,
                    en_output,
                    tokenizer,
                    config,
                    "English (fineweb-edu)",
                    config.en_max_samples
                )
    else:
        logger.info("Skipping English dataset")

    # Process Portuguese dataset
    if not args.skip_portuguese and not _shutdown_requested:
        pt_output = output_dir / "pt"
        is_processed, token_count = check_dataset_already_processed(pt_output)

        if is_processed and not args.force:
            logger.info(f"\n✓ Portuguese dataset already processed: {token_count:,} tokens")
            logger.info(f"  Use --force to reprocess")
            total_pt_tokens = token_count
        else:
            if is_processed:
                logger.info(f"\nForce reprocessing Portuguese dataset...")

            # Load Portuguese dataset(s)
            if args.pt_combined_sources > 1:
                pt_dataset, source_name = create_combined_portuguese_iterator(
                    config.pt_max_samples,
                    sources_to_use=args.pt_combined_sources
                )
                text_field = "text"
            else:
                pt_dataset, text_field, source_name = load_portuguese_dataset(config.pt_max_samples)

            if pt_dataset:
                total_pt_tokens = tokenize_and_save_dataset(
                    pt_dataset,
                    text_field,
                    pt_output,
                    tokenizer,
                    config,
                    f"Portuguese ({source_name})",
                    config.pt_max_samples
                )
            else:
                logger.error("Could not load any Portuguese dataset!")
                logger.error("Please check your internet connection.")
    else:
        logger.info("Skipping Portuguese dataset")

    # Check if we were interrupted
    if _shutdown_requested:
        logger.info("=" * 60)
        logger.info("Preprocessing Interrupted")
        logger.info("=" * 60)
        logger.info("Progress has been saved as chunks.")
        logger.info("To complete preprocessing, run:")
        logger.info(f"  python preprocess_datasets.py --output_dir {output_dir} --merge_chunks")
        return

    # Summary
    logger.info("=" * 60)
    logger.info("Preprocessing Complete!")
    logger.info("=" * 60)
    logger.info(f"English tokens: {total_en_tokens:,}")
    logger.info(f"Portuguese tokens: {total_pt_tokens:,}")
    logger.info(f"Total tokens: {total_en_tokens + total_pt_tokens:,}")
    logger.info(f"Output directory: {output_dir}")

    # Verify if requested
    if args.verify:
        verify_preprocessed_data(output_dir, tokenizer)

    logger.info("\nTo use this data for training:")
    logger.info(f"  python train_hybrid-mamba-bitnet.py --data_dir {output_dir}")


if __name__ == "__main__":
    main()
