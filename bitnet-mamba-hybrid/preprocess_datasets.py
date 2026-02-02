#!/usr/bin/env python3
"""
Dataset Preprocessing Script for BitNet-Mamba Hybrid Training

Downloads, tokenizes, and saves datasets as memory-mapped files for efficient training.
Eliminates HTTP requests during training by pre-processing all data.

Datasets:
- English: HuggingFaceFW/fineweb-edu (sample-10BT split)
- Portuguese: eduagarcia/portuguese_benchmark (train split)

Features:
- Saves progress in chunks (resilient to interruptions)
- Can merge chunks from interrupted downloads with --merge_chunks
- Graceful shutdown on Ctrl+C (saves current progress)

Usage:
    # Normal preprocessing
    python preprocess_datasets.py --output_dir ./data/tokenized

    # Merge chunks from interrupted download
    python preprocess_datasets.py --output_dir ./data/tokenized --merge_chunks

    # Merge only English chunks
    python preprocess_datasets.py --output_dir ./data/tokenized --merge_chunks --merge_lang en
"""

import os
import sys
import signal
import argparse
import logging
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
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


def load_portuguese_dataset(max_samples: Optional[int] = None):
    """Load Portuguese dataset from HuggingFace"""
    try:
        from datasets import load_dataset

        logger.info("Loading Portuguese dataset: eduagarcia/portuguese_benchmark")

        dataset = load_dataset(
            "eduagarcia/portuguese_benchmark",
            "assin2-rte",
            split="train",
            streaming=True
        )

        logger.info("Successfully loaded Portuguese dataset (streaming mode)")
        # This dataset has 'premise' and 'hypothesis' fields
        return dataset, "premise"

    except Exception as e:
        logger.warning(f"Could not load Portuguese dataset: {e}")
        logger.info("Trying mc4 Portuguese subset")

        try:
            from datasets import load_dataset

            dataset = load_dataset(
                "mc4",
                "pt",
                split="train",
                streaming=True
            )
            return dataset, "text"
        except Exception as e2:
            logger.warning(f"Could not load mc4 Portuguese: {e2}")
            return None, None


def extract_text(sample: Dict[str, Any], text_field: str) -> str:
    """Extract text from a dataset sample"""
    if text_field in sample:
        return sample[text_field]

    # Try common text field names
    for key in ['text', 'content', 'sentence', 'document', 'article', 'premise']:
        if key in sample:
            return sample[key]

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
                        'chunks': chunk_idx
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
        logger.info(f"  Total tokens: {metadata['total_tokens']:,}")
        logger.info(f"  Total samples: {metadata.get('total_samples', 'N/A'):,}")

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
    logger.info("Chunk Status")
    logger.info("=" * 60)

    for lang in ['en', 'pt']:
        lang_dir = output_dir / lang
        if not lang_dir.exists():
            continue

        chunks = find_existing_chunks(lang_dir)
        tokens_file = lang_dir / "tokens.bin"

        logger.info(f"\n{lang.upper()} dataset ({lang_dir}):")

        if tokens_file.exists():
            size_gb = tokens_file.stat().st_size / (1024**3)
            logger.info(f"  tokens.bin: {size_gb:.2f} GB (COMPLETE)")
        elif chunks:
            total_size = sum(c.stat().st_size for c in chunks)
            size_gb = total_size / (1024**3)
            logger.info(f"  {len(chunks)} chunks found ({size_gb:.2f} GB total)")
            logger.info("  Status: INCOMPLETE - run with --merge_chunks to complete")
            for chunk in chunks:
                size_mb = chunk.stat().st_size / (1024**1024)
                logger.info(f"    - {chunk.name}: {size_mb:.1f} MB")
        else:
            logger.info("  No data found")


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
                    dataset_name="Portuguese (portuguese_benchmark)"
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
    logger.info("")
    logger.info("Press Ctrl+C to interrupt (progress will be saved)")
    logger.info("=" * 60)

    # Initialize tokenizer
    tokenizer = get_tokenizer()

    total_en_tokens = 0
    total_pt_tokens = 0

    # Process English dataset
    if not args.skip_english and not _shutdown_requested:
        en_dataset, en_text_field = load_english_dataset(config.en_max_samples)
        if en_dataset:
            total_en_tokens = tokenize_and_save_dataset(
                en_dataset,
                en_text_field,
                output_dir / "en",
                tokenizer,
                config,
                "English (fineweb-edu)",
                config.en_max_samples
            )
    else:
        logger.info("Skipping English dataset")

    # Process Portuguese dataset
    if not args.skip_portuguese and not _shutdown_requested:
        pt_dataset, pt_text_field = load_portuguese_dataset(config.pt_max_samples)
        if pt_dataset:
            total_pt_tokens = tokenize_and_save_dataset(
                pt_dataset,
                pt_text_field,
                output_dir / "pt",
                tokenizer,
                config,
                "Portuguese (portuguese_benchmark)",
                config.pt_max_samples
            )
        else:
            logger.warning("Portuguese dataset not available")
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
