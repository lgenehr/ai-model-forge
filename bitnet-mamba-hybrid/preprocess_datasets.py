#!/usr/bin/env python3
"""
Dataset Preprocessing Script for BitNet-Mamba Hybrid Training

Downloads, tokenizes, and saves datasets as memory-mapped files for efficient training.
Eliminates HTTP requests during training by pre-processing all data.

Datasets:
- English: HuggingFaceFW/fineweb-edu (sample-10BT split)
- Portuguese: eduagarcia/portuguese_benchmark (train split)

Usage:
    python preprocess_datasets.py --output_dir ./data/tokenized
    python preprocess_datasets.py --output_dir ./data/tokenized --max_samples 1000000
"""

import os
import sys
import argparse
import logging
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
    """
    import json

    output_path.mkdir(parents=True, exist_ok=True)

    tokens_file = output_path / "tokens.bin"
    metadata_file = output_path / "metadata.json"

    logger.info(f"Processing {dataset_name} dataset...")
    logger.info(f"Output: {output_path}")

    all_tokens = []
    total_samples = 0
    total_tokens = 0
    chunk_idx = 0

    # Process in batches
    batch_texts = []

    with tqdm(desc=f"Tokenizing {dataset_name}", unit=" samples") as pbar:
        for sample in dataset:
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
                    'samples': f'{total_samples:,}'
                })

                batch_texts = []

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
        else:
            # Single file
            save_tokens_memmap(all_tokens, tokens_file)

    # Merge chunks if multiple exist
    if chunk_idx > 0:
        merge_chunks(output_path, tokens_file, chunk_idx + 1)

    # Save metadata
    metadata = {
        "dataset_name": dataset_name,
        "total_samples": total_samples,
        "total_tokens": total_tokens,
        "vocab_size": config.vocab_size,
        "max_seq_len": config.max_seq_len,
        "tokenizer": "gpt2",
        "dtype": "uint16",  # Fits vocab_size up to 65535
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


def merge_chunks(output_dir: Path, output_file: Path, num_chunks: int):
    """Merge multiple token chunks into a single file"""
    logger.info(f"Merging {num_chunks} chunks into {output_file}...")

    # First pass: count total tokens
    total_tokens = 0
    chunk_files = []

    for i in range(num_chunks):
        chunk_file = output_dir / f"tokens_chunk_{i:04d}.bin"
        if chunk_file.exists():
            chunk_files.append(chunk_file)
            chunk_data = np.memmap(chunk_file, dtype=np.uint16, mode='r')
            total_tokens += len(chunk_data)
            del chunk_data

    logger.info(f"Total tokens to merge: {total_tokens:,}")

    # Create output memmap
    output_memmap = np.memmap(
        output_file,
        dtype=np.uint16,
        mode='w+',
        shape=(total_tokens,)
    )

    # Second pass: copy data
    offset = 0
    for chunk_file in tqdm(chunk_files, desc="Merging chunks"):
        chunk_data = np.memmap(chunk_file, dtype=np.uint16, mode='r')
        output_memmap[offset:offset + len(chunk_data)] = chunk_data
        offset += len(chunk_data)
        del chunk_data

    output_memmap.flush()
    del output_memmap

    # Delete chunk files
    for chunk_file in chunk_files:
        chunk_file.unlink()

    logger.info(f"Merged {num_chunks} chunks into {output_file}")


def verify_preprocessed_data(output_dir: Path, tokenizer):
    """Verify preprocessed data by decoding a sample"""
    import json

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
        logger.info(f"  Total samples: {metadata['total_samples']:,}")

        # Load and decode sample
        tokens = np.memmap(tokens_file, dtype=np.uint16, mode='r')

        # Decode first 100 tokens
        sample_tokens = tokens[:100].tolist()
        decoded = tokenizer.decode(sample_tokens)

        logger.info(f"  Sample (first 100 tokens):")
        logger.info(f"    {decoded[:200]}...")

        del tokens


def main():
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

    args = parser.parse_args()

    config = PreprocessConfig(
        output_dir=args.output_dir,
        max_seq_len=args.max_seq_len,
        en_max_samples=args.en_max_samples,
        pt_max_samples=args.pt_max_samples,
        tokenizer_batch_size=args.tokenizer_batch_size,
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("BitNet-Mamba Dataset Preprocessing")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Max sequence length: {config.max_seq_len}")
    logger.info(f"Tokenizer batch size: {config.tokenizer_batch_size}")

    # Initialize tokenizer
    tokenizer = get_tokenizer()

    total_en_tokens = 0
    total_pt_tokens = 0

    # Process English dataset
    if not args.skip_english:
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
    if not args.skip_portuguese:
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
