#!/usr/bin/env python3
"""
Efficient Data Loader for BitNet-Mamba Hybrid Training

Provides memory-mapped data loading with:
- Zero HTTP requests during training
- Efficient batching with proper padding/truncation
- Mixed English/Portuguese sampling based on configurable ratios
- Multi-worker data loading with prefetching

Usage:
    from data_loader import create_dataloader, PreTokenizedDataset

    dataloader = create_dataloader(
        data_dir="/home/lgene/meu_modelo_temp/ai-model-forge/datasets/tokenized",
        batch_size=6,
        max_seq_len=2048,
        en_ratio=0.5,
        pt_ratio=0.5,
        num_workers=4
    )

    for batch in dataloader:
        input_ids = batch['input_ids']  # [batch_size, max_seq_len]
        labels = batch['labels']        # [batch_size, max_seq_len]
        ...
"""

import os
import json
import random
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Iterator, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler

logger = logging.getLogger(__name__)

DEFAULT_DATASET_ROOT = "/home/lgene/meu_modelo_temp/ai-model-forge/datasets/tokenized"


class MemoryMappedTokens:
    """
    Memory-mapped token array for efficient random access.

    Uses numpy memmap for zero-copy reads directly from disk.
    """

    def __init__(self, tokens_file: Path, metadata_file: Path):
        self.tokens_file = tokens_file
        self.metadata_file = metadata_file

        # Load metadata
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)

        self.total_tokens = self.metadata['total_tokens']
        self.vocab_size = self.metadata.get('vocab_size', 50304)
        self.dtype_str = self.metadata.get('dtype', 'uint16')

        # Determine numpy dtype
        self.dtype = np.uint16 if self.dtype_str == 'uint16' else np.uint32

        # Memory-map the token file
        self._mmap = None
        self._load_memmap()

    def _load_memmap(self):
        """Load or reload the memory-mapped array"""
        if self._mmap is not None:
            del self._mmap

        self._mmap = np.memmap(
            self.tokens_file,
            dtype=self.dtype,
            mode='r'
        )

        logger.info(f"Loaded {len(self._mmap):,} tokens from {self.tokens_file}")

    def __len__(self) -> int:
        return len(self._mmap)

    def __getitem__(self, idx) -> np.ndarray:
        return self._mmap[idx]

    def get_chunk(self, start: int, length: int) -> np.ndarray:
        """Get a contiguous chunk of tokens"""
        end = min(start + length, len(self._mmap))
        return self._mmap[start:end].copy()


class PreTokenizedDataset(Dataset):
    """
    PyTorch Dataset for pre-tokenized data with bilingual mixing.

    Features:
    - Memory-mapped loading for efficiency
    - Random sampling with language ratio control
    - Fixed-length sequence generation
    - Proper handling of document boundaries
    - Train/val split support (last 0.5% reserved for validation)
    """

    # Fraction of each dataset reserved for validation
    VAL_FRACTION = 0.005  # 0.5%

    def __init__(
        self,
        data_dir: str,
        max_seq_len: int = 2048,
        en_ratio: float = 0.5,
        pt_ratio: float = 0.5,
        seed: int = 42,
        epoch_tokens: Optional[int] = None,
        split: str = "train",
    ):
        """
        Initialize the dataset.

        Args:
            data_dir: Directory containing 'en' and 'pt' subdirectories
            max_seq_len: Maximum sequence length
            en_ratio: Ratio of English samples (0.0-1.0)
            pt_ratio: Ratio of Portuguese samples (0.0-1.0)
            seed: Random seed for reproducibility
            epoch_tokens: Total tokens per epoch (for __len__ calculation)
            split: "train" or "val" - determines which portion of data to use
        """
        self.data_dir = Path(data_dir)
        self.max_seq_len = max_seq_len
        self.en_ratio = en_ratio
        self.pt_ratio = pt_ratio
        self.seed = seed
        self.split = split

        # Normalize ratios
        total_ratio = en_ratio + pt_ratio
        self.en_ratio_norm = en_ratio / total_ratio
        self.pt_ratio_norm = pt_ratio / total_ratio

        # Random state
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)

        # Load datasets
        self.en_tokens: Optional[MemoryMappedTokens] = None
        self.pt_tokens: Optional[MemoryMappedTokens] = None

        self._load_datasets()

        # Compute split boundaries for each language
        self.en_train_end = 0
        self.en_val_start = 0
        self.pt_train_end = 0
        self.pt_val_start = 0

        if self.en_tokens:
            en_val_size = int(len(self.en_tokens) * self.VAL_FRACTION)
            self.en_train_end = len(self.en_tokens) - en_val_size
            self.en_val_start = self.en_train_end

        if self.pt_tokens:
            pt_val_size = int(len(self.pt_tokens) * self.VAL_FRACTION)
            self.pt_train_end = len(self.pt_tokens) - pt_val_size
            self.pt_val_start = self.pt_train_end

        # Calculate dataset length (number of possible sequences)
        self.total_tokens = 0
        if self.en_tokens:
            if split == "train":
                self.total_tokens += self.en_train_end
            else:
                self.total_tokens += len(self.en_tokens) - self.en_val_start
        if self.pt_tokens:
            if split == "train":
                self.total_tokens += self.pt_train_end
            else:
                self.total_tokens += len(self.pt_tokens) - self.pt_val_start

        # Epoch tokens for length calculation
        if epoch_tokens:
            self._epoch_sequences = epoch_tokens // max_seq_len
        else:
            self._epoch_sequences = self.total_tokens // max_seq_len

        logger.info(f"Dataset initialized (split={split}):")
        logger.info(f"  Total tokens in split: {self.total_tokens:,}")
        logger.info(f"  Sequences per epoch: {self._epoch_sequences:,}")
        logger.info(f"  EN/PT ratio: {self.en_ratio_norm:.2f}/{self.pt_ratio_norm:.2f}")
        if self.en_tokens:
            logger.info(f"  EN split: [0, {self.en_train_end:,}) train | [{self.en_val_start:,}, {len(self.en_tokens):,}) val")
        if self.pt_tokens:
            logger.info(f"  PT split: [0, {self.pt_train_end:,}) train | [{self.pt_val_start:,}, {len(self.pt_tokens):,}) val")

    def _load_datasets(self):
        """Load English and Portuguese token files"""
        en_dir = self.data_dir / "en"
        pt_dir = self.data_dir / "pt"

        # Load English dataset
        en_tokens_file = en_dir / "tokens.bin"
        en_metadata_file = en_dir / "metadata.json"

        if en_tokens_file.exists() and en_metadata_file.exists():
            self.en_tokens = MemoryMappedTokens(en_tokens_file, en_metadata_file)
            logger.info(f"Loaded English dataset: {len(self.en_tokens):,} tokens")
        else:
            logger.warning(f"English dataset not found at {en_dir}")

        # Load Portuguese dataset
        pt_tokens_file = pt_dir / "tokens.bin"
        pt_metadata_file = pt_dir / "metadata.json"

        if pt_tokens_file.exists() and pt_metadata_file.exists():
            self.pt_tokens = MemoryMappedTokens(pt_tokens_file, pt_metadata_file)
            logger.info(f"Loaded Portuguese dataset: {len(self.pt_tokens):,} tokens")
        else:
            logger.warning(f"Portuguese dataset not found at {pt_dir}")

        # Ensure at least one dataset is available
        if self.en_tokens is None and self.pt_tokens is None:
            raise FileNotFoundError(
                f"No preprocessed datasets found in {self.data_dir}. "
                f"Run preprocess_datasets.py first."
            )

    def __len__(self) -> int:
        return self._epoch_sequences

    def _sample_language(self) -> str:
        """Sample language based on ratios"""
        if self.en_tokens is None:
            return "pt"
        if self.pt_tokens is None:
            return "en"

        return "en" if self.rng.random() < self.en_ratio_norm else "pt"

    def _get_random_sequence(self, lang: str) -> Tuple[np.ndarray, int]:
        """
        Get a random sequence of max_seq_len + 1 tokens.

        Respects train/val split boundaries:
        - train: samples from [0, train_end)
        - val: samples from [val_start, total_len)

        Returns:
            Tuple of (sequence, valid_length) where valid_length is the number
            of real tokens (excluding padding). Padding positions should use
            ignore_index=-100 in the labels.
        """
        tokens = self.en_tokens if lang == "en" else self.pt_tokens

        if tokens is None:
            tokens = self.pt_tokens if lang == "en" else self.en_tokens
            lang = "pt" if lang == "en" else "en"

        # Determine sampling range based on split
        if self.split == "train":
            range_start = 0
            range_end = self.en_train_end if lang == "en" else self.pt_train_end
        else:
            range_start = self.en_val_start if lang == "en" else self.pt_val_start
            range_end = len(tokens)

        # Random start position within the split range
        max_start = max(range_start, range_end - self.max_seq_len - 1)
        start_idx = self.np_rng.randint(range_start, max_start + 1)

        # Get sequence (max_seq_len + 1 for input/label split)
        # Clamp end to not exceed split boundary
        end_idx = min(start_idx + self.max_seq_len + 1, range_end)
        sequence = tokens.get_chunk(start_idx, end_idx - start_idx)
        valid_length = len(sequence)

        # Pad if necessary
        if len(sequence) < self.max_seq_len + 1:
            pad_length = self.max_seq_len + 1 - len(sequence)
            sequence = np.pad(sequence, (0, pad_length), constant_values=0)

        return sequence, valid_length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training sample.

        Returns:
            Dict with 'input_ids' and 'labels' tensors.
            Labels use -100 (ignore_index) for padding positions to exclude
            them from loss computation.
        """
        # Sample language
        lang = self._sample_language()

        # Get sequence and valid length
        sequence, valid_length = self._get_random_sequence(lang)

        # Split into input and labels
        input_ids = torch.from_numpy(sequence[:-1].astype(np.int64))
        labels = torch.from_numpy(sequence[1:].astype(np.int64))

        # CRITICAL: Set padding positions in labels to -100 (ignore_index)
        # This prevents the model from learning to predict padding tokens
        # valid_length includes the +1 for labels, so valid positions are [0, valid_length-1)
        # Since labels = sequence[1:], valid label positions are [0, valid_length-2]
        if valid_length < self.max_seq_len + 1:
            # Positions from valid_length-1 onwards in labels should be ignored
            # (they correspond to sequence[valid_length:] which are padding)
            labels[valid_length - 1:] = -100

        return {
            'input_ids': input_ids,
            'labels': labels,
        }

    def set_epoch(self, epoch: int):
        """Set epoch for deterministic shuffling"""
        self.rng = random.Random(self.seed + epoch)
        self.np_rng = np.random.RandomState(self.seed + epoch)


class InfiniteDataLoader:
    """
    Wrapper for DataLoader that provides infinite iteration.

    Automatically resets the dataloader when exhausted.
    """

    def __init__(self, dataloader: DataLoader):
        self.dataloader = dataloader
        self.dataset = dataloader.dataset
        self._iterator = None
        self.epoch = 0

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        return self

    def __next__(self) -> Dict[str, torch.Tensor]:
        if self._iterator is None:
            self._reset()

        try:
            return next(self._iterator)
        except StopIteration:
            self.epoch += 1
            self._reset()
            return next(self._iterator)

    def _reset(self):
        """Reset the iterator for a new epoch"""
        if hasattr(self.dataset, 'set_epoch'):
            self.dataset.set_epoch(self.epoch)
        self._iterator = iter(self.dataloader)


def create_dataloader(
    data_dir: str,
    batch_size: int,
    max_seq_len: int = 2048,
    en_ratio: float = 0.5,
    pt_ratio: float = 0.5,
    num_workers: int = 4,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    seed: int = 42,
    epoch_tokens: Optional[int] = None,
    drop_last: bool = True,
    split: str = "train",
) -> InfiniteDataLoader:
    """
    Create an efficient DataLoader for pre-tokenized data.

    Args:
        data_dir: Directory containing preprocessed data
        batch_size: Batch size
        max_seq_len: Maximum sequence length
        en_ratio: English data ratio
        pt_ratio: Portuguese data ratio
        num_workers: Number of worker processes
        pin_memory: Pin memory for faster GPU transfer
        prefetch_factor: Number of batches to prefetch per worker
        seed: Random seed
        epoch_tokens: Total tokens per epoch
        drop_last: Drop last incomplete batch
        split: "train" or "val"

    Returns:
        InfiniteDataLoader instance
    """
    dataset = PreTokenizedDataset(
        data_dir=data_dir,
        max_seq_len=max_seq_len,
        en_ratio=en_ratio,
        pt_ratio=pt_ratio,
        seed=seed,
        epoch_tokens=epoch_tokens,
        split=split,
    )

    # Create DataLoader with optimizations
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
    )

    logger.info(f"Created DataLoader:")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Num workers: {num_workers}")
    logger.info(f"  Pin memory: {pin_memory}")
    logger.info(f"  Prefetch factor: {prefetch_factor}")

    return InfiniteDataLoader(dataloader)


def verify_dataloader(dataloader: InfiniteDataLoader, num_batches: int = 5):
    """Verify dataloader by fetching a few batches"""
    logger.info("Verifying dataloader...")

    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break

        input_ids = batch['input_ids']
        labels = batch['labels']

        logger.info(f"  Batch {i}:")
        logger.info(f"    input_ids shape: {input_ids.shape}")
        logger.info(f"    labels shape: {labels.shape}")
        logger.info(f"    input_ids dtype: {input_ids.dtype}")
        logger.info(f"    Sample tokens: {input_ids[0, :10].tolist()}")

    logger.info("DataLoader verification complete!")


def check_preprocessed_data(data_dir: str) -> bool:
    """
    Check if preprocessed data exists and is valid.

    Args:
        data_dir: Directory to check

    Returns:
        True if valid preprocessed data exists
    """
    data_path = Path(data_dir)

    # Check for at least one dataset
    en_valid = (
        (data_path / "en" / "tokens.bin").exists() and
        (data_path / "en" / "metadata.json").exists()
    )

    pt_valid = (
        (data_path / "pt" / "tokens.bin").exists() and
        (data_path / "pt" / "metadata.json").exists()
    )

    if not en_valid and not pt_valid:
        return False

    # Verify metadata is readable
    try:
        if en_valid:
            with open(data_path / "en" / "metadata.json", 'r') as f:
                en_meta = json.load(f)
                if en_meta.get('total_tokens', 0) == 0:
                    return False

        if pt_valid:
            with open(data_path / "pt" / "metadata.json", 'r') as f:
                pt_meta = json.load(f)
                if pt_meta.get('total_tokens', 0) == 0:
                    return False

    except (json.JSONDecodeError, KeyError):
        return False

    return True


def get_dataset_info(data_dir: str) -> Dict:
    """Get information about preprocessed datasets"""
    data_path = Path(data_dir)
    info = {
        'en_tokens': 0,
        'pt_tokens': 0,
        'en_samples': 0,
        'pt_samples': 0,
        'total_tokens': 0,
    }

    # English dataset
    en_meta_file = data_path / "en" / "metadata.json"
    if en_meta_file.exists():
        with open(en_meta_file, 'r') as f:
            meta = json.load(f)
            info['en_tokens'] = meta.get('total_tokens', 0)
            info['en_samples'] = meta.get('total_samples', 0)

    # Portuguese dataset
    pt_meta_file = data_path / "pt" / "metadata.json"
    if pt_meta_file.exists():
        with open(pt_meta_file, 'r') as f:
            meta = json.load(f)
            info['pt_tokens'] = meta.get('total_tokens', 0)
            info['pt_samples'] = meta.get('total_samples', 0)

    info['total_tokens'] = info['en_tokens'] + info['pt_tokens']

    return info


if __name__ == "__main__":
    # Test the dataloader
    import argparse

    parser = argparse.ArgumentParser(description="Test data loader")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if not check_preprocessed_data(args.data_dir):
        print(f"No preprocessed data found at {args.data_dir}")
        print("Run preprocess_datasets.py first:")
        print(f"  python preprocess_datasets.py --output_dir {args.data_dir}")
        exit(1)

    # Show dataset info
    info = get_dataset_info(args.data_dir)
    print(f"\nDataset Info:")
    print(f"  English tokens: {info['en_tokens']:,}")
    print(f"  Portuguese tokens: {info['pt_tokens']:,}")
    print(f"  Total tokens: {info['total_tokens']:,}")

    # Create and test dataloader
    dataloader = create_dataloader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        num_workers=args.num_workers,
    )

    verify_dataloader(dataloader, num_batches=5)
