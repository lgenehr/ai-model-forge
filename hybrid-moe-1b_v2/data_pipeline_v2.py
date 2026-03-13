#!/usr/bin/env python3
"""
V2-only data pipeline for local, staged, multi-source pretraining.

This module is intentionally isolated from the shared V1 loader. The training
hot path still consumes local memory-mapped token files; Hugging Face access is
handled offline by prepare_gigaverbo_v2.py.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SourceConfig:
    name: str
    path: str
    language: str
    weight: float
    split: str = "train"
    repeat: float = 1.0
    max_tokens: Optional[int] = None
    text_domain: Optional[str] = None


@dataclass(frozen=True)
class StageConfig:
    name: str
    start_step: int
    end_step: Optional[int]
    sources: List[SourceConfig]
    notes: str = ""


class MemoryMappedTokensV2:
    """Small wrapper around a token memmap plus metadata."""

    def __init__(self, tokens_file: Path, metadata_file: Path):
        with metadata_file.open("r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.tokens_file = tokens_file
        self.metadata_file = metadata_file
        dtype_str = self.metadata.get("dtype", "uint16")
        self.dtype = np.uint16 if dtype_str == "uint16" else np.uint32
        self.total_tokens = int(self.metadata.get("total_tokens", 0))
        if self.total_tokens <= 0:
            raise ValueError(f"Invalid total_tokens in {metadata_file}")

        self._mmap = np.memmap(self.tokens_file, dtype=self.dtype, mode="r")
        if len(self._mmap) < self.total_tokens:
            raise ValueError(
                f"Memmap shorter than metadata for {tokens_file}: "
                f"{len(self._mmap)} < {self.total_tokens}"
            )

    def __len__(self) -> int:
        return self.total_tokens

    def get_chunk(self, start: int, length: int) -> np.ndarray:
        end = min(start + length, self.total_tokens)
        return self._mmap[start:end].copy()


@dataclass(frozen=True)
class LoadedSource:
    idx: int
    config: SourceConfig
    tokens: MemoryMappedTokensV2
    range_start: int
    range_end: int
    sample_weight: float

    @property
    def available_tokens(self) -> int:
        return max(0, self.range_end - self.range_start)


def load_data_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid data config: {path}")
    return cfg


def _normalize_sources(source_dicts: Sequence[Dict]) -> List[SourceConfig]:
    sources: List[SourceConfig] = []
    for entry in source_dicts:
        sources.append(
            SourceConfig(
                name=entry["name"],
                path=entry["path"],
                language=entry.get("language", "unknown"),
                weight=float(entry.get("weight", 1.0)),
                split=entry.get("split", "train"),
                repeat=float(entry.get("repeat", 1.0)),
                max_tokens=entry.get("max_tokens"),
                text_domain=entry.get("text_domain"),
            )
        )
    if not sources:
        raise ValueError("No data sources configured for stage.")
    return sources


def parse_stage_configs(data_cfg: Dict) -> List[StageConfig]:
    stage_dicts = data_cfg.get("stages", [])
    if not stage_dicts:
        default_sources = _normalize_sources(data_cfg.get("sources", []))
        return [
            StageConfig(
                name="default",
                start_step=0,
                end_step=None,
                sources=default_sources,
                notes="Single-stage fallback.",
            )
        ]

    stages: List[StageConfig] = []
    for stage in stage_dicts:
        stages.append(
            StageConfig(
                name=stage["name"],
                start_step=int(stage.get("start_step", 0)),
                end_step=stage.get("end_step"),
                sources=_normalize_sources(stage.get("sources", [])),
                notes=stage.get("notes", ""),
            )
        )

    stages.sort(key=lambda item: item.start_step)
    return stages


def resolve_stage(stages: Sequence[StageConfig], step: int) -> StageConfig:
    active = stages[0]
    for stage in stages:
        if step < stage.start_step:
            break
        if stage.end_step is not None and step >= stage.end_step:
            continue
        active = stage
    return active


def _split_bounds(total_tokens: int, split: str, val_fraction: float) -> Tuple[int, int]:
    val_tokens = int(total_tokens * val_fraction)
    train_end = total_tokens - val_tokens
    if split == "train":
        return 0, train_end
    if split == "val":
        return train_end, total_tokens
    raise ValueError(f"Unsupported split: {split}")


class MultiSourceTokenDatasetV2(Dataset):
    """
    Random sequence sampler across multiple local token sources.

    Sampling happens per sequence to avoid building document-level state in the
    hot path. This preserves the throughput characteristics of the original
    memmap-based loader while allowing stage-aware dataset weighting.
    """

    def __init__(
        self,
        *,
        stage: StageConfig,
        max_seq_len: int,
        seed: int,
        epoch_tokens: Optional[int],
        split: str,
        val_fraction: float,
    ):
        self.stage = stage
        self.max_seq_len = max_seq_len
        self.seed = seed
        self.split = split
        self.val_fraction = val_fraction
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        self.sources = self._load_sources(stage.sources, split)

        total_tokens = sum(source.available_tokens for source in self.sources)
        self.total_tokens = total_tokens
        if epoch_tokens:
            self._epoch_sequences = max(1, epoch_tokens // max_seq_len)
        else:
            self._epoch_sequences = max(1, total_tokens // max_seq_len)

        logger.info(
            "V2 dataset stage=%s split=%s sequences=%d total_tokens=%d",
            stage.name,
            split,
            self._epoch_sequences,
            self.total_tokens,
        )
        for source in self.sources:
            logger.info(
                "  source=%s lang=%s weight=%.3f tokens=%d range=[%d,%d)",
                source.config.name,
                source.config.language,
                source.sample_weight,
                source.available_tokens,
                source.range_start,
                source.range_end,
            )

    def _load_sources(self, source_cfgs: Sequence[SourceConfig], split: str) -> List[LoadedSource]:
        sources: List[LoadedSource] = []
        for idx, cfg in enumerate(source_cfgs):
            base = Path(cfg.path)
            tokens_file = base / "tokens.bin"
            metadata_file = base / "metadata.json"
            if not tokens_file.exists() or not metadata_file.exists():
                raise FileNotFoundError(
                    f"Configured V2 dataset source missing files: {base}"
                )
            mm = MemoryMappedTokensV2(tokens_file, metadata_file)
            range_start, range_end = _split_bounds(len(mm), split, self.val_fraction)
            if cfg.max_tokens is not None:
                capped = min(range_end, range_start + int(cfg.max_tokens))
                range_end = max(range_start, capped)
            sample_weight = max(0.0, cfg.weight * max(cfg.repeat, 0.0))
            sources.append(
                LoadedSource(
                    idx=idx,
                    config=cfg,
                    tokens=mm,
                    range_start=range_start,
                    range_end=range_end,
                    sample_weight=sample_weight,
                )
            )

        positive = [source for source in sources if source.available_tokens > self.max_seq_len + 1]
        if not positive:
            raise RuntimeError("No V2 dataset source has enough tokens for one sequence.")
        return positive

    def __len__(self) -> int:
        return self._epoch_sequences

    def _sample_source(self) -> LoadedSource:
        weights = [source.sample_weight for source in self.sources]
        total = sum(weights)
        if total <= 0:
            return self.sources[0]
        choice = self.rng.random() * total
        accum = 0.0
        for source, weight in zip(self.sources, weights):
            accum += weight
            if choice <= accum:
                return source
        return self.sources[-1]

    def _sample_sequence(self, source: LoadedSource) -> Tuple[np.ndarray, int]:
        max_start = max(source.range_start, source.range_end - self.max_seq_len - 1)
        start = int(self.np_rng.randint(source.range_start, max_start + 1))
        end = min(start + self.max_seq_len + 1, source.range_end)
        seq = source.tokens.get_chunk(start, end - start)
        valid_length = len(seq)
        if valid_length < self.max_seq_len + 1:
            seq = np.pad(seq, (0, self.max_seq_len + 1 - valid_length), constant_values=0)
        return seq, valid_length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        source = self._sample_source()
        seq, valid_length = self._sample_sequence(source)
        input_ids = torch.from_numpy(seq[:-1].astype(np.int64))
        labels = torch.from_numpy(seq[1:].astype(np.int64))
        if valid_length < self.max_seq_len + 1:
            labels[valid_length - 1:] = -100
        return {
            "input_ids": input_ids,
            "labels": labels,
            "source_id": torch.tensor(source.idx, dtype=torch.int64),
        }

    def set_epoch(self, epoch: int) -> None:
        self.rng = random.Random(self.seed + epoch)
        self.np_rng = np.random.RandomState(self.seed + epoch)

    def source_name_map(self) -> Dict[int, str]:
        return {source.idx: source.config.name for source in self.sources}


class InfiniteDataLoaderV2:
    def __init__(self, dataloader: DataLoader):
        self.dataloader = dataloader
        self.dataset = dataloader.dataset
        self._iterator: Optional[Iterator[Dict[str, torch.Tensor]]] = None
        self.epoch = 0

    def __iter__(self) -> "InfiniteDataLoaderV2":
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

    def _reset(self) -> None:
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(self.epoch)
        self._iterator = iter(self.dataloader)


def create_dataloader_v2(
    *,
    stage: StageConfig,
    batch_size: int,
    max_seq_len: int,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int,
    seed: int,
    epoch_tokens: Optional[int],
    split: str,
    val_fraction: float,
    drop_last: bool,
) -> Tuple[InfiniteDataLoaderV2, Dict[int, str]]:
    dataset = MultiSourceTokenDatasetV2(
        stage=stage,
        max_seq_len=max_seq_len,
        seed=seed,
        epoch_tokens=epoch_tokens,
        split=split,
        val_fraction=val_fraction,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        drop_last=drop_last,
    )
    return InfiniteDataLoaderV2(dataloader), dataset.source_name_map()


def build_fixed_validation_batches_v2(
    *,
    data_cfg: Dict,
    batch_size: int,
    max_seq_len: int,
    seed: int,
    n_batches: int,
) -> Tuple[List[Dict[str, torch.Tensor]], Dict[int, str], str]:
    val_fraction = float(data_cfg.get("validation_fraction", 0.005))
    validation = data_cfg.get("validation", {})
    if validation.get("sources"):
        stage = StageConfig(
            name=validation.get("name", "validation"),
            start_step=0,
            end_step=None,
            sources=_normalize_sources(validation.get("sources", [])),
            notes=validation.get("notes", ""),
        )
    else:
        stage = resolve_stage(parse_stage_configs(data_cfg), step=0)

    dataset = MultiSourceTokenDatasetV2(
        stage=stage,
        max_seq_len=max_seq_len,
        seed=seed,
        epoch_tokens=None,
        split="val",
        val_fraction=val_fraction,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
    )

    fixed_batches: List[Dict[str, torch.Tensor]] = []
    for idx, batch in enumerate(loader):
        if idx >= n_batches:
            break
        fixed_batches.append(
            {
                "input_ids": batch["input_ids"].clone(),
                "labels": batch["labels"].clone(),
                "source_id": batch["source_id"].clone(),
            }
        )
    total_tokens = sum(int((b["labels"] != -100).sum().item()) for b in fixed_batches)
    logger.info(
        "V2 validation subset fixed: stage=%s batches=%d tokens=%d seed=%d",
        stage.name,
        len(fixed_batches),
        total_tokens,
        seed,
    )
    return fixed_batches, dataset.source_name_map(), stage.name


def get_dataset_info_v2(data_cfg: Dict) -> Dict:
    info = {
        "total_tokens": 0,
        "sources": [],
        "stages": [],
    }
    for stage in parse_stage_configs(data_cfg):
        stage_info = {"name": stage.name, "sources": []}
        for source in stage.sources:
            base = Path(source.path)
            meta_file = base / "metadata.json"
            if not meta_file.exists():
                source_tokens = 0
            else:
                with meta_file.open("r", encoding="utf-8") as f:
                    meta = json.load(f)
                source_tokens = int(meta.get("total_tokens", 0))
            stage_info["sources"].append(
                {
                    "name": source.name,
                    "language": source.language,
                    "weight": source.weight,
                    "repeat": source.repeat,
                    "path": source.path,
                    "tokens": source_tokens,
                }
            )
            info["total_tokens"] += source_tokens
        info["stages"].append(stage_info)
    return info


def format_stage_mix(stage: StageConfig) -> str:
    total = sum(max(0.0, source.weight * max(source.repeat, 0.0)) for source in stage.sources)
    parts = []
    for source in stage.sources:
        effective = max(0.0, source.weight * max(source.repeat, 0.0))
        pct = (100.0 * effective / total) if total > 0 else 0.0
        parts.append(f"{source.name}={pct:.1f}%")
    return ", ".join(parts)


def get_missing_source_paths(data_cfg: Dict) -> List[str]:
    missing: List[str] = []
    seen = set()
    all_source_dicts: List[Dict] = []
    all_source_dicts.extend(data_cfg.get("sources", []))
    for stage in data_cfg.get("stages", []):
        all_source_dicts.extend(stage.get("sources", []))
    validation = data_cfg.get("validation", {})
    all_source_dicts.extend(validation.get("sources", []))

    for source in all_source_dicts:
        path = Path(source["path"])
        tokens = path / "tokens.bin"
        meta = path / "metadata.json"
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        if not tokens.exists() or not meta.exists():
            missing.append(str(path))
    return missing
