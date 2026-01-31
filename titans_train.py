#!/usr/bin/env python3
"""Train a Titans Memory-As-Context model on bilingual EN/PT-BR data.

This script builds a project folder, streams datasets, and trains with
bfloat16 AMP, torch.compile, and gradient accumulation to fit on a
consumer GPU.
"""

from __future__ import annotations

import csv
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

import torch
import torch.nn.functional as F
from datasets import load_dataset
from titans_pytorch import MemoryAsContextTransformer
import tiktoken


@dataclass
class TrainConfig:
    segment_len: int = 128
    num_persist_mem_tokens: int = 4
    num_longterm_mem_tokens: int = 128
    vocab_size: int = 50257
    dim: int = 768
    depth: int = 12
    heads: int = 12
    dim_head: int = 64
    ff_mult: int = 4
    dropout: float = 0.1
    micro_batch_size: int = 4
    grad_accum_steps: int = 8
    total_steps: int = 1000
    lr: float = 3e-4
    weight_decay: float = 0.1
    log_every: int = 10
    checkpoint_every: int = 200
    seed: int = 1337


def build_project_dir() -> Path:
    root = Path(os.getcwd()) / "ai-model-forge" / "titans-architecture-test"
    root.mkdir(parents=True, exist_ok=True)
    return root


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_en_stream() -> Iterable[Dict[str, str]]:
    return load_dataset(
        "HuggingFaceFW/fineweb-edu",
        "sample-10BT",
        streaming=True,
        split="train",
    )


def load_pt_stream() -> Iterable[Dict[str, str]]:
    try:
        return load_dataset(
            "wikipedia",
            "20220301.pt",
            streaming=True,
            split="train",
        )
    except Exception:
        return load_dataset(
            "uonlp/CulturaX",
            "pt",
            streaming=True,
            split="train",
        )


def interleaved_samples(
    en_stream: Iterable[Dict[str, str]],
    pt_stream: Iterable[Dict[str, str]],
    mix_prob: float = 0.5,
) -> Iterator[Dict[str, str]]:
    en_iter = iter(en_stream)
    pt_iter = iter(pt_stream)
    while True:
        if random.random() < mix_prob:
            yield next(en_iter)
        else:
            yield next(pt_iter)


def extract_text(sample: Dict[str, str]) -> str:
    for key in ("text", "content", "article", "body"):
        if key in sample and sample[key]:
            return sample[key]
    return ""


def batch_token_stream(
    samples: Iterable[Dict[str, str]],
    tokenizer: tiktoken.Encoding,
    batch_size: int,
    seq_len: int,
) -> Iterator[torch.Tensor]:
    buffer: List[int] = []
    needed = batch_size * seq_len + 1
    for sample in samples:
        text = extract_text(sample)
        if not text:
            continue
        buffer.extend(tokenizer.encode(text))
        while len(buffer) >= needed:
            chunk = buffer[:needed]
            buffer = buffer[batch_size * seq_len :]
            tokens = torch.tensor(chunk, dtype=torch.long)
            yield tokens


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    step: int,
) -> None:
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "step": step,
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
) -> int:
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler.load_state_dict(checkpoint["scaler"])
    return int(checkpoint.get("step", 0))


def append_log(log_path: Path, step: int, loss: float) -> None:
    file_exists = log_path.exists()
    with log_path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if not file_exists:
            writer.writerow(["step", "loss"])
        writer.writerow([step, loss])


def main() -> None:
    config = TrainConfig()
    set_seed(config.seed)

    project_dir = build_project_dir()
    checkpoint_path = project_dir / "checkpoint.pt"
    log_path = project_dir / "training_log.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MemoryAsContextTransformer(
        dim=config.dim,
        depth=config.depth,
        heads=config.heads,
        dim_head=config.dim_head,
        ff_mult=config.ff_mult,
        vocab_size=config.vocab_size,
        dropout=config.dropout,
        segment_len=config.segment_len,
        num_persist_mem_tokens=config.num_persist_mem_tokens,
        num_longterm_mem_tokens=config.num_longterm_mem_tokens,
    ).to(device)

    if hasattr(torch, "compile"):
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=False)

    start_step = 0
    if checkpoint_path.exists():
        start_step = load_checkpoint(checkpoint_path, model, optimizer, scaler)

    tokenizer = tiktoken.get_encoding("gpt2")
    en_stream = load_en_stream()
    pt_stream = load_pt_stream()
    mixed_samples = interleaved_samples(en_stream, pt_stream, mix_prob=0.5)
    token_stream = batch_token_stream(
        mixed_samples,
        tokenizer,
        config.micro_batch_size,
        config.segment_len,
    )

    model.train()
    optimizer.zero_grad(set_to_none=True)

    running_loss = 0.0
    start_time = time.time()

    for step in range(start_step, config.total_steps):
        accum_loss = 0.0
        for _ in range(config.grad_accum_steps):
            batch_tokens = next(token_stream)
            batch_tokens = batch_tokens.to(device)
            input_ids = batch_tokens[:-1]
            target_ids = batch_tokens[1:]
            input_ids = input_ids.view(config.micro_batch_size, config.segment_len)
            target_ids = target_ids.view(config.micro_batch_size, config.segment_len)

            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = model(input_ids)
                loss = F.cross_entropy(
                    logits.reshape(-1, config.vocab_size),
                    target_ids.reshape(-1),
                )

            loss = loss / config.grad_accum_steps
            loss.backward()
            accum_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        running_loss += accum_loss

        if (step + 1) % config.log_every == 0:
            avg_loss = running_loss / config.log_every
            running_loss = 0.0
            append_log(log_path, step + 1, avg_loss)
            elapsed = time.time() - start_time
            print(f"step={step + 1} loss={avg_loss:.4f} elapsed={elapsed:.1f}s")
            start_time = time.time()

        if (step + 1) % config.checkpoint_every == 0:
            save_checkpoint(checkpoint_path, model, optimizer, scaler, step + 1)

    save_checkpoint(checkpoint_path, model, optimizer, scaler, config.total_steps)


if __name__ == "__main__":
    main()
