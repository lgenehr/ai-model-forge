#!/usr/bin/env python3
"""
Training script for Hybrid Transformer-Mamba-MoE (~820M parameters).

Features:
  • BF16 mixed precision (autocast)
  • Gradient checkpointing (saves ~10× activation memory)
  • Gradient accumulation (configurable effective batch size)
  • Cosine LR schedule with linear warmup
  • AdamW with parameter-group weight decay
  • torch.compile (reduce-overhead mode)
  • Checkpointing + resume
  • W&B logging (optional)
  • tokens/sec throughput tracking

Usage:
    python train.py                           # use config.yaml defaults
    python train.py --d_model 1536            # override any config key
    python train.py --resume model_1b/ckpt/latest.pt
    python train.py --wandb --wandb_run_name my-run
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import yaml

# Add parent dir so we can import the existing data_loader
sys.path.insert(0, str(Path(__file__).parent.parent / "bitnet-mamba-hybrid"))
from data_loader import create_dataloader, get_dataset_info  # type: ignore

from model import HybridMoEModel, ModelConfig, estimate_memory_gb

logger = logging.getLogger(__name__)


# ─── Config helpers ───────────────────────────────────────────────────────────

def load_yaml_config(path: str) -> Dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_config_from_args(args: argparse.Namespace) -> Tuple[ModelConfig, Dict]:
    """Merge config.yaml + CLI overrides into ModelConfig + training dict."""
    cfg_path = Path(__file__).parent / "config.yaml"
    raw = load_yaml_config(str(cfg_path))
    model_raw = raw.get("model", {})
    train_raw = raw.get("training", {})

    # CLI overrides (only keys that were explicitly set)
    cli = vars(args)
    model_keys = {f.name for f in ModelConfig.__dataclass_fields__.values()}
    for k, v in cli.items():
        if v is None:
            continue
        if k in model_keys:
            model_raw[k] = v
        else:
            train_raw[k] = v

    cfg = ModelConfig(**{k: v for k, v in model_raw.items()
                         if k in model_keys})
    return cfg, train_raw


# ─── LR schedule ──────────────────────────────────────────────────────────────

def get_lr(step: int, warmup_steps: int, max_steps: int,
           peak_lr: float, min_lr: float) -> float:
    """Linear warmup → cosine decay to min_lr."""
    if step < warmup_steps:
        return peak_lr * step / max(1, warmup_steps)
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (peak_lr - min_lr)


# ─── Optimizer ────────────────────────────────────────────────────────────────

def build_optimizer(model: HybridMoEModel, train_cfg: Dict) -> torch.optim.Optimizer:
    """
    AdamW with 4 parameter groups:
      1. weight-decayed matrices (attention, MoE projections, Mamba projections)
      2. no weight decay: biases, norms, SSM-specific params (A_log, D, dt_bias)
    """
    decay_params = []
    no_decay_params = []

    no_wd_names = {"bias", "norm", "A_log", ".D", "embed_tokens"}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(nd in name for nd in no_wd_names) or param.ndim < 2:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params,    "weight_decay": train_cfg["weight_decay"]},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    opt = torch.optim.AdamW(
        param_groups,
        lr=train_cfg["lr"],
        betas=(train_cfg.get("beta1", 0.9), train_cfg.get("beta2", 0.95)),
        eps=train_cfg.get("adam_eps", 1e-8),
        fused=True,  # PyTorch 2.x fused CUDA kernel (faster, less overhead)
    )
    logger.info(f"Optimizer: AdamW fused  |  "
                f"decay params: {sum(p.numel() for p in decay_params):,}  |  "
                f"no-decay params: {sum(p.numel() for p in no_decay_params):,}")
    return opt


# ─── Checkpointing ────────────────────────────────────────────────────────────

def save_checkpoint(
    model: HybridMoEModel,
    optimizer: torch.optim.Optimizer,
    step: int,
    best_val_loss: float,
    train_cfg: Dict,
    output_dir: Path,
    is_best: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "step":          step,
        "best_val_loss": best_val_loss,
        "model":         model.state_dict(),
        "optimizer":     optimizer.state_dict(),
        "model_config":  model.config.__dict__,
        "train_cfg":     train_cfg,
    }
    ckpt_path = output_dir / f"step_{step:08d}.pt"
    torch.save(state, ckpt_path)

    # Symlink latest / best
    link_latest = output_dir / "latest.pt"
    if link_latest.is_symlink() or link_latest.exists():
        link_latest.unlink()
    link_latest.symlink_to(ckpt_path.name)

    if is_best:
        link_best = output_dir / "best.pt"
        if link_best.is_symlink() or link_best.exists():
            link_best.unlink()
        link_best.symlink_to(ckpt_path.name)

    logger.info(f"Checkpoint saved → {ckpt_path}  (best={is_best})")


def load_checkpoint(
    path: str,
    model: HybridMoEModel,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
) -> Tuple[int, float]:
    """Load checkpoint; returns (step, best_val_loss)."""
    logger.info(f"Loading checkpoint: {path}")
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    step          = state.get("step", 0)
    best_val_loss = state.get("best_val_loss", float("inf"))
    logger.info(f"  Resumed from step {step}  |  best_val_loss={best_val_loss:.4f}")
    return step, best_val_loss


# ─── Evaluation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: HybridMoEModel,
    val_loader,
    device: torch.device,
    dtype: torch.dtype,
    n_batches: int = 50,
) -> float:
    """Compute mean validation loss over n_batches."""
    model.eval()
    total_loss = 0.0
    count = 0
    for batch in val_loader:
        if count >= n_batches:
            break
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels    = batch["labels"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=dtype):
            _, loss = model(input_ids, targets=labels, use_checkpoint=False)
        if loss is not None:
            total_loss += loss.item()
            count += 1
    model.train()
    return total_loss / max(1, count)


# ─── Setup logging ────────────────────────────────────────────────────────────

def setup_logging(output_dir: Path, level: int = logging.INFO) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(output_dir / "train.log"),
    ]
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


# ─── Main training loop ───────────────────────────────────────────────────────

def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    cfg, train_cfg = build_config_from_args(args)

    output_dir = Path(train_cfg.get("output_dir", "model_1b"))
    ckpt_dir   = output_dir / "checkpoints"
    setup_logging(output_dir)

    logger.info("=" * 72)
    logger.info("Hybrid Transformer-Mamba-MoE  |  Training")
    logger.info("=" * 72)

    # ── Device setup ──────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.bfloat16 if train_cfg.get("dtype", "bfloat16") == "bfloat16" else torch.float16
    logger.info(f"Device: {device}  |  dtype: {dtype}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}  |  "
                    f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True

    torch.manual_seed(train_cfg.get("seed", 42))

    # ── Data ──────────────────────────────────────────────────────────────────
    data_dir = train_cfg.get("data_dir",
        str(Path(__file__).parent.parent / "bitnet-mamba-hybrid" / "data" / "tokenized"))

    info = get_dataset_info(data_dir)
    logger.info(f"Dataset  EN: {info['en_tokens']:,} tokens  "
                f"|  PT: {info['pt_tokens']:,} tokens")

    batch_size  = train_cfg.get("batch_size", 2)
    seq_len     = cfg.max_seq_len
    grad_accum  = train_cfg.get("grad_accum", 16)
    num_workers = train_cfg.get("num_workers", 4)

    effective_batch_tokens = batch_size * seq_len * grad_accum
    logger.info(f"Batch: {batch_size}  |  grad_accum: {grad_accum}  |  "
                f"effective: {effective_batch_tokens:,} tokens/step")

    train_loader = create_dataloader(
        data_dir=data_dir,
        batch_size=batch_size,
        max_seq_len=seq_len,
        en_ratio=train_cfg.get("en_ratio", 0.3),
        pt_ratio=train_cfg.get("pt_ratio", 0.7),
        num_workers=num_workers,
        seed=train_cfg.get("seed", 42),
        split="train",
    )
    val_loader = create_dataloader(
        data_dir=data_dir,
        batch_size=batch_size,
        max_seq_len=seq_len,
        en_ratio=train_cfg.get("en_ratio", 0.3),
        pt_ratio=train_cfg.get("pt_ratio", 0.7),
        num_workers=min(num_workers, 2),
        seed=train_cfg.get("seed", 42) + 999,
        split="val",
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    logger.info("Building model…")
    model = HybridMoEModel(cfg).to(device)

    mem = estimate_memory_gb(cfg, batch_size, seq_len)
    logger.info(f"Estimated VRAM: {mem}")
    if device.type == "cuda" and mem["total_gb"] > torch.cuda.get_device_properties(0).total_memory / 1e9 * 0.95:
        logger.warning("⚠  Estimated VRAM may exceed GPU capacity. "
                       "Consider reducing batch_size or n_experts.")

    use_grad_ckpt = train_cfg.get("gradient_checkpointing", True)
    use_compile   = train_cfg.get("compile", True)

    if use_compile:
        from model import HAS_MAMBA_SSM
        if HAS_MAMBA_SSM:
            logger.warning(
                "torch.compile is disabled: mamba-ssm CUDA extension (selective_scan_cuda) "
                "is not torch.compile-compatible. Training is already fast (~6K tok/s) "
                "without compilation. Remove '--compile' or set 'compile: false' in config.yaml."
            )
        else:
            logger.info("Compiling model with torch.compile(mode='reduce-overhead')…")
            model = torch.compile(model, mode="reduce-overhead", dynamic=False)  # type: ignore[assignment]

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = build_optimizer(model, train_cfg)  # type: ignore[arg-type]

    # ── Resume ────────────────────────────────────────────────────────────────
    start_step    = 0
    best_val_loss = float("inf")
    resume_path   = train_cfg.get("resume", None)
    if args.resume:
        resume_path = args.resume
    if resume_path and Path(resume_path).exists():
        start_step, best_val_loss = load_checkpoint(
            resume_path, model, optimizer, device  # type: ignore[arg-type]
        )

    # ── W&B ───────────────────────────────────────────────────────────────────
    wandb_run = None
    if train_cfg.get("wandb", False):
        try:
            import wandb
            wandb_run = wandb.init(
                project=train_cfg.get("wandb_project", "hybrid-moe-1b"),
                name=train_cfg.get("wandb_run_name", None),
                config={**cfg.__dict__, **train_cfg},
                resume="allow",
            )
            logger.info("W&B initialized ✓")
        except Exception as e:
            logger.warning(f"W&B init failed: {e}")

    # ── Training state ────────────────────────────────────────────────────────
    max_steps   = train_cfg.get("max_steps", 200_000)
    warmup      = train_cfg.get("warmup_steps", 2_000)
    peak_lr     = train_cfg.get("lr", 3e-4)
    min_lr      = train_cfg.get("min_lr", 3e-5)
    grad_clip   = train_cfg.get("grad_clip", 1.0)
    log_every   = train_cfg.get("log_every", 10)
    eval_every  = train_cfg.get("eval_every", 500)
    save_every  = train_cfg.get("save_every", 1000)

    scaler = torch.amp.GradScaler(device="cuda", enabled=(dtype == torch.float16))

    model.train()
    train_iter = iter(train_loader)
    optimizer.zero_grad(set_to_none=True)

    step               = start_step
    micro_step         = 0
    accum_loss         = 0.0
    accum_tokens       = 0
    t_last             = time.perf_counter()
    tokens_since_log   = 0
    grad_norm_ema      = 0.0   # exponential moving average for monitoring

    logger.info(f"Starting training from step {step} → {max_steps}")
    logger.info(f"gradient_checkpointing={use_grad_ckpt}  |  compile={use_compile}")

    while step < max_steps:
        # ── Update LR ─────────────────────────────────────────────────────────
        lr = get_lr(step, warmup, max_steps, peak_lr, min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # ── Micro-batch accumulation loop ─────────────────────────────────────
        for micro in range(grad_accum):
            batch = next(train_iter)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels    = batch["labels"].to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=dtype):
                _, loss = model(input_ids, targets=labels, use_checkpoint=use_grad_ckpt)

            # Scale loss by grad_accum so gradients average correctly
            loss_scaled = loss / grad_accum
            scaler.scale(loss_scaled).backward()

            accum_loss += loss.item()
            tokens_in_batch = (labels != -100).sum().item()
            accum_tokens       += tokens_in_batch
            tokens_since_log   += tokens_in_batch

        # ── Optimizer step ────────────────────────────────────────────────────
        scaler.unscale_(optimizer)
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        step += 1
        grad_norm_ema = 0.9 * grad_norm_ema + 0.1 * grad_norm.item()

        # ── Logging ───────────────────────────────────────────────────────────
        if step % log_every == 0:
            t_now  = time.perf_counter()
            elapsed = t_now - t_last
            tok_per_sec = tokens_since_log / max(elapsed, 1e-6)
            avg_loss    = accum_loss / log_every

            logger.info(
                f"step {step:>7d}/{max_steps}  |  "
                f"loss: {avg_loss:.4f}  |  "
                f"lr: {lr:.2e}  |  "
                f"grad_norm: {grad_norm_ema:.3f}  |  "
                f"tok/s: {tok_per_sec:,.0f}  |  "
                f"mem: {torch.cuda.memory_allocated()/1e9:.1f}GB"
            )

            if wandb_run is not None:
                wandb_run.log({
                    "train/loss": avg_loss,
                    "train/lr":   lr,
                    "train/grad_norm": grad_norm_ema,
                    "perf/tok_per_sec": tok_per_sec,
                    "perf/mem_gb": torch.cuda.memory_allocated() / 1e9,
                    "step": step,
                })

            accum_loss       = 0.0
            tokens_since_log = 0
            t_last           = t_now

        # ── Evaluation ────────────────────────────────────────────────────────
        if step % eval_every == 0:
            val_loss = evaluate(model, val_loader, device, dtype,  # type: ignore[arg-type]
                                n_batches=train_cfg.get("eval_batches", 50))
            is_best  = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss

            logger.info(
                f"[EVAL]  step {step}  |  val_loss: {val_loss:.4f}  "
                f"|  best: {best_val_loss:.4f}  {'★' if is_best else ''}"
            )
            if wandb_run is not None:
                wandb_run.log({"val/loss": val_loss, "val/best_loss": best_val_loss, "step": step})

            # Always save on eval; also mark best
            raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            save_checkpoint(raw_model, optimizer, step, best_val_loss, train_cfg, ckpt_dir, is_best)  # type: ignore[arg-type]

        # ── Periodic save ─────────────────────────────────────────────────────
        elif step % save_every == 0:
            raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            save_checkpoint(raw_model, optimizer, step, best_val_loss, train_cfg, ckpt_dir, False)  # type: ignore[arg-type]

    logger.info("Training complete.")
    if wandb_run is not None:
        wandb_run.finish()


# ─── Argument parser ──────────────────────────────────────────────────────────

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train Hybrid Transformer-Mamba-MoE LLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model architecture overrides
    arch = p.add_argument_group("Model Architecture (override config.yaml)")
    arch.add_argument("--d_model",          type=int,   default=None)
    arch.add_argument("--n_layers",         type=int,   default=None)
    arch.add_argument("--n_heads",          type=int,   default=None)
    arch.add_argument("--n_kv_heads",       type=int,   default=None)
    arch.add_argument("--mamba_d_state",    type=int,   default=None)
    arch.add_argument("--mamba_expand",     type=int,   default=None)
    arch.add_argument("--moe_n_experts",    type=int,   default=None)
    arch.add_argument("--moe_top_k",        type=int,   default=None)
    arch.add_argument("--moe_intermediate", type=int,   default=None)
    arch.add_argument("--vocab_size",       type=int,   default=None)
    arch.add_argument("--max_seq_len",      type=int,   default=None)

    # Training overrides
    train = p.add_argument_group("Training (override config.yaml)")
    train.add_argument("--batch_size",      type=int,   default=None)
    train.add_argument("--grad_accum",      type=int,   default=None)
    train.add_argument("--max_steps",       type=int,   default=None)
    train.add_argument("--warmup_steps",    type=int,   default=None)
    train.add_argument("--lr",              type=float, default=None)
    train.add_argument("--min_lr",          type=float, default=None)
    train.add_argument("--weight_decay",    type=float, default=None)
    train.add_argument("--grad_clip",       type=float, default=None)
    train.add_argument("--seed",            type=int,   default=None)
    train.add_argument("--data_dir",        type=str,   default=None)
    train.add_argument("--output_dir",      type=str,   default=None)
    train.add_argument("--en_ratio",        type=float, default=None)
    train.add_argument("--pt_ratio",        type=float, default=None)
    train.add_argument("--resume",          type=str,   default=None,
                       help="Path to checkpoint to resume from")

    # Feature flags
    feat = p.add_argument_group("Features")
    feat.add_argument("--no_compile",          action="store_true", help="Disable torch.compile")
    feat.add_argument("--no_grad_checkpoint",  action="store_true", help="Disable gradient checkpointing")
    feat.add_argument("--wandb",               action="store_true", help="Enable W&B logging")
    feat.add_argument("--wandb_project",       type=str, default=None)
    feat.add_argument("--wandb_run_name",      type=str, default=None)
    feat.add_argument("--dtype",               type=str, default=None, choices=["bfloat16", "float16"])

    return p


if __name__ == "__main__":
    main()
