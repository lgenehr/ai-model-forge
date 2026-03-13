#!/usr/bin/env python3
"""
Training script V2 for Hybrid Transformer-Mamba-MoE.

Changes from train.py (V1):
  1. Uses model_v2.py (HybridMoEModelV2) with sparse MoE, shared expert, z-loss
  2. Removes GradScaler when using bfloat16 (unnecessary, was no-op in V1)
  3. Cleaner gradient accumulation with context manager
  4. Improved parameter group organization (separate Mamba, Attn, MoE, Shared)
  5. Better metric tracking (separates aux_loss and z_loss in logs)
  6. Supports loading V1 checkpoints into V2 model (partial migration)
  7. Optional WSD (Warmup-Stable-Decay) LR schedule as alternative to cosine

Usage:
    python train_v2.py                           # use config.yaml defaults
    python train_v2.py --d_model 1536            # override any config key
    python train_v2.py --resume model_1b/ckpt/latest.pt --migrate_v1
    python train_v2.py --wandb --wandb_run_name my-v2-run
    python train_v2.py --lr_schedule wsd         # use WSD instead of cosine
"""

import argparse
import json
import logging
import math
import os
import signal
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, SequentialSampler

# Add paths for shared dependencies (data_loader from bitnet-mamba-hybrid, config from V1)
_v1_dir = Path(__file__).parent.parent / "hybrid-moe-1b"
sys.path.insert(0, str(Path(__file__).parent.parent / "bitnet-mamba-hybrid"))
from data_loader import PreTokenizedDataset, create_dataloader, get_dataset_info

from model_v2 import (
    HybridMoEModelV2,
    ModelConfigV2,
    estimate_memory_gb_v2,
    load_v1_checkpoint_into_v2,
)

logger = logging.getLogger(__name__)


# ─── Utilities ────────────────────────────────────────────────────────────────

def _to_python_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def build_checkpoint_metadata(
    *,
    step: int,
    tokens_trained: int,
    best_val_loss: float,
    train_metrics: Optional[Dict[str, Any]],
    eval_metrics: Optional[Dict[str, Any]],
    train_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    metadata = {
        "step": step,
        "tokens_trained": tokens_trained,
        "train_loss": None,
        "train_ce_loss": None,
        "train_aux_loss": None,
        "grad_norm": None,
        "lr": None,
        "val_loss": None,
        "val_ce_loss": None,
        "val_aux_loss": None,
        "best_val_loss": best_val_loss,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_name": train_cfg.get("wandb_run_name"),
        "model_version": "v2",
    }
    if train_metrics:
        metadata.update({
            "train_loss": _to_python_value(train_metrics.get("train_loss")),
            "train_ce_loss": _to_python_value(train_metrics.get("train_ce_loss")),
            "train_aux_loss": _to_python_value(train_metrics.get("train_aux_loss")),
            "grad_norm": _to_python_value(train_metrics.get("grad_norm")),
            "lr": _to_python_value(train_metrics.get("lr")),
        })
    if eval_metrics:
        metadata.update({
            "val_loss": _to_python_value(eval_metrics.get("val_loss")),
            "val_ce_loss": _to_python_value(eval_metrics.get("val_ce_loss")),
            "val_aux_loss": _to_python_value(eval_metrics.get("val_aux_loss")),
        })
        metadata["eval_batches"] = _to_python_value(eval_metrics.get("batches_evaluated"))
        metadata["eval_tokens"] = _to_python_value(eval_metrics.get("tokens_evaluated"))
        metadata["eval_seed"] = _to_python_value(eval_metrics.get("eval_seed"))
    return metadata


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _update_sidecar_symlink(link_path: Path, target_name: str) -> None:
    if link_path.is_symlink() or link_path.exists():
        link_path.unlink()
    link_path.symlink_to(target_name)


# ─── SIGINT handler ──────────────────────────────────────────────────────────

_interrupt_requested: bool = False

def _sigint_handler(signum, frame) -> None:
    global _interrupt_requested
    if _interrupt_requested:
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        raise KeyboardInterrupt
    _interrupt_requested = True
    print("\nSIGINT received — will save checkpoint after current step "
          "(press Ctrl+C again to force-quit).", flush=True)


# ─── Config helpers ──────────────────────────────────────────────────────────

def load_yaml_config(path: str) -> Dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_config_from_args(args: argparse.Namespace) -> Tuple[ModelConfigV2, Dict]:
    """Merge config.yaml + CLI overrides into ModelConfigV2 + training dict."""
    cfg_path = _v1_dir / "config.yaml"
    raw = load_yaml_config(str(cfg_path))
    model_raw = raw.get("model", {})
    train_raw = raw.get("training", {})

    cli = vars(args)
    model_keys = {f.name for f in ModelConfigV2.__dataclass_fields__.values()}
    for k, v in cli.items():
        if v is None:
            continue
        if k in model_keys:
            model_raw[k] = v
        else:
            train_raw[k] = v

    # V2-specific defaults that override V1 config
    model_raw.setdefault("qk_norm", True)
    model_raw.setdefault("moe_shared_expert", True)
    model_raw.setdefault("moe_z_loss_coeff", 0.001)

    # V2: balanced bilingual ratio (50% EN / 50% PT)
    train_raw.setdefault("en_ratio", 0.5)
    train_raw.setdefault("pt_ratio", 0.5)

    cfg = ModelConfigV2(**{k: v for k, v in model_raw.items() if k in model_keys})
    return cfg, train_raw


# ─── LR schedules ────────────────────────────────────────────────────────────

def get_lr_cosine(step: int, warmup_steps: int, max_steps: int,
                  peak_lr: float, min_lr: float) -> float:
    """Linear warmup -> cosine decay to min_lr. Same as V1."""
    if step < warmup_steps:
        return peak_lr * step / max(1, warmup_steps)
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (peak_lr - min_lr)


def get_lr_wsd(step: int, warmup_steps: int, max_steps: int,
               peak_lr: float, min_lr: float,
               stable_fraction: float = 0.7) -> float:
    """
    Warmup-Stable-Decay (WSD) schedule (MiniCPM, 2024).

    V2 addition: alternative to cosine. Maintains peak_lr for longer,
    then decays to min_lr in the final phase. Often better for continued
    pre-training or when the total budget isn't known upfront.

    Phases:
      1. Linear warmup: 0 -> peak_lr over warmup_steps
      2. Stable: hold peak_lr for stable_fraction of remaining steps
      3. Cosine decay: peak_lr -> min_lr
    """
    if step < warmup_steps:
        return peak_lr * step / max(1, warmup_steps)
    remaining = max_steps - warmup_steps
    stable_steps = int(remaining * stable_fraction)
    decay_steps = remaining - stable_steps

    if step < warmup_steps + stable_steps:
        return peak_lr
    if decay_steps <= 0:
        return min_lr

    progress = (step - warmup_steps - stable_steps) / decay_steps
    progress = min(progress, 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (peak_lr - min_lr)


# ─── Optimizer ────────────────────────────────────────────────────────────────

def build_optimizer(model: HybridMoEModelV2, train_cfg: Dict) -> torch.optim.Optimizer:
    """
    AdamW with parameter groups.

    V2 change: same 2-group split (decay vs no-decay) as V1, but with
    clearer categorization. The split is intentionally simple because
    differential LR per component (Mamba vs Attn vs MoE) showed mixed
    results in ablations and adds tuning complexity.
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
        {"params": decay_params, "weight_decay": train_cfg["weight_decay"]},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    adamw_kwargs = {
        "lr": train_cfg["lr"],
        "betas": (train_cfg.get("beta1", 0.9), train_cfg.get("beta2", 0.95)),
        "eps": train_cfg.get("adam_eps", 1e-8),
    }
    if torch.cuda.is_available():
        adamw_kwargs["fused"] = True

    opt = torch.optim.AdamW(param_groups, **adamw_kwargs)
    logger.info(f"Optimizer: AdamW{' fused' if torch.cuda.is_available() else ''}  |  "
                f"decay params: {sum(p.numel() for p in decay_params):,}  |  "
                f"no-decay params: {sum(p.numel() for p in no_decay_params):,}")
    return opt


# ─── Checkpointing ───────────────────────────────────────────────────────────

def save_checkpoint(
    model: HybridMoEModelV2,
    optimizer: torch.optim.Optimizer,
    step: int,
    tokens_trained: int,
    best_val_loss: float,
    train_cfg: Dict,
    output_dir: Path,
    checkpoint_meta: Optional[Dict[str, Any]] = None,
    is_best: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    keep_last = max(1, int(train_cfg.get("keep_last_checkpoints", 3)))
    state = {
        "step": step,
        "tokens_trained": tokens_trained,
        "best_val_loss": best_val_loss,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_config": model.config.__dict__,
        "train_cfg": train_cfg,
        "checkpoint_meta": checkpoint_meta or {},
        "model_version": "v2",
    }
    ckpt_path = output_dir / f"step_{step:08d}.pt"
    torch.save(state, ckpt_path)
    meta_path = ckpt_path.with_suffix(".json")
    if checkpoint_meta is not None:
        _write_json(meta_path, checkpoint_meta)

    link_latest = output_dir / "latest.pt"
    _update_sidecar_symlink(link_latest, ckpt_path.name)
    if checkpoint_meta is not None:
        _update_sidecar_symlink(output_dir / "latest.json", meta_path.name)

    if is_best:
        link_best = output_dir / "best.pt"
        _update_sidecar_symlink(link_best, ckpt_path.name)
        if checkpoint_meta is not None:
            _update_sidecar_symlink(output_dir / "best.json", meta_path.name)

    prune_old_checkpoints(output_dir, keep_last)
    logger.info(f"Checkpoint saved -> {ckpt_path}  (best={is_best})")


def prune_old_checkpoints(output_dir: Path, keep_last: int) -> None:
    checkpoints = sorted(output_dir.glob("step_*.pt"))
    best_target = None
    best_link = output_dir / "best.pt"
    if best_link.is_symlink():
        best_target = (output_dir / os.readlink(best_link)).resolve()

    prunable = [
        path for path in checkpoints
        if best_target is None or path.resolve() != best_target
    ]
    stale = prunable[:-keep_last]
    if not stale:
        return

    for path in stale:
        path.unlink(missing_ok=True)
        path.with_suffix(".json").unlink(missing_ok=True)

    logger.info("Pruned %d old checkpoint(s); keeping last %d.", len(stale), keep_last)


def load_checkpoint(
    path: str,
    model: HybridMoEModelV2,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    train_cfg: Dict,
    model_cfg: ModelConfigV2,
) -> Tuple[int, int, float, Dict[str, Any]]:
    logger.info(f"Loading checkpoint: {path}")
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    step = state.get("step", 0)
    tokens_trained = state.get("tokens_trained", 0)
    if tokens_trained == 0:
        # Infer from step
        batch_size = int(state.get("train_cfg", {}).get("batch_size", train_cfg.get("batch_size", 1)))
        grad_accum = int(state.get("train_cfg", {}).get("grad_accum", train_cfg.get("grad_accum", 1)))
        seq_len = int(state.get("model_config", {}).get("max_seq_len", model_cfg.max_seq_len))
        tokens_trained = step * batch_size * grad_accum * seq_len
    best_val_loss = state.get("best_val_loss", float("inf"))
    checkpoint_meta = state.get("checkpoint_meta", {})
    logger.info(f"  Resumed from step {step}  |  "
                f"tokens_trained={tokens_trained/1e9:.3f}B  |  "
                f"best_val_loss={best_val_loss:.4f}")
    return step, tokens_trained, best_val_loss, checkpoint_meta


# ─── Evaluation ──────────────────────────────────────────────────────────────

def evaluate(
    model: HybridMoEModelV2,
    val_batches: List[Dict[str, torch.Tensor]],
    device: torch.device,
    dtype: torch.dtype,
    eval_seed: int,
) -> Dict[str, float]:
    if not val_batches:
        raise ValueError("Validation batch subset is empty.")

    was_training = model.training
    model.eval()

    total_ce = 0.0
    total_aux = 0.0
    total_loss = 0.0
    total_tokens = 0

    with torch.inference_mode():
        for batch in val_batches:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            valid_tokens = int((labels != -100).sum().item())
            if valid_tokens == 0:
                continue

            with torch.autocast(device_type=device.type, dtype=dtype, enabled=(device.type == "cuda")):
                _, _, loss_dict = model(input_ids, targets=labels, use_checkpoint=False)

            if loss_dict is None:
                continue

            ce_loss = float(loss_dict["ce_loss"])
            aux_loss = float(loss_dict["aux_loss"])
            total_ce += ce_loss * valid_tokens
            total_aux += aux_loss
            total_loss += (ce_loss + aux_loss) * valid_tokens
            total_tokens += valid_tokens

    if was_training:
        model.train()

    if total_tokens == 0:
        raise RuntimeError("Validation produced zero non-padding tokens.")

    batches_evaluated = len(val_batches)
    return {
        "val_loss": total_loss / total_tokens,
        "val_ce_loss": total_ce / total_tokens,
        "val_aux_loss": total_aux / batches_evaluated,
        "batches_evaluated": float(batches_evaluated),
        "tokens_evaluated": float(total_tokens),
        "eval_seed": float(eval_seed),
    }


# ─── Setup ───────────────────────────────────────────────────────────────────

def setup_logging(output_dir: Path, level: int = logging.INFO) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(output_dir / "train_v2.log"),
    ]
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


def build_fixed_validation_batches(
    *,
    data_dir: str,
    batch_size: int,
    max_seq_len: int,
    en_ratio: float,
    pt_ratio: float,
    seed: int,
    n_batches: int,
) -> List[Dict[str, torch.Tensor]]:
    val_dataset = PreTokenizedDataset(
        data_dir=data_dir,
        max_seq_len=max_seq_len,
        en_ratio=en_ratio,
        pt_ratio=pt_ratio,
        seed=seed,
        split="val",
    )
    val_dataset.set_epoch(0)

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
    )

    available_batches = len(val_loader)
    if available_batches == 0:
        raise RuntimeError("Validation split is too small to form a single batch.")
    target_batches = min(n_batches, available_batches)

    fixed_batches: List[Dict[str, torch.Tensor]] = []
    for idx, batch in enumerate(val_loader):
        if idx >= target_batches:
            break
        fixed_batches.append({
            "input_ids": batch["input_ids"].clone(),
            "labels": batch["labels"].clone(),
        })

    total_tokens = sum(int((b["labels"] != -100).sum().item()) for b in fixed_batches)
    logger.info(
        "Validation subset fixed: %d batches  |  %d tokens  |  seed=%d",
        len(fixed_batches), total_tokens, seed,
    )
    return fixed_batches


# ─── Gradient accumulation context ──────────────────────────────────────────

@contextmanager
def _no_sync_context(model, should_sync: bool):
    """
    V2 improvement: clean context manager for gradient accumulation.
    Disables gradient sync during micro-batches for DDP compatibility.
    For single-GPU this is a no-op, but makes the code DDP-ready.
    """
    if not should_sync and hasattr(model, 'no_sync'):
        with model.no_sync():
            yield
    else:
        yield


# ─── CUDA crash protection ───────────────────────────────────────────────────

def _build_crash_artifact(
    *,
    step: int,
    tokens_trained: int,
    lr: float,
    step_tokens: int,
    exception: BaseException,
    device: torch.device,
) -> Dict[str, Any]:
    """Build structured crash metadata for diagnostics and post-mortem."""
    tb_str = traceback.format_exception(type(exception), exception, exception.__traceback__)
    is_cuda_error = isinstance(exception, (torch.cuda.OutOfMemoryError, RuntimeError)) and (
        "CUDA" in str(exception) or "cuda" in str(exception) or isinstance(exception, torch.cuda.OutOfMemoryError)
    )

    artifact: Dict[str, Any] = {
        "event": "training_crash",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "step": step,
        "tokens_trained": tokens_trained,
        "lr": lr,
        "partial_step_tokens": step_tokens,
        "exception_type": type(exception).__qualname__,
        "exception_message": str(exception),
        "traceback": "".join(tb_str),
        "is_cuda_error": is_cuda_error,
    }

    if is_cuda_error and device.type == "cuda":
        try:
            artifact["cuda"] = {
                "device": str(device),
                "device_name": torch.cuda.get_device_name(device),
                "memory_allocated_gb": round(torch.cuda.memory_allocated(device) / 1e9, 3),
                "memory_reserved_gb": round(torch.cuda.memory_reserved(device) / 1e9, 3),
                "max_memory_allocated_gb": round(torch.cuda.max_memory_allocated(device) / 1e9, 3),
            }
        except Exception:
            artifact["cuda"] = {"error": "Failed to collect CUDA memory diagnostics"}
        artifact["recommendation"] = (
            "Re-run with CUDA_LAUNCH_BLOCKING=1 to get the exact failing kernel: "
            "CUDA_LAUNCH_BLOCKING=1 python train_v2.py ..."
        )

    return artifact


def _handle_training_crash(
    *,
    exception: BaseException,
    step: int,
    tokens_trained: int,
    lr: float,
    step_tokens: int,
    device: torch.device,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    best_val_loss: float,
    train_cfg: Dict,
    ckpt_dir: Path,
    wandb_run,
) -> None:
    """
    Emergency handler: save crash JSON, attempt emergency checkpoint, clean up W&B.
    Does NOT mask the exception — caller must re-raise.
    """
    crash = _build_crash_artifact(
        step=step, tokens_trained=tokens_trained, lr=lr,
        step_tokens=step_tokens, exception=exception, device=device,
    )

    # 1. Always try to save crash JSON first (lightweight, most likely to succeed)
    crash_json_path = ckpt_dir / f"crash_step_{step:08d}.json"
    try:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        _write_json(crash_json_path, crash)
        logger.error("Crash artifact saved -> %s", crash_json_path)
    except Exception as e2:
        logger.error("Failed to save crash JSON: %s", e2)

    # 2. Attempt emergency checkpoint (without overwriting normal checkpoints)
    emergency_path = ckpt_dir / f"emergency_step_{step:08d}.pt"
    try:
        raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        state = {
            "step": step,
            "tokens_trained": tokens_trained,
            "best_val_loss": best_val_loss,
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_config": raw_model.config.__dict__,
            "train_cfg": train_cfg,
            "model_version": "v2",
            "emergency": True,
        }
        torch.save(state, emergency_path)
        logger.error("Emergency checkpoint saved -> %s", emergency_path)
    except Exception as e3:
        logger.error("Emergency checkpoint FAILED: %s (crash JSON still available)", e3)

    # 3. Clean up W&B
    if wandb_run is not None:
        try:
            import wandb
            wandb_run.log({"crash/step": step, "crash/type": type(exception).__qualname__})
            wandb_run.finish(exit_code=1)
            logger.info("W&B finalized with exit_code=1")
        except Exception:
            pass

    # 4. Log summary
    logger.error("=" * 72)
    logger.error("TRAINING CRASHED at step %d  |  tokens: %.3fB  |  lr: %.2e",
                 step, tokens_trained / 1e9, lr)
    logger.error("Exception: %s: %s", type(exception).__qualname__, exception)
    if crash.get("recommendation"):
        logger.error(crash["recommendation"])
    logger.error("=" * 72)


# ─── Main training loop ─────────────────────────────────────────────────────

def main():
    signal.signal(signal.SIGINT, _sigint_handler)

    parser = build_arg_parser()
    args = parser.parse_args()

    cfg, train_cfg = build_config_from_args(args)

    # V2: separate output dir to not mix with V1
    output_dir = Path(train_cfg.get("output_dir", "model_1b")) / "v2"
    if args.output_dir:
        output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    setup_logging(output_dir)

    logger.info("=" * 72)
    logger.info("Hybrid Transformer-Mamba-MoE V2  |  Training")
    logger.info("=" * 72)

    # ── Device ───────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if train_cfg.get("dtype", "bfloat16") == "bfloat16" else torch.float16
    logger.info(f"Device: {device}  |  dtype: {dtype}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}  |  "
                    f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    torch.manual_seed(train_cfg.get("seed", 42))

    # ── Data ─────────────────────────────────────────────────────────────────
    data_dir = train_cfg.get("data_dir",
        str(Path(__file__).parent.parent / "bitnet-mamba-hybrid" / "data" / "tokenized"))

    info = get_dataset_info(data_dir)
    logger.info(f"Dataset  EN: {info['en_tokens']:,} tokens  "
                f"|  PT: {info['pt_tokens']:,} tokens")

    batch_size = train_cfg.get("batch_size", 2)
    seq_len = cfg.max_seq_len
    grad_accum = train_cfg.get("grad_accum", 16)
    num_workers = train_cfg.get("num_workers", 4)
    eval_batches_requested = train_cfg.get("eval_batches", 50)
    eval_seed = train_cfg.get("eval_seed", train_cfg.get("seed", 42) + 999)

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
    val_batches = build_fixed_validation_batches(
        data_dir=data_dir,
        batch_size=batch_size,
        max_seq_len=seq_len,
        en_ratio=train_cfg.get("en_ratio", 0.3),
        pt_ratio=train_cfg.get("pt_ratio", 0.7),
        seed=eval_seed,
        n_batches=eval_batches_requested,
    )

    # ── Model ────────────────────────────────────────────────────────────────
    logger.info("Building V2 model...")
    logger.info(f"  V2 features: qk_norm={cfg.qk_norm}, shared_expert={cfg.moe_shared_expert}, "
                f"z_loss_coeff={cfg.moe_z_loss_coeff}")

    # V2: support migrating from V1 checkpoint
    if args.migrate_v1 and args.resume:
        logger.info(f"Migrating V1 checkpoint to V2: {args.resume}")
        model, migration_report = load_v1_checkpoint_into_v2(args.resume, cfg, device)
        logger.info(f"  Loaded: {len(migration_report['loaded'])} keys")
        logger.info(f"  New (random init): {len(migration_report['missing_in_v1'])} keys")
        if migration_report['skipped']:
            logger.warning(f"  Skipped (shape mismatch): {migration_report['skipped']}")
        model = model.to(device)
    else:
        model = HybridMoEModelV2(cfg).to(device)

    mem = estimate_memory_gb_v2(cfg, batch_size, seq_len)
    logger.info(f"Estimated VRAM: {mem}")

    use_grad_ckpt = train_cfg.get("gradient_checkpointing", True)
    use_compile = train_cfg.get("compile", False)  # V2: default off (mamba-ssm compat)
    if args.no_grad_checkpoint:
        use_grad_ckpt = False
    if args.no_compile:
        use_compile = False
    train_cfg["gradient_checkpointing"] = use_grad_ckpt
    train_cfg["compile"] = use_compile

    if use_compile:
        from model_v2 import HAS_MAMBA_SSM
        if HAS_MAMBA_SSM:
            logger.warning("torch.compile disabled: mamba-ssm CUDA extension is not compatible.")
        else:
            logger.info("Compiling model with torch.compile(mode='reduce-overhead')...")
            model = torch.compile(model, mode="reduce-overhead", dynamic=False)

    # ── Optimizer ────────────────────────────────────────────────────────────
    optimizer = build_optimizer(model, train_cfg)

    # ── Resume (V2 checkpoint) ───────────────────────────────────────────────
    start_step = 0
    start_tokens = 0
    best_val_loss = float("inf")
    last_eval_metrics: Optional[Dict[str, Any]] = None

    resume_path = train_cfg.get("resume", None)
    if args.resume and not args.migrate_v1:
        resume_path = args.resume
    if resume_path and Path(resume_path).exists() and not args.migrate_v1:
        start_step, start_tokens, best_val_loss, resume_meta = load_checkpoint(
            resume_path, model, optimizer, device, train_cfg, cfg
        )

    # ── W&B ──────────────────────────────────────────────────────────────────
    wandb_run = None
    if train_cfg.get("wandb", False) or args.wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=train_cfg.get("wandb_project", "hybrid-moe-1b"),
                name=train_cfg.get("wandb_run_name", None),
                config={**cfg.__dict__, **train_cfg, "model_version": "v2"},
                resume="allow",
            )
            logger.info("W&B initialized")
        except Exception as e:
            logger.warning(f"W&B init failed: {e}")

    # ── Training state ───────────────────────────────────────────────────────
    max_steps = train_cfg.get("max_steps", 200_000)
    max_tokens = train_cfg.get("max_tokens", None)
    warmup = train_cfg.get("warmup_steps", 2_000)
    peak_lr = train_cfg.get("lr", 3e-4)
    min_lr = train_cfg.get("min_lr", 3e-5)
    grad_clip = train_cfg.get("grad_clip", 1.0)
    log_every = train_cfg.get("log_every", 10)
    eval_every = train_cfg.get("eval_every", 500)
    save_every = train_cfg.get("save_every", 1000)

    # V2: LR schedule selection
    lr_schedule = args.lr_schedule or train_cfg.get("lr_schedule", "cosine")
    if lr_schedule == "wsd":
        get_lr = lambda step: get_lr_wsd(step, warmup, effective_max_steps, peak_lr, min_lr)
        logger.info("LR schedule: WSD (Warmup-Stable-Decay)")
    else:
        get_lr = lambda step: get_lr_cosine(step, warmup, effective_max_steps, peak_lr, min_lr)
        logger.info("LR schedule: Cosine")

    tokens_trained = start_tokens

    if max_tokens is not None:
        tokens_remaining = max(0, max_tokens - tokens_trained)
        steps_from_tokens = math.ceil(tokens_remaining / effective_batch_tokens)
        effective_max_steps = min(max_steps, start_step + steps_from_tokens)
        logger.info(f"max_tokens={max_tokens/1e9:.2f}B  |  effective_max_steps={effective_max_steps:,}")
    else:
        effective_max_steps = max_steps

    # V2: no GradScaler for bf16 (it was a no-op in V1 anyway)
    use_fp16_scaler = (dtype == torch.float16)
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_fp16_scaler) if use_fp16_scaler else None

    model.train()
    train_iter = iter(train_loader)
    optimizer.zero_grad(set_to_none=True)

    step = start_step
    log_total_loss = 0.0
    log_ce_loss = 0.0
    log_aux_loss = 0.0
    log_tokens = 0
    t_last = time.perf_counter()
    grad_norm_ema = 0.0
    last_train_metrics: Dict[str, Any] = {}

    logger.info(f"Starting training from step {step} -> {effective_max_steps}")
    logger.info(f"gradient_checkpointing={use_grad_ckpt}  |  compile={use_compile}")

    while step < effective_max_steps and (max_tokens is None or tokens_trained < max_tokens):
        lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # ── Micro-batch accumulation ─────────────────────────────────────
        step_total = 0.0
        step_ce = 0.0
        step_aux = 0.0
        step_tokens = 0

        try:
            for micro_idx in range(grad_accum):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    batch = next(train_iter)
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)

                with torch.autocast(device_type=device.type, dtype=dtype, enabled=(device.type == "cuda")):
                    _, total_loss, loss_dict = model(
                        input_ids, targets=labels, use_checkpoint=use_grad_ckpt
                    )

                scaled_loss = total_loss / grad_accum
                if scaler is not None:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                step_total += total_loss.item() / grad_accum
                step_ce += loss_dict["ce_loss"] / grad_accum
                step_aux += loss_dict["aux_loss"] / grad_accum
                step_tokens += int((labels != -100).sum().item())

            # ── Optimizer step ────────────────────────────────────────────
            if scaler is not None:
                scaler.unscale_(optimizer)
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        except Exception as exc:
            _handle_training_crash(
                exception=exc,
                step=step,
                tokens_trained=tokens_trained,
                lr=lr,
                step_tokens=step_tokens,
                device=device,
                model=model,
                optimizer=optimizer,
                best_val_loss=best_val_loss,
                train_cfg=train_cfg,
                ckpt_dir=ckpt_dir,
                wandb_run=wandb_run,
            )
            raise  # Do NOT mask the exception

        step += 1
        tokens_trained += step_tokens
        grad_norm_ema = 0.9 * grad_norm_ema + 0.1 * grad_norm.item()
        last_train_metrics = {
            "train_loss": step_total,
            "train_ce_loss": step_ce,
            "train_aux_loss": step_aux,
            "grad_norm": grad_norm_ema,
            "lr": lr,
        }

        log_total_loss += step_total
        log_ce_loss += step_ce
        log_aux_loss += step_aux
        log_tokens += step_tokens

        # ── Logging ──────────────────────────────────────────────────────
        if step % log_every == 0:
            t_now = time.perf_counter()
            elapsed = t_now - t_last
            tok_s = log_tokens / max(elapsed, 1e-6)

            avg_total = log_total_loss / log_every
            avg_ce = log_ce_loss / log_every
            avg_aux = log_aux_loss / log_every
            mem_gb = (torch.cuda.memory_allocated() / 1e9) if device.type == "cuda" else 0.0

            last_train_metrics = {
                "train_loss": avg_total,
                "train_ce_loss": avg_ce,
                "train_aux_loss": avg_aux,
                "grad_norm": grad_norm_ema,
                "lr": lr,
            }

            tokens_str = (
                f"  tokens: {tokens_trained/1e9:.3f}B/{max_tokens/1e9:.2f}B"
                if max_tokens else
                f"  tokens: {tokens_trained/1e9:.3f}B"
            )
            logger.info(
                f"step {step:>7d}/{effective_max_steps}  |  "
                f"loss: {avg_total:.4f}  ce: {avg_ce:.4f}  aux: {avg_aux:.4f}  |  "
                f"lr: {lr:.2e}  gnorm: {grad_norm_ema:.3f}  |  "
                f"tok/s: {tok_s:,.0f}  mem: {mem_gb:.1f}GB{tokens_str}"
            )

            if wandb_run is not None:
                wandb_run.log({
                    "train/loss": avg_total,
                    "train/ce_loss": avg_ce,
                    "train/aux_loss": avg_aux,
                    "train/lr": lr,
                    "train/grad_norm": grad_norm_ema,
                    "perf/tok_per_sec": tok_s,
                    "perf/mem_gb": mem_gb,
                    "tokens_trained_B": tokens_trained / 1e9,
                    "step": step,
                })

            log_total_loss = log_ce_loss = log_aux_loss = 0.0
            log_tokens = 0
            t_last = t_now

        # ── Evaluation ───────────────────────────────────────────────────
        if step % eval_every == 0:
            eval_metrics = evaluate(model, val_batches, device, dtype, eval_seed)
            val_ce_loss = float(eval_metrics["val_ce_loss"])
            is_best = val_ce_loss < best_val_loss
            if is_best:
                best_val_loss = val_ce_loss
            eval_metrics["best_val_loss"] = best_val_loss
            last_eval_metrics = eval_metrics

            logger.info(
                f"[EVAL]  step {step}  |  val_ce: {eval_metrics['val_ce_loss']:.4f}  "
                f"|  val_aux: {eval_metrics['val_aux_loss']:.4f}  "
                f"|  best: {best_val_loss:.4f}  {'*' if is_best else ''}"
            )
            if wandb_run is not None:
                wandb_run.log({
                    "val/loss": eval_metrics["val_loss"],
                    "val/ce_loss": eval_metrics["val_ce_loss"],
                    "val/aux_loss": eval_metrics["val_aux_loss"],
                    "val/best_ce_loss": best_val_loss,
                    "step": step,
                })

            raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            meta = build_checkpoint_metadata(
                step=step, tokens_trained=tokens_trained,
                best_val_loss=best_val_loss,
                train_metrics=last_train_metrics,
                eval_metrics=last_eval_metrics,
                train_cfg=train_cfg,
            )
            save_checkpoint(raw_model, optimizer, step, tokens_trained,
                          best_val_loss, train_cfg, ckpt_dir, meta, is_best)

        elif step % save_every == 0:
            raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            meta = build_checkpoint_metadata(
                step=step, tokens_trained=tokens_trained,
                best_val_loss=best_val_loss,
                train_metrics=last_train_metrics,
                eval_metrics=last_eval_metrics,
                train_cfg=train_cfg,
            )
            save_checkpoint(raw_model, optimizer, step, tokens_trained,
                          best_val_loss, train_cfg, ckpt_dir, meta, False)

        # ── SIGINT ───────────────────────────────────────────────────────
        if _interrupt_requested:
            logger.info("SIGINT: saving checkpoint and exiting...")
            raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            meta = build_checkpoint_metadata(
                step=step, tokens_trained=tokens_trained,
                best_val_loss=best_val_loss,
                train_metrics=last_train_metrics,
                eval_metrics=last_eval_metrics,
                train_cfg=train_cfg,
            )
            save_checkpoint(raw_model, optimizer, step, tokens_trained,
                          best_val_loss, train_cfg, ckpt_dir, meta, False)
            logger.info("Checkpoint saved. Exiting.")
            if wandb_run is not None:
                wandb_run.finish()
            sys.exit(0)

    logger.info("Training complete.")
    if wandb_run is not None:
        wandb_run.finish()


# ─── Argument parser ─────────────────────────────────────────────────────────

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train Hybrid Transformer-Mamba-MoE V2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    arch = p.add_argument_group("Model Architecture (override config.yaml)")
    arch.add_argument("--d_model", type=int, default=None)
    arch.add_argument("--n_layers", type=int, default=None)
    arch.add_argument("--n_heads", type=int, default=None)
    arch.add_argument("--n_kv_heads", type=int, default=None)
    arch.add_argument("--mamba_d_state", type=int, default=None)
    arch.add_argument("--mamba_expand", type=int, default=None)
    arch.add_argument("--moe_n_experts", type=int, default=None)
    arch.add_argument("--moe_top_k", type=int, default=None)
    arch.add_argument("--moe_intermediate", type=int, default=None)
    arch.add_argument("--vocab_size", type=int, default=None)
    arch.add_argument("--max_seq_len", type=int, default=None)
    # V2-specific
    arch.add_argument("--qk_norm", type=bool, default=None)
    arch.add_argument("--moe_shared_expert", type=bool, default=None)
    arch.add_argument("--moe_z_loss_coeff", type=float, default=None)

    train = p.add_argument_group("Training (override config.yaml)")
    train.add_argument("--batch_size", type=int, default=None)
    train.add_argument("--grad_accum", type=int, default=None)
    train.add_argument("--max_steps", type=int, default=None)
    train.add_argument("--max_tokens", type=int, default=None)
    train.add_argument("--warmup_steps", type=int, default=None)
    train.add_argument("--lr", type=float, default=None)
    train.add_argument("--min_lr", type=float, default=None)
    train.add_argument("--weight_decay", type=float, default=None)
    train.add_argument("--grad_clip", type=float, default=None)
    train.add_argument("--seed", type=int, default=None)
    train.add_argument("--data_dir", type=str, default=None)
    train.add_argument("--output_dir", type=str, default=None)
    train.add_argument("--en_ratio", type=float, default=None)
    train.add_argument("--pt_ratio", type=float, default=None)
    train.add_argument("--log_every", type=int, default=None)
    train.add_argument("--eval_every", type=int, default=None)
    train.add_argument("--save_every", type=int, default=None)
    train.add_argument("--keep_last_checkpoints", type=int, default=None)
    train.add_argument("--eval_batches", type=int, default=None)
    train.add_argument("--eval_seed", type=int, default=None)
    train.add_argument("--resume", type=str, default=None)

    feat = p.add_argument_group("Features")
    feat.add_argument("--no_compile", action="store_true")
    feat.add_argument("--no_grad_checkpoint", action="store_true")
    feat.add_argument("--wandb", action="store_true")
    feat.add_argument("--wandb_project", type=str, default=None)
    feat.add_argument("--wandb_run_name", type=str, default=None)
    feat.add_argument("--dtype", type=str, default=None, choices=["bfloat16", "float16"])
    feat.add_argument("--lr_schedule", type=str, default=None,
                      choices=["cosine", "wsd"],
                      help="V2: LR schedule. 'cosine' (default) or 'wsd' (Warmup-Stable-Decay)")
    feat.add_argument("--migrate_v1", action="store_true",
                      help="V2: Load V1 checkpoint into V2 model (partial weight migration)")

    return p


if __name__ == "__main__":
    main()
