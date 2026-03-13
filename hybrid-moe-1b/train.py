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
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, SequentialSampler

# Add parent dir so we can import the existing data_loader
sys.path.insert(0, str(Path(__file__).parent.parent / "bitnet-mamba-hybrid"))
from data_loader import PreTokenizedDataset, create_dataloader, get_dataset_info  # type: ignore

from model import HybridMoEModel, ModelConfig, estimate_memory_gb

logger = logging.getLogger(__name__)


def _to_python_value(value: Any) -> Any:
    """Convert tensors/scalars to JSON-safe Python values."""
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
    """Build a structured metadata payload persisted with each checkpoint."""
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


def log_cuda_debug_context(device: torch.device) -> None:
    """Log best-effort CUDA diagnostics after an accelerator failure."""
    if device.type != "cuda":
        return
    try:
        mem_alloc = torch.cuda.memory_allocated(device) / 1e9
        mem_reserved = torch.cuda.memory_reserved(device) / 1e9
        max_mem = torch.cuda.max_memory_allocated(device) / 1e9
        logger.error(
            "CUDA debug  |  alloc=%.2fGB  reserved=%.2fGB  peak=%.2fGB  device=%s",
            mem_alloc,
            mem_reserved,
            max_mem,
            device,
        )
    except Exception as debug_exc:
        logger.error("Failed to collect CUDA debug info: %s", debug_exc)


# ─── SIGINT handler (Ctrl+C → safe checkpoint) ───────────────────────────────

_interrupt_requested: bool = False

def _sigint_handler(signum, frame) -> None:
    """First Ctrl+C: request graceful exit after current step.
       Second Ctrl+C: force-exit immediately (default Python behaviour restored).
    """
    global _interrupt_requested
    if _interrupt_requested:
        # User is impatient — restore default and let it propagate
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        raise KeyboardInterrupt
    _interrupt_requested = True
    # Print directly; logger may not be initialised yet at import time
    print("\nSIGINT received — will save checkpoint after current step "
          "(press Ctrl+C again to force-quit).", flush=True)


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


# ─── Checkpointing ────────────────────────────────────────────────────────────

def save_checkpoint(
    model: HybridMoEModel,
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
        "step":           step,
        "tokens_trained": tokens_trained,   # cumulative tokens seen so far
        "best_val_loss":  best_val_loss,
        "model":          model.state_dict(),
        "optimizer":      optimizer.state_dict(),
        "model_config":   model.config.__dict__,
        "train_cfg":      train_cfg,
        "checkpoint_meta": checkpoint_meta or {},
    }
    ckpt_path = output_dir / f"step_{step:08d}.pt"
    torch.save(state, ckpt_path)
    meta_path = ckpt_path.with_suffix(".json")
    if checkpoint_meta is not None:
        _write_json(meta_path, checkpoint_meta)

    # Symlink latest / best
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
    logger.info(f"Checkpoint saved → {ckpt_path}  (best={is_best})")


def prune_old_checkpoints(output_dir: Path, keep_last: int) -> None:
    """Keep the most recent numbered checkpoints and preserve the best one."""
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
    model: HybridMoEModel,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    train_cfg: Dict,
    model_cfg: ModelConfig,
) -> Tuple[int, int, float, Dict[str, Any]]:
    """Load checkpoint; returns step, tokens, best loss, and checkpoint metadata."""
    logger.info(f"Loading checkpoint: {path}")
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    step           = state.get("step", 0)
    tokens_trained = state.get("tokens_trained")
    if tokens_trained is None:
        tokens_trained = infer_tokens_trained(state, step, train_cfg, model_cfg)
    best_val_loss  = state.get("best_val_loss", float("inf"))
    checkpoint_meta = state.get("checkpoint_meta") or load_checkpoint_metadata(path)
    logger.info(f"  Resumed from step {step}  |  "
                f"tokens_trained={tokens_trained/1e9:.3f}B  |  "
                f"best_val_loss={best_val_loss:.4f}")
    if checkpoint_meta:
        logger.info(
            "  Resume metadata: train_ce=%s  |  val_ce=%s  |  run=%s",
            checkpoint_meta.get("train_ce_loss"),
            checkpoint_meta.get("val_ce_loss"),
            checkpoint_meta.get("run_name"),
        )
    return step, tokens_trained, best_val_loss, checkpoint_meta


def load_checkpoint_metadata(path: str) -> Dict[str, Any]:
    sidecar_path = Path(path).with_suffix(".json")
    if not sidecar_path.exists():
        return {}
    with sidecar_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def infer_tokens_trained(state: Dict, step: int, train_cfg: Dict, model_cfg: ModelConfig) -> int:
    """
    Best-effort fallback for legacy checkpoints that did not persist token counts.
    """
    saved_train_cfg = state.get("train_cfg", {})
    saved_model_cfg = state.get("model_config", {})

    batch_size = int(saved_train_cfg.get("batch_size", train_cfg.get("batch_size", 1)))
    grad_accum = int(saved_train_cfg.get("grad_accum", train_cfg.get("grad_accum", 1)))
    seq_len = int(saved_model_cfg.get("max_seq_len", model_cfg.max_seq_len))

    inferred = step * batch_size * grad_accum * seq_len
    logger.warning(
        "Checkpoint does not contain tokens_trained; inferred %s tokens from "
        "step=%s, batch_size=%s, grad_accum=%s, max_seq_len=%s.",
        f"{inferred:,}",
        step,
        batch_size,
        grad_accum,
        seq_len,
    )
    return inferred


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(
    model: HybridMoEModel,
    val_batches: List[Dict[str, torch.Tensor]],
    device: torch.device,
    dtype: torch.dtype,
    eval_seed: int,
) -> Dict[str, float]:
    """Compute deterministic validation metrics over a fixed batch subset.

    CE is aggregated by valid-token count so padding/truncation differences do
    not distort the metric. Aux loss is tracked only for audit.
    """
    if not val_batches:
        raise ValueError("Validation batch subset is empty.")

    was_training = model.training
    model.eval()
    assert not model.training, "Validation must run with model.eval()."

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
    """Materialize a deterministic validation subset once and reuse it."""
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
    assert isinstance(val_loader.sampler, SequentialSampler), (
        "Validation DataLoader must use sequential sampling."
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

    assert fixed_batches, "Failed to materialize deterministic validation batches."
    first_hash = int(fixed_batches[0]["input_ids"][0, :16].sum().item())
    total_tokens = sum(int((batch["labels"] != -100).sum().item()) for batch in fixed_batches)
    logger.info(
        "Validation subset fixed: %d batches  |  %d tokens  |  seed=%d  |  first_batch_hash=%d",
        len(fixed_batches),
        total_tokens,
        seed,
        first_hash,
    )
    if target_batches < n_batches:
        logger.warning(
            "Requested %d eval batches but validation split provides only %d; clamping.",
            n_batches,
            target_batches,
        )

    return fixed_batches


# ─── Main training loop ───────────────────────────────────────────────────────

def main():
    # Register graceful-exit handler immediately so it covers the whole run
    signal.signal(signal.SIGINT, _sigint_handler)

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

    # ── Optional data sanity check ────────────────────────────────────────────
    # Verifies that input_ids/labels are correctly shifted (labels = tokens[1:])
    if train_cfg.get("debug_data_check", False) or args.debug_data_check:
        sample = next(iter(train_loader))
        ids_s  = sample["input_ids"]
        lbl_s  = sample["labels"]
        val_sample = val_batches[0]
        logger.info("[debug] Data check:")
        logger.info(f"  input_ids shape : {list(ids_s.shape)}  dtype={ids_s.dtype}")
        logger.info(f"  labels shape    : {list(lbl_s.shape)}  dtype={lbl_s.dtype}")
        logger.info(f"  input_ids[0,:8] : {ids_s[0, :8].tolist()}")
        logger.info(f"  labels[0,:8]    : {lbl_s[0, :8].tolist()}")
        logger.info(f"  val input shape : {list(val_sample['input_ids'].shape)}")
        logger.info(f"  val labels shape: {list(val_sample['labels'].shape)}")
        # input_ids[t+1] should equal labels[t] for valid positions
        valid_mask = lbl_s[0] != -100
        if valid_mask.sum() > 1:
            shift_ok = (ids_s[0, 1:][valid_mask[1:]] == lbl_s[0, :-1][valid_mask[1:]]).all()
            logger.info(f"  label shift OK  : {bool(shift_ok)}")
        else:
            logger.warning("  label shift check skipped (too many -100 labels)")
        assert val_sample["input_ids"].shape == ids_s.shape, "Train/val input shape mismatch."
        assert val_sample["labels"].shape == lbl_s.shape, "Train/val label shape mismatch."

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
    if args.no_grad_checkpoint:
        use_grad_ckpt = False
    if args.no_compile:
        use_compile = False
    train_cfg["gradient_checkpointing"] = use_grad_ckpt
    train_cfg["compile"] = use_compile

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
    start_tokens  = 0
    best_val_loss = float("inf")
    last_eval_metrics: Optional[Dict[str, Any]] = None
    resume_path   = train_cfg.get("resume", None)
    if args.resume:
        resume_path = args.resume
    if resume_path and Path(resume_path).exists():
        start_step, start_tokens, best_val_loss, resume_meta = load_checkpoint(
            resume_path, model, optimizer, device, train_cfg, cfg  # type: ignore[arg-type]
        )
        if resume_meta:
            last_eval_metrics = {
                "val_loss": resume_meta.get("val_loss"),
                "val_ce_loss": resume_meta.get("val_ce_loss"),
                "val_aux_loss": resume_meta.get("val_aux_loss"),
                "batches_evaluated": resume_meta.get("eval_batches"),
                "tokens_evaluated": resume_meta.get("eval_tokens"),
                "eval_seed": resume_meta.get("eval_seed", eval_seed),
            }
            if not train_cfg.get("wandb_run_name") and resume_meta.get("run_name"):
                train_cfg["wandb_run_name"] = resume_meta["run_name"]

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
    max_tokens  = train_cfg.get("max_tokens", None)   # optional token-budget limit
    warmup      = train_cfg.get("warmup_steps", 2_000)
    peak_lr     = train_cfg.get("lr", 3e-4)
    min_lr      = train_cfg.get("min_lr", 3e-5)
    grad_clip   = train_cfg.get("grad_clip", 1.0)
    log_every   = train_cfg.get("log_every", 10)
    eval_every  = train_cfg.get("eval_every", 500)
    save_every  = train_cfg.get("save_every", 1000)

    tokens_trained = start_tokens

    # When a token budget is set, derive the equivalent max_steps so the
    # cosine LR schedule is calibrated to the full planned training run.
    # effective_batch_tokens is already computed above.
    if max_tokens is not None:
        tokens_remaining = max(0, max_tokens - tokens_trained)
        steps_from_tokens = math.ceil(tokens_remaining / effective_batch_tokens)
        # Use whichever limit is tighter; also use max_tokens-derived steps
        # to calibrate the LR schedule horizon.
        effective_max_steps = min(max_steps, start_step + steps_from_tokens)
        logger.info(
            f"max_tokens={max_tokens/1e9:.2f}B  |  already seen={tokens_trained/1e9:.3f}B  |  "
            f"steps remaining≈{steps_from_tokens:,}  |  effective_max_steps={effective_max_steps:,}"
        )
    else:
        effective_max_steps = max_steps
        logger.info(f"max_tokens not set — training for {max_steps:,} steps")

    scaler = torch.amp.GradScaler(device="cuda", enabled=(dtype == torch.float16))

    model.train()
    train_iter = iter(train_loader)
    optimizer.zero_grad(set_to_none=True)

    step             = start_step
    # Per-log-window accumulators (reset every log_every steps)
    log_total_loss   = 0.0   # sum of per-step average total losses
    log_ce_loss      = 0.0   # sum of per-step average CE losses
    log_aux_loss     = 0.0   # sum of per-step average aux losses
    log_tokens       = 0     # valid tokens seen in this window
    t_last           = time.perf_counter()
    grad_norm_ema    = 0.0   # EMA of grad norm for monitoring
    last_train_metrics: Dict[str, Any] = {
        "train_loss": None,
        "train_ce_loss": None,
        "train_aux_loss": None,
        "grad_norm": None,
        "lr": None,
    }

    logger.info(f"Starting training from step {step} → {effective_max_steps}"
                + (f"  ({max_tokens/1e9:.2f}B token budget)" if max_tokens else ""))
    logger.info(f"gradient_checkpointing={use_grad_ckpt}  |  compile={use_compile}")

    while step < effective_max_steps and (max_tokens is None or tokens_trained < max_tokens):
        step_total = 0.0
        step_ce = 0.0
        step_aux = 0.0
        step_tokens = 0
        lr = 0.0
        try:
            # ── Update LR ─────────────────────────────────────────────────────
            lr = get_lr(step, warmup, effective_max_steps, peak_lr, min_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # ── Micro-batch accumulation loop ─────────────────────────────────
            # Each micro-batch produces a mean-over-tokens CE loss. We divide by
            # grad_accum so the gradient is the average over the effective batch.
            for _ in range(grad_accum):
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

                scaler.scale(total_loss / grad_accum).backward()

                step_total += total_loss.item() / grad_accum
                step_ce += loss_dict["ce_loss"] / grad_accum  # type: ignore[index]
                step_aux += loss_dict["aux_loss"] / grad_accum  # type: ignore[index]
                step_tokens += int((labels != -100).sum().item())

            # ── Optimizer step ────────────────────────────────────────────────
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

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

            # Accumulate into per-window totals
            log_total_loss += step_total
            log_ce_loss += step_ce
            log_aux_loss += step_aux
            log_tokens += step_tokens

            # ── Logging ───────────────────────────────────────────────────────
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

            # ── Evaluation ────────────────────────────────────────────────────
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
                    f"|  eval_tokens: {int(eval_metrics['tokens_evaluated'])}  "
                    f"|  best: {best_val_loss:.4f}  {'★' if is_best else ''}"
                )
                if wandb_run is not None:
                    wandb_run.log({
                        "val/loss": eval_metrics["val_loss"],
                        "val/ce_loss": eval_metrics["val_ce_loss"],
                        "val/aux_loss": eval_metrics["val_aux_loss"],
                        "val/best_ce_loss": best_val_loss,
                        "val/tokens": eval_metrics["tokens_evaluated"],
                        "step": step,
                    })

                raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
                checkpoint_meta = build_checkpoint_metadata(
                    step=step,
                    tokens_trained=tokens_trained,
                    best_val_loss=best_val_loss,
                    train_metrics=last_train_metrics,
                    eval_metrics=last_eval_metrics,
                    train_cfg=train_cfg,
                )
                save_checkpoint(
                    raw_model,
                    optimizer,
                    step,
                    tokens_trained,
                    best_val_loss,
                    train_cfg,
                    ckpt_dir,
                    checkpoint_meta,
                    is_best,
                )

            elif step % save_every == 0:
                raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
                checkpoint_meta = build_checkpoint_metadata(
                    step=step,
                    tokens_trained=tokens_trained,
                    best_val_loss=best_val_loss,
                    train_metrics=last_train_metrics,
                    eval_metrics=last_eval_metrics,
                    train_cfg=train_cfg,
                )
                save_checkpoint(
                    raw_model,
                    optimizer,
                    step,
                    tokens_trained,
                    best_val_loss,
                    train_cfg,
                    ckpt_dir,
                    checkpoint_meta,
                    False,
                )

            if _interrupt_requested:
                logger.info("SIGINT: saving checkpoint and exiting…")
                raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
                checkpoint_meta = build_checkpoint_metadata(
                    step=step,
                    tokens_trained=tokens_trained,
                    best_val_loss=best_val_loss,
                    train_metrics=last_train_metrics,
                    eval_metrics=last_eval_metrics,
                    train_cfg=train_cfg,
                )
                save_checkpoint(
                    raw_model,
                    optimizer,
                    step,
                    tokens_trained,
                    best_val_loss,
                    train_cfg,
                    ckpt_dir,
                    checkpoint_meta,
                    False,
                )
                logger.info("Checkpoint saved. Exiting.")
                if wandb_run is not None:
                    wandb_run.finish()
                sys.exit(0)
        except Exception as exc:
            logger.exception(
                "Training step failed  |  step=%s  tokens=%.3fB  lr=%.2e  partial_tokens=%s",
                step,
                tokens_trained / 1e9,
                lr,
                step_tokens,
            )
            if "CUDA" in str(exc) or isinstance(exc, torch.AcceleratorError):
                log_cuda_debug_context(device)
                logger.error(
                    "CUDA failure detected. Re-run with CUDA_LAUNCH_BLOCKING=1 to pinpoint the failing kernel."
                )

            checkpoint_meta = build_checkpoint_metadata(
                step=step,
                tokens_trained=tokens_trained,
                best_val_loss=best_val_loss,
                train_metrics=last_train_metrics,
                eval_metrics=last_eval_metrics,
                train_cfg=train_cfg,
            )
            checkpoint_meta["crash"] = {
                "error_type": type(exc).__name__,
                "message": str(exc),
                "partial_step_tokens": step_tokens,
                "partial_train_loss": step_total if step_total else None,
                "partial_train_ce_loss": step_ce if step_ce else None,
                "partial_train_aux_loss": step_aux if step_aux else None,
            }
            crash_step = step if step_tokens == 0 else step + 1
            crash_path = ckpt_dir / f"crash_step_{crash_step:08d}.json"
            _write_json(crash_path, checkpoint_meta)

            try:
                raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
                save_checkpoint(
                    raw_model,
                    optimizer,
                    crash_step,
                    tokens_trained,
                    best_val_loss,
                    train_cfg,
                    ckpt_dir,
                    checkpoint_meta,
                    False,
                )
                logger.error("Emergency checkpoint saved after failure.")
            except Exception as save_exc:
                logger.exception("Emergency checkpoint save failed: %s", save_exc)
                logger.error("Crash metadata still written to %s", crash_path)

            if wandb_run is not None:
                wandb_run.finish(exit_code=1)
            raise

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
    train.add_argument("--max_tokens",      type=int,   default=None,
                       help="Stop training after this many tokens (e.g. 4_000_000_000 for 4B)")
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
    train.add_argument("--log_every",       type=int,   default=None)
    train.add_argument("--eval_every",      type=int,   default=None)
    train.add_argument("--save_every",      type=int,   default=None)
    train.add_argument("--keep_last_checkpoints", type=int, default=None)
    train.add_argument("--eval_batches",    type=int,   default=None)
    train.add_argument("--eval_seed",       type=int,   default=None)
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
    feat.add_argument("--debug_data_check",    action="store_true",
                      help="Log input/label shapes and first tokens at startup "
                           "to verify correct causal-LM label shifting")

    return p


if __name__ == "__main__":
    main()
