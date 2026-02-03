#!/usr/bin/env python3
"""
BitNet-Mamba Hybrid Training Script

A professional, object-oriented implementation combining:
- BitNet b1.58 quantization (weights in {-1, 0, 1})
- Mamba State Space Model architecture

Optimized for: NVIDIA RTX 4070 Ti Super (16GB) + Ryzen 9 7950X
Uses bfloat16 mixed precision throughout.

Performance optimizations:
- Pre-tokenized memory-mapped datasets (no HTTP requests during training)
- Multi-worker data loading with prefetching
- CUDA optimizations (cudnn.benchmark, TF32)
- Optimized Mamba CUDA kernels

Author: AI Model Forge
License: MIT
"""

import os
import sys
import csv
import math
import json
import pickle
import random
import signal
import logging
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional, Iterator, Tuple, Dict, Any, List

# =============================================================================
# Graceful Shutdown Handling
# =============================================================================
_shutdown_requested = False
_trainer_instance = None  # Global reference for signal handler


def _training_signal_handler(signum, frame):
    """Handle interrupt signals gracefully during training"""
    global _shutdown_requested

    if _shutdown_requested:
        print("\n" + "=" * 60)
        print("Forced shutdown requested. Exiting immediately...")
        print("=" * 60)
        sys.exit(1)

    _shutdown_requested = True
    print("\n" + "=" * 60)
    print("Interrupt received! Finishing current step and saving checkpoint...")
    print("Press Ctrl+C again to force quit (may lose current step)")
    print("=" * 60)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.checkpoint import checkpoint

# ============================================================================
# CUDA Performance Optimizations
# ============================================================================
# Enable cuDNN autotuner - finds optimal convolution algorithms
torch.backends.cudnn.benchmark = True

# Enable TF32 for faster matrix multiplications on Ampere+ GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Disable torch.compile/Dynamo by default (incompatible with Mamba CUDA kernels)
# This prevents accidental compilation that hurts performance
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True

# Reduce verbosity of HTTP requests and dataset loading
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("filelock").setLevel(logging.WARNING)

# Weights & Biases for experiment tracking
WANDB_AVAILABLE = False
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    print("Warning: wandb not available, experiment tracking disabled")

try:
    from einops import rearrange, repeat
except ImportError:
    rearrange = None
    print("Warning: einops not available, using manual reshape operations")

# Import efficient data loader
try:
    from data_loader import (
        create_dataloader,
        check_preprocessed_data,
        get_dataset_info,
        InfiniteDataLoader
    )
    DATA_LOADER_AVAILABLE = True
except ImportError:
    DATA_LOADER_AVAILABLE = False
    print("Warning: data_loader module not found, will use streaming fallback")

# Optional: mamba-ssm for optimized CUDA kernels
# pip install mamba-ssm (requires CUDA toolkit)
MAMBA_SSM_AVAILABLE = False
try:
    from mamba_ssm import Mamba
    MAMBA_SSM_AVAILABLE = True
except ImportError:
    pass

# Import selective_scan_fn for optimized SSM computation
selective_scan_fn = None
SELECTIVE_SCAN_CUDA = False
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    SELECTIVE_SCAN_CUDA = True
except ImportError:
    pass


# =============================================================================
# Mamba Optimization Verification
# =============================================================================

def verify_mamba_optimizations() -> Dict[str, Any]:
    """
    Verify Mamba optimizations are properly configured.

    Returns:
        Dictionary with optimization status and details
    """
    status = {
        'mamba_ssm_installed': MAMBA_SSM_AVAILABLE,
        'selective_scan_cuda': SELECTIVE_SCAN_CUDA,
        'cuda_available': torch.cuda.is_available(),
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
        'tf32_cudnn': torch.backends.cudnn.allow_tf32,
        'gpu_name': None,
        'gpu_memory_gb': None,
        'warnings': [],
        'optimizations_ok': True,
    }

    # Get GPU info
    if torch.cuda.is_available():
        status['gpu_name'] = torch.cuda.get_device_name(0)
        status['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9

    # Check critical optimizations
    if not MAMBA_SSM_AVAILABLE:
        status['warnings'].append(
            "mamba-ssm not installed. Install with: pip install mamba-ssm"
        )
        status['optimizations_ok'] = False

    if not SELECTIVE_SCAN_CUDA:
        status['warnings'].append(
            "selective_scan_fn CUDA kernel not available. "
            "Training will be significantly slower."
        )
        status['optimizations_ok'] = False

    if not torch.cuda.is_available():
        status['warnings'].append("CUDA not available. Training on CPU will be very slow.")
        status['optimizations_ok'] = False

    return status


def print_optimization_status(status: Dict[str, Any], logger=None):
    """Print optimization status to console/logger"""
    log_fn = logger.info if logger else print

    log_fn("=" * 60)
    log_fn("Mamba Optimization Status")
    log_fn("=" * 60)

    # Hardware
    log_fn(f"CUDA Available: {status['cuda_available']}")
    if status['gpu_name']:
        log_fn(f"GPU: {status['gpu_name']}")
        log_fn(f"GPU Memory: {status['gpu_memory_gb']:.1f} GB")

    # Mamba optimizations
    log_fn(f"mamba-ssm Installed: {status['mamba_ssm_installed']}")
    log_fn(f"selective_scan_fn CUDA: {status['selective_scan_cuda']}")

    # PyTorch optimizations
    log_fn(f"cuDNN Benchmark: {status['cudnn_benchmark']}")
    log_fn(f"TF32 MatMul: {status['tf32_matmul']}")
    log_fn(f"TF32 cuDNN: {status['tf32_cudnn']}")

    # Warnings
    if status['warnings']:
        log_fn("-" * 60)
        log_fn("WARNINGS:")
        for warning in status['warnings']:
            log_fn(f"  - {warning}")

    # Overall status
    log_fn("-" * 60)
    if status['optimizations_ok']:
        log_fn("All critical optimizations are enabled.")
    else:
        log_fn("Some optimizations are missing. Performance may be degraded.")

    log_fn("=" * 60)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ModelConfig:
    """Configuration for BitNet-Mamba Hybrid Model"""
    vocab_size: int = 50304  # Padded to multiple of 64 for efficiency
    d_model: int = 768
    n_layers: int = 12
    d_state: int = 16  # SSM state expansion factor
    d_conv: int = 4    # Local convolution width
    expand: int = 2    # Block expansion factor
    dropout: float = 0.1
    bias: bool = False
    max_seq_len: int = 2048

    # BitNet specific
    bitnet_groups: int = 1  # Number of groups for BitLinear

    # Memory optimization
    use_gradient_checkpointing: bool = False  # Trade compute for memory

    def __post_init__(self):
        self.d_inner = self.expand * self.d_model


@dataclass
class TrainingConfig:
    """Configuration for training hyperparameters"""
    # Data
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    max_seq_len: int = 2048

    # Training
    max_tokens: int = 4_000_000_000  # 4B tokens target
    max_steps: Optional[int] = None  # Will be computed
    warmup_steps: int = 2000
    learning_rate: float = 3e-4
    min_lr: float = 1e-5
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0

    # Mixed precision
    use_amp: bool = True
    dtype: str = "bfloat16"

    # Logging & Checkpointing
    log_interval: int = 10
    eval_interval: int = 500
    checkpoint_interval: int = 1000

    # Data mixing (EN:PT ratio)
    en_ratio: float = 0.5
    pt_ratio: float = 0.5

    # Paths
    data_dir: str = "./data/tokenized"  # Pre-tokenized data directory
    output_dir: str = "./ai-model-forge/bitnet-mamba-hybrid"
    checkpoint_dir: str = "./ai-model-forge/bitnet-mamba-hybrid/checkpoints"
    log_file: str = "./ai-model-forge/bitnet-mamba-hybrid/training.log"
    csv_file: str = "./ai-model-forge/bitnet-mamba-hybrid/loss_history.csv"

    # DataLoader settings
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2

    # Weights & Biases
    use_wandb: bool = False
    wandb_project: str = "bitnet-mamba-hybrid"
    wandb_run_name: Optional[str] = None
    wandb_entity: Optional[str] = None

    def __post_init__(self):
        # Compute max steps from token budget
        tokens_per_step = self.batch_size * self.gradient_accumulation_steps * self.max_seq_len
        self.max_steps = self.max_tokens // tokens_per_step
        self.tokens_per_step = tokens_per_step


# =============================================================================
# BitNet Components
# =============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


def ste_sign(x: torch.Tensor) -> torch.Tensor:
    """Straight-Through Estimator for sign function"""
    return (x.sign() - x).detach() + x


def quantize_weights_ternary(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize weights to {-1, 0, 1} following BitNet b1.58 paper.

    Steps:
    1. Compute mean absolute value (scale factor)
    2. Normalize weights
    3. Round to nearest integer in {-1, 0, 1}
    """
    # Compute scale as mean of absolute values
    scale = weight.abs().mean().clamp(min=1e-5)

    # Normalize and quantize
    weight_normalized = weight / scale

    # Round to nearest of {-1, 0, 1}
    # Using round-to-nearest with threshold
    weight_quant = weight_normalized.clamp(-1, 1).round()

    # STE: gradient flows through as if no quantization
    weight_quant = (weight_quant - weight_normalized).detach() + weight_normalized

    return weight_quant, scale


class BitLinear(nn.Module):
    """
    BitNet b1.58 Linear Layer

    Weights are quantized to {-1, 0, 1} during forward pass.
    Uses RMSNorm for activation normalization.
    Straight-Through Estimator enables gradient flow.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        groups: int = 1
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.groups = groups

        # Full precision weights (will be quantized during forward)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        # Input normalization
        self.input_norm = RMSNorm(in_features)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize input
        x_norm = self.input_norm(x)

        # Quantize weights to ternary
        weight_quant, scale = quantize_weights_ternary(self.weight)

        # Linear operation with quantized weights
        output = F.linear(x_norm, weight_quant, self.bias)

        # Scale output
        output = output * scale

        return output

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'


# =============================================================================
# Mamba SSM Components
# =============================================================================

class MambaBlock(nn.Module):
    """
    Versão Otimizada:
    - Projeções (in_proj, out_proj) usam BitLinear (1.58 bits)
    - Recorrência (SSM) usa o kernel CUDA oficial da NVIDIA/Mamba
    """

    def __init__(self, config: ModelConfig, use_bitlinear: bool = True):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.d_state = config.d_state
        self.d_conv = config.d_conv
        self.d_inner = config.d_inner
        self.expand = config.expand

        # Define qual Linear usar (BitNet ou Padrão)
        Linear = BitLinear if use_bitlinear else nn.Linear

        # 1. Projeção de Entrada (Usa BitNet para economizar VRAM/Compute)
        self.in_proj = Linear(self.d_model, 2 * self.d_inner, bias=config.bias)

        # 2. Convolução 1D (Contexto local)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=config.d_conv,
            groups=self.d_inner,
            padding=config.d_conv - 1,
        )

        # 3. Projeção x -> (dt, B, C)
        # Essa geralmente mantemos em full precision ou standard linear pois é sensível
        self.x_proj = nn.Linear(
            self.d_inner,
            self.config.d_state + self.config.d_state + 1,  # dt, B, C
            bias=False
        )

        # 4. Projeção dt
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        # Parâmetros A e D (Logarítmicos para estabilidade)
        A = repeat(
            torch.arange(1, config.d_state + 1, dtype=torch.float32),
            'n -> d n',
            d=self.d_inner,
        ).contiguous()
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # 5. Projeção de Saída (Usa BitNet)
        self.out_proj = Linear(self.d_inner, self.d_model, bias=config.bias)
        self.norm = RMSNorm(self.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        residual = x

        # Normaliza
        x = self.norm(x)

        # Projeção BitNet (Input -> x, z)
        xz = self.in_proj(x)
        x_path, z = xz.chunk(2, dim=-1)

        # Convolução
        x_path = x_path.transpose(1, 2)  # [B, D, L]
        x_path = self.conv1d(x_path)[:, :, :seq_len]
        x_path = F.silu(x_path)
        x_path = x_path.transpose(1, 2)  # [B, L, D]

        # --- Início da Lógica Otimizada com CUDA ---

        # Calcula dt, B, C
        x_dbc = self.x_proj(x_path)
        dt, B, C = torch.split(x_dbc, [1, self.config.d_state, self.config.d_state], dim=-1)

        # Prepara para o Kernel CUDA
        # O kernel espera shapes específicos. Geralmente (Batch, Seq, Dim) ou (Batch, Dim, Seq)
        # Vamos usar a implementação oficial que lida com a matemática complexa

        A = -torch.exp(self.A_log.float())  # Força float32 para estabilidade do SSM

        # Se tivermos o kernel instalado, usamos ele:
        if selective_scan_fn is not None:
            # O kernel selective_scan_fn espera:
            # u: [B, D, L]
            # delta: [B, D, L]
            # A: [D, N]
            # B: [B, N, L]
            # C: [B, N, L]
            # D: [D]

            # Ajuste de shapes
            u = x_path.transpose(1, 2)                  # [B, D, L]
            dt = self.dt_proj(dt).transpose(1, 2)       # [B, D, L]
            B_t = B.transpose(1, 2)                     # [B, N, L]
            C_t = C.transpose(1, 2)                     # [B, N, L]

            y = selective_scan_fn(
                u, dt, A, B_t, C_t, self.D.float(),
                z=None,  # Fazemos a multiplicação por z depois
                delta_bias=None,
                delta_softplus=True,
                return_last_state=False
            )

            y = y.transpose(1, 2)  # [B, L, D]

        else:
            # Fallback lento (Sua implementação original caso CUDA falhe)
            raise ImportError("mamba-ssm não detectado. O treino será inviável sem GPU kernel.")

        # --- Fim da Lógica Otimizada ---

        # Gate com z (Swish gate)
        y = y * F.silu(z)

        # Projeção de Saída BitNet
        out = self.out_proj(y)
        out = self.dropout(out)

        return out + residual


# =============================================================================
# Full Model
# =============================================================================

class BitNetMambaLM(nn.Module):
    """
    BitNet-Mamba Language Model

    Combines Mamba SSM architecture with BitNet b1.58 quantization
    for efficient language modeling.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)

        # Mamba blocks
        self.layers = nn.ModuleList([
            MambaBlock(config, use_bitlinear=True)
            for _ in range(config.n_layers)
        ])

        # Output normalization
        self.norm_f = RMSNorm(config.d_model)

        # Language model head (tied with embeddings)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.embedding.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, BitLinear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the language model.

        Args:
            input_ids: Token indices [batch, seq_len]
            labels: Target token indices [batch, seq_len] (optional)

        Returns:
            Dictionary with 'logits' and optionally 'loss'
        """
        # Token embeddings
        x = self.embedding(input_ids)

        # Apply Mamba blocks with optional gradient checkpointing
        if self.config.use_gradient_checkpointing and self.training:
            # Use gradient checkpointing to save memory
            for layer in self.layers:
                x = checkpoint(layer, x, use_reentrant=False)
        else:
            # Standard forward pass
            for layer in self.layers:
                x = layer(x)

        # Final normalization
        x = self.norm_f(x)

        # Language model head
        logits = self.lm_head(x)

        output = {'logits': logits}

        # Compute loss if labels provided
        if labels is not None:
            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
            output['loss'] = loss

        return output

    def get_num_params(self) -> int:
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())

    def get_num_trainable_params(self) -> int:
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# Data Pipeline (Efficient Memory-Mapped Loading)
# =============================================================================

def create_efficient_dataloader(train_config: TrainingConfig, seed: int = 42) -> InfiniteDataLoader:
    """
    Create an efficient dataloader using pre-tokenized memory-mapped data.

    This eliminates all HTTP requests during training by loading pre-processed
    data from disk.

    Args:
        train_config: Training configuration
        seed: Random seed

    Returns:
        InfiniteDataLoader instance

    Raises:
        FileNotFoundError: If preprocessed data is not found
    """
    if not DATA_LOADER_AVAILABLE:
        raise ImportError(
            "data_loader module not available. "
            "Make sure data_loader.py is in the same directory."
        )

    # Check for preprocessed data
    if not check_preprocessed_data(train_config.data_dir):
        raise FileNotFoundError(
            f"Preprocessed data not found at {train_config.data_dir}\n"
            f"Please run the preprocessing script first:\n"
            f"  python preprocess_datasets.py --output_dir {train_config.data_dir}\n"
        )

    # Get dataset info
    dataset_info = get_dataset_info(train_config.data_dir)
    logging.info(f"Dataset info:")
    logging.info(f"  English tokens: {dataset_info['en_tokens']:,}")
    logging.info(f"  Portuguese tokens: {dataset_info['pt_tokens']:,}")
    logging.info(f"  Total tokens: {dataset_info['total_tokens']:,}")

    # Create dataloader
    dataloader = create_dataloader(
        data_dir=train_config.data_dir,
        batch_size=train_config.batch_size,
        max_seq_len=train_config.max_seq_len,
        en_ratio=train_config.en_ratio,
        pt_ratio=train_config.pt_ratio,
        num_workers=train_config.num_workers,
        pin_memory=train_config.pin_memory,
        prefetch_factor=train_config.prefetch_factor,
        seed=seed,
        epoch_tokens=train_config.max_tokens,
        drop_last=True,
    )

    return dataloader


# =============================================================================
# Trainer
# =============================================================================

class Trainer:
    """
    Training orchestrator with support for:
    - Mixed precision training (bfloat16)
    - Gradient accumulation
    - Resumable checkpoints
    - Logging and metrics tracking
    - Weights & Biases integration
    """

    def __init__(
        self,
        model: BitNetMambaLM,
        train_config: TrainingConfig,
        model_config: ModelConfig,
        wandb_api_key: Optional[str] = None
    ):
        self.model = model
        self.train_config = train_config
        self.model_config = model_config

        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Setup dtype
        self.dtype = torch.bfloat16 if train_config.dtype == "bfloat16" else torch.float16

        # Optimizer
        self.optimizer = self._create_optimizer()

        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=train_config.max_steps - train_config.warmup_steps,
            eta_min=train_config.min_lr
        )

        # Gradient scaler for mixed precision
        self.scaler = GradScaler('cuda', enabled=train_config.use_amp and self.dtype == torch.float16)

        # Training state
        self.global_step = 0
        self.total_tokens = 0
        self.best_loss = float('inf')

        # Setup paths
        self._setup_paths()

        # Setup logging
        self._setup_logging()

        # Setup Weights & Biases
        self._setup_wandb(wandb_api_key)

        # Try to resume from checkpoint
        self._try_resume()

    def _create_optimizer(self) -> AdamW:
        """Create optimizer with weight decay for non-bias parameters"""
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'bias' in name or 'norm' in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)

        optimizer_groups = [
            {'params': decay_params, 'weight_decay': self.train_config.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]

        return AdamW(
            optimizer_groups,
            lr=self.train_config.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8
        )

    def _setup_paths(self):
        """Create necessary directories"""
        Path(self.train_config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.train_config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def _setup_logging(self):
        """Setup logging to file and console"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)s | %(message)s',
            handlers=[
                logging.FileHandler(self.train_config.log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

        # CSV writer for loss history
        self.csv_file = open(self.train_config.csv_file, 'a', newline='')
        self.csv_writer = csv.writer(self.csv_file)

        # Write header if file is empty
        if os.path.getsize(self.train_config.csv_file) == 0:
            self.csv_writer.writerow([
                'step', 'loss', 'lr', 'tokens', 'tokens_per_sec', 'timestamp'
            ])
            self.csv_file.flush()

    def _setup_wandb(self, api_key: Optional[str] = None):
        """Setup Weights & Biases for experiment tracking"""
        self.wandb_enabled = False

        if not self.train_config.use_wandb:
            self.logger.info("Weights & Biases disabled")
            return

        if not WANDB_AVAILABLE:
            self.logger.warning("wandb not installed. Install with: pip install wandb")
            return

        try:
            # Login with API key if provided
            if api_key:
                wandb.login(key=api_key)
                self.logger.info("Logged in to Weights & Biases with API key")

            # Prepare config for wandb
            config_dict = {
                **asdict(self.model_config),
                **asdict(self.train_config),
                'num_params': self.model.get_num_params(),
                'device': str(self.device),
                'dtype': str(self.dtype),
            }

            # Initialize wandb run
            wandb.init(
                project=self.train_config.wandb_project,
                name=self.train_config.wandb_run_name,
                entity=self.train_config.wandb_entity,
                config=config_dict,
                resume="allow",
                dir=self.train_config.output_dir
            )

            # Watch model for gradient tracking
            wandb.watch(self.model, log="gradients", log_freq=100)

            self.wandb_enabled = True
            self.logger.info(f"Weights & Biases initialized: {wandb.run.url}")

        except Exception as e:
            self.logger.warning(f"Failed to initialize wandb: {e}")
            self.wandb_enabled = False

    def _log_to_wandb(self, metrics: Dict[str, Any], step: int):
        """Log metrics to Weights & Biases"""
        if self.wandb_enabled and WANDB_AVAILABLE:
            wandb.log(metrics, step=step)

    def _finish_wandb(self):
        """Finish wandb run and upload final artifacts"""
        if self.wandb_enabled and WANDB_AVAILABLE:
            # Log final model as artifact
            try:
                best_model_path = Path(self.train_config.output_dir) / "best_model.pt"
                if best_model_path.exists():
                    artifact = wandb.Artifact(
                        name=f"model-{wandb.run.id}",
                        type="model",
                        description="Best BitNet-Mamba model checkpoint"
                    )
                    artifact.add_file(str(best_model_path))
                    wandb.log_artifact(artifact)
                    self.logger.info("Uploaded best model to Weights & Biases")
            except Exception as e:
                self.logger.warning(f"Failed to upload model artifact: {e}")

            wandb.finish()
            self.logger.info("Weights & Biases run finished")

    def _get_lr(self) -> float:
        """Get current learning rate with warmup"""
        if self.global_step < self.train_config.warmup_steps:
            # Linear warmup
            return self.train_config.learning_rate * (self.global_step + 1) / self.train_config.warmup_steps
        else:
            return self.scheduler.get_last_lr()[0]

    def _set_lr(self, lr: float):
        """Set learning rate for all parameter groups"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def _try_resume(self):
        """Try to resume from the latest checkpoint"""
        checkpoint_dir = Path(self.train_config.checkpoint_dir)
        checkpoints = sorted(checkpoint_dir.glob("checkpoint_*.pt"))

        if checkpoints:
            # Try checkpoints from newest to oldest
            for checkpoint_path in reversed(checkpoints):
                self.logger.info(f"Attempting to resume from checkpoint: {checkpoint_path}")
                if self._load_checkpoint(checkpoint_path):
                    return  # Successfully loaded
                else:
                    self.logger.warning(f"Failed to load checkpoint: {checkpoint_path}")

            self.logger.warning("All checkpoints failed to load. Starting from scratch.")

    def _save_checkpoint_atomic(self, checkpoint: dict, target_path: Path):
        """Save checkpoint atomically to prevent corruption from interrupts"""
        temp_path = target_path.with_suffix('.pt.tmp')
        try:
            torch.save(checkpoint, temp_path)
            # Atomic rename - if this is interrupted, old checkpoint remains valid
            os.replace(temp_path, target_path)
        except Exception:
            # Clean up temp file if save failed
            if temp_path.exists():
                temp_path.unlink()
            raise

    def _save_checkpoint(self, is_best: bool = False):
        """Save training checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'total_tokens': self.total_tokens,
            'best_loss': self.best_loss,
            'model_config': asdict(self.model_config),
            'train_config': asdict(self.train_config)
        }

        # Save regular checkpoint (atomic to prevent corruption)
        checkpoint_path = Path(self.train_config.checkpoint_dir) / f"checkpoint_{self.global_step:08d}.pt"
        self._save_checkpoint_atomic(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")

        # Save best model (atomic)
        if is_best:
            best_path = Path(self.train_config.output_dir) / "best_model.pt"
            self._save_checkpoint_atomic(checkpoint, best_path)
            self.logger.info(f"Saved best model: {best_path}")

        # Clean up old checkpoints (keep last 3)
        checkpoints = sorted(Path(self.train_config.checkpoint_dir).glob("checkpoint_*.pt"))
        for old_checkpoint in checkpoints[:-3]:
            old_checkpoint.unlink()

    def _load_checkpoint(self, checkpoint_path: Path) -> bool:
        """Load training checkpoint

        Handles both regular and torch.compile() models by remapping state_dict keys.

        Returns:
            True if checkpoint was loaded successfully, False otherwise
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            model_state = checkpoint['model_state_dict']

            # Check if model is compiled (has _orig_mod prefix in keys)
            model_keys = list(self.model.state_dict().keys())
            checkpoint_keys = list(model_state.keys())

            model_is_compiled = any(k.startswith('_orig_mod.') for k in model_keys)
            checkpoint_is_compiled = any(k.startswith('_orig_mod.') for k in checkpoint_keys)

            # Remap keys if there's a mismatch
            if model_is_compiled and not checkpoint_is_compiled:
                # Model is compiled but checkpoint is not - add _orig_mod. prefix
                self.logger.info("Remapping checkpoint keys for compiled model...")
                model_state = {'_orig_mod.' + k: v for k, v in model_state.items()}
            elif not model_is_compiled and checkpoint_is_compiled:
                # Model is not compiled but checkpoint is - remove _orig_mod. prefix
                self.logger.info("Remapping checkpoint keys for non-compiled model...")
                model_state = {k.replace('_orig_mod.', ''): v for k, v in model_state.items()}

            self.model.load_state_dict(model_state)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.global_step = checkpoint['global_step']
            self.total_tokens = checkpoint['total_tokens']
            self.best_loss = checkpoint['best_loss']

            self.logger.info(f"Resumed from step {self.global_step}, tokens: {self.total_tokens:,}")
            return True
        except (EOFError, pickle.UnpicklingError) as e:
            # Only rename truly corrupted files (file format issues)
            self.logger.error(f"Checkpoint file corrupted {checkpoint_path}: {e}")
            corrupted_path = checkpoint_path.with_suffix('.pt.corrupted')
            try:
                checkpoint_path.rename(corrupted_path)
                self.logger.info(f"Renamed corrupted checkpoint to: {corrupted_path}")
            except OSError as rename_error:
                self.logger.warning(f"Could not rename corrupted checkpoint: {rename_error}")
            return False
        except (RuntimeError, KeyError) as e:
            # Key mismatches or missing keys - don't rename, just skip
            self.logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
            self.logger.warning("Checkpoint may have incompatible architecture. Trying next...")
            return False

    def _save_interrupt_checkpoint(self):
        """Save checkpoint when training is interrupted (Ctrl+C)"""
        self.logger.info("=" * 60)
        self.logger.info("Saving interrupt checkpoint...")

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'total_tokens': self.total_tokens,
            'best_loss': self.best_loss,
            'model_config': asdict(self.model_config),
            'train_config': asdict(self.train_config),
            'interrupted': True,
        }

        # Save interrupt checkpoint with special naming (atomic to prevent corruption)
        checkpoint_path = Path(self.train_config.checkpoint_dir) / f"checkpoint_interrupt_{self.global_step:08d}.pt"
        self._save_checkpoint_atomic(checkpoint, checkpoint_path)
        self.logger.info(f"Saved interrupt checkpoint: {checkpoint_path}")

        # Also save as latest checkpoint for easy resume (atomic)
        latest_path = Path(self.train_config.checkpoint_dir) / f"checkpoint_{self.global_step:08d}.pt"
        if not latest_path.exists():
            self._save_checkpoint_atomic(checkpoint, latest_path)
            self.logger.info(f"Saved checkpoint: {latest_path}")

        self.logger.info("=" * 60)
        self.logger.info("Training interrupted safely. To resume, run the same command.")
        self.logger.info(f"Progress: Step {self.global_step:,}, Tokens {self.total_tokens:,}")
        self.logger.info("=" * 60)

    def train(self, dataloader: InfiniteDataLoader):
        """
        Main training loop using efficient memory-mapped dataloader.

        Args:
            dataloader: InfiniteDataLoader instance from data_loader module

        Supports graceful shutdown:
        - Press Ctrl+C once to finish current step and save checkpoint
        - Press Ctrl+C twice to force quit immediately
        """
        global _shutdown_requested, _trainer_instance

        # Setup signal handler for graceful shutdown
        _trainer_instance = self
        signal.signal(signal.SIGINT, _training_signal_handler)
        signal.signal(signal.SIGTERM, _training_signal_handler)

        self.model.train()

        self.logger.info("=" * 60)
        self.logger.info("Starting BitNet-Mamba Hybrid Training")
        self.logger.info(f"Model parameters: {self.model.get_num_params():,}")
        self.logger.info(f"Target tokens: {self.train_config.max_tokens:,}")
        self.logger.info(f"Max steps: {self.train_config.max_steps:,}")
        self.logger.info(f"Effective batch size: {self.train_config.batch_size * self.train_config.gradient_accumulation_steps}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Dtype: {self.dtype}")
        self.logger.info(f"Num workers: {self.train_config.num_workers}")
        self.logger.info(f"Pin memory: {self.train_config.pin_memory}")
        self.logger.info("")
        self.logger.info("Press Ctrl+C to save checkpoint and stop training")
        self.logger.info("=" * 60)

        # Training metrics
        accumulated_loss = 0.0
        accumulated_tokens = 0
        step_start_time = datetime.now()
        batch_idx = 0
        interrupted = False

        try:
            # Use the efficient infinite dataloader
            for batch in dataloader:
                # Check for shutdown request at start of each batch
                if _shutdown_requested:
                    self.logger.info("Shutdown requested, finishing current step...")
                    interrupted = True
                    break
                # Get input_ids and labels from batch dict
                input_ids = batch['input_ids'].to(self.device, non_blocking=True)
                labels = batch['labels'].to(self.device, non_blocking=True)
                tokens_in_batch = input_ids.numel()

                # Forward pass with mixed precision
                with autocast('cuda', dtype=self.dtype, enabled=self.train_config.use_amp):
                    outputs = self.model(input_ids, labels)
                    loss = outputs['loss'] / self.train_config.gradient_accumulation_steps

                # Backward pass
                if self.train_config.use_amp and self.dtype == torch.float16:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                accumulated_loss += loss.item() * self.train_config.gradient_accumulation_steps
                accumulated_tokens += tokens_in_batch
                batch_idx += 1

                # Gradient accumulation step
                if batch_idx % self.train_config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.train_config.use_amp and self.dtype == torch.float16:
                        self.scaler.unscale_(self.optimizer)

                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.train_config.max_grad_norm
                    )

                    # Optimizer step
                    if self.train_config.use_amp and self.dtype == torch.float16:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    self.optimizer.zero_grad(set_to_none=True)  # More efficient

                    # Update learning rate
                    current_lr = self._get_lr()
                    self._set_lr(current_lr)

                    if self.global_step >= self.train_config.warmup_steps:
                        self.scheduler.step()

                    self.global_step += 1
                    self.total_tokens += accumulated_tokens

                    # Logging
                    if self.global_step % self.train_config.log_interval == 0:
                        step_time = (datetime.now() - step_start_time).total_seconds()
                        tokens_per_sec = accumulated_tokens / step_time if step_time > 0 else 0

                        avg_loss = accumulated_loss / self.train_config.log_interval

                        self.logger.info(
                            f"Step {self.global_step:>7} | "
                            f"Loss: {avg_loss:.4f} | "
                            f"LR: {current_lr:.2e} | "
                            f"Tokens: {self.total_tokens:>12,} | "
                            f"Tok/s: {tokens_per_sec:>8,.0f}"
                        )

                        # Write to CSV
                        self.csv_writer.writerow([
                            self.global_step,
                            avg_loss,
                            current_lr,
                            self.total_tokens,
                            tokens_per_sec,
                            datetime.now().isoformat()
                        ])
                        self.csv_file.flush()

                        # Log to Weights & Biases
                        self._log_to_wandb({
                            'train/loss': avg_loss,
                            'train/learning_rate': current_lr,
                            'train/tokens': self.total_tokens,
                            'train/tokens_per_sec': tokens_per_sec,
                            'train/epoch': self.total_tokens / self.train_config.max_tokens,
                        }, step=self.global_step)

                        # Track best loss
                        if avg_loss < self.best_loss:
                            self.best_loss = avg_loss
                            self._log_to_wandb({'train/best_loss': self.best_loss}, step=self.global_step)

                        accumulated_loss = 0.0
                        step_start_time = datetime.now()

                    accumulated_tokens = 0

                    # Checkpoint
                    if self.global_step % self.train_config.checkpoint_interval == 0:
                        self._save_checkpoint(is_best=False)

                    # Check if training is complete
                    if self.global_step >= self.train_config.max_steps:
                        self.logger.info("Reached max steps, training complete!")
                        break

                    # Check for shutdown request after each optimizer step
                    if _shutdown_requested:
                        self.logger.info("Shutdown requested after step completion...")
                        interrupted = True
                        break

        except KeyboardInterrupt:
            # This catches any KeyboardInterrupt not handled by signal handler
            self.logger.warning("KeyboardInterrupt caught!")
            interrupted = True

        except Exception as e:
            # Save checkpoint on unexpected errors too
            self.logger.error(f"Training error: {e}")
            self.logger.info("Saving emergency checkpoint due to error...")
            self._save_interrupt_checkpoint()
            raise

        # Handle interrupted training
        if interrupted:
            self._save_interrupt_checkpoint()
            self.csv_file.close()
            self._finish_wandb()
            return

        # Normal completion - Final checkpoint and cleanup
        self._save_checkpoint(is_best=True)
        self.csv_file.close()

        # Log final metrics to wandb
        self._log_to_wandb({
            'final/total_steps': self.global_step,
            'final/total_tokens': self.total_tokens,
            'final/best_loss': self.best_loss,
        }, step=self.global_step)

        # Finish wandb run
        self._finish_wandb()

        self.logger.info("=" * 60)
        self.logger.info("Training Complete!")
        self.logger.info(f"Final step: {self.global_step}")
        self.logger.info(f"Total tokens: {self.total_tokens:,}")
        self.logger.info(f"Best loss: {self.best_loss:.4f}")
        self.logger.info("=" * 60)


# =============================================================================
# Main
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train BitNet-Mamba Hybrid Language Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model arguments
    parser.add_argument("--d_model", type=int, default=768, help="Model dimension")
    parser.add_argument("--n_layers", type=int, default=12, help="Number of layers")
    parser.add_argument("--d_state", type=int, default=16, help="SSM state dimension")
    parser.add_argument("--d_conv", type=int, default=4, help="Convolution kernel size")
    parser.add_argument("--expand", type=int, default=2, help="Block expansion factor")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing to save memory (trades compute for memory)")

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--max_seq_len", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--max_tokens", type=int, default=4_000_000_000, help="Maximum tokens to train on")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=2000, help="Warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")

    # Data arguments
    parser.add_argument("--data_dir", type=str, default="./data/tokenized",
                        help="Directory containing pre-tokenized data (run preprocess_datasets.py first)")
    parser.add_argument("--en_ratio", type=float, default=0.5, help="English data ratio")
    parser.add_argument("--pt_ratio", type=float, default=0.5, help="Portuguese data ratio")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--prefetch_factor", type=int, default=2, help="Dataloader prefetch factor")
    parser.add_argument("--no_pin_memory", action="store_true", help="Disable pinned memory for dataloader")

    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./ai-model-forge/bitnet-mamba-hybrid",
                        help="Output directory")

    # Other arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no_amp", action="store_true", help="Disable automatic mixed precision")
    parser.add_argument("--skip_mamba_check", action="store_true", help="Skip Mamba optimization verification")

    # High throughput optimization arguments
    parser.add_argument("--compile", action="store_true",
                        help="Enable torch.compile() for model optimization (requires PyTorch 2.0+)")
    parser.add_argument("--compile_mode", type=str, default="reduce-overhead",
                        choices=["default", "reduce-overhead", "max-autotune"],
                        help="torch.compile optimization mode")
    parser.add_argument("--high_throughput", action="store_true",
                        help="Enable high throughput mode: larger batch, more workers, torch.compile")

    # Weights & Biases arguments
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_api_key", type=str, default=None,
                        help="Weights & Biases API key (or set WANDB_API_KEY env var)")
    parser.add_argument("--wandb_project", type=str, default="bitnet-mamba-hybrid",
                        help="Weights & Biases project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="Weights & Biases run name (auto-generated if not provided)")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="Weights & Biases entity (username or team name)")

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup basic logging early
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    # High throughput mode: automatically adjust parameters for maximum GPU utilization
    if args.high_throughput:
        print("=" * 60)
        print("HIGH THROUGHPUT MODE ENABLED")
        print("=" * 60)
        print("Adjusting parameters for maximum GPU utilization...")

        # Increase batch size if not explicitly set higher
        if args.batch_size <= 8:
            args.batch_size = 12
            print(f"  -> batch_size: 12 (increased from default)")

        # Reduce gradient accumulation to compensate
        if args.grad_accum >= 4:
            args.grad_accum = 3
            print(f"  -> grad_accum: 3 (adjusted for larger batch)")

        # Increase data loading parallelism
        if args.num_workers <= 4:
            args.num_workers = 8
            print(f"  -> num_workers: 8 (increased for faster data loading)")

        if args.prefetch_factor <= 2:
            args.prefetch_factor = 4
            print(f"  -> prefetch_factor: 4 (increased for better pipelining)")

        # Enable torch.compile by default in high throughput mode
        if hasattr(torch, 'compile') and not args.compile:
            args.compile = True
            print(f"  -> torch.compile: enabled (JIT optimization)")

        # Disable gradient checkpointing for speed (uses more memory but faster)
        if args.gradient_checkpointing:
            args.gradient_checkpointing = False
            print(f"  -> gradient_checkpointing: disabled (speed over memory)")

        print("=" * 60)

    # Set random seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Verify Mamba optimizations at startup
    if not args.skip_mamba_check:
        opt_status = verify_mamba_optimizations()
        print_optimization_status(opt_status)

        if not opt_status['optimizations_ok']:
            print("\nWARNING: Some critical optimizations are missing.")
            print("Training will proceed but may be slower than expected.")
            print("Use --skip_mamba_check to suppress this warning.\n")

    # Create configurations
    model_config = ModelConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        d_state=args.d_state,
        d_conv=args.d_conv,
        expand=args.expand,
        dropout=args.dropout,
        max_seq_len=args.max_seq_len,
        use_gradient_checkpointing=args.gradient_checkpointing
    )

    train_config = TrainingConfig(
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_seq_len=args.max_seq_len,
        max_tokens=args.max_tokens,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        use_amp=not args.no_amp,
        # Data settings
        data_dir=args.data_dir,
        en_ratio=args.en_ratio,
        pt_ratio=args.pt_ratio,
        num_workers=args.num_workers,
        pin_memory=not args.no_pin_memory,
        prefetch_factor=args.prefetch_factor,
        # Output paths
        output_dir=args.output_dir,
        checkpoint_dir=f"{args.output_dir}/checkpoints",
        log_file=f"{args.output_dir}/training.log",
        csv_file=f"{args.output_dir}/loss_history.csv",
        # Weights & Biases
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_entity=args.wandb_entity
    )

    # Print configuration
    print("=" * 60)
    print("BitNet-Mamba Hybrid Training Configuration")
    print("=" * 60)
    print(f"Model Config: {asdict(model_config)}")
    print(f"Training Config: {asdict(train_config)}")
    print("=" * 60)

    # Create model
    model = BitNetMambaLM(model_config)
    print(f"Model created with {model.get_num_params():,} parameters")

    # Apply torch.compile() for optimized execution (PyTorch 2.0+)
    if args.compile and hasattr(torch, 'compile'):
        print(f"Compiling model with torch.compile(mode='{args.compile_mode}')...")
        print("  This may take a few minutes on first run but speeds up training significantly.")
        try:
            model = torch.compile(model, mode=args.compile_mode)
            print("  Model compiled successfully!")
        except Exception as e:
            print(f"  Warning: torch.compile() failed: {e}")
            print("  Continuing without compilation...")
    elif args.compile:
        print("Warning: torch.compile requested but not available (requires PyTorch 2.0+)")

    # Create efficient dataloader (memory-mapped, no HTTP requests)
    try:
        dataloader = create_efficient_dataloader(train_config, seed=args.seed)
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("\nTo preprocess the datasets, run:")
        print(f"  python preprocess_datasets.py --output_dir {args.data_dir}")
        print("\nThis will download and tokenize the datasets once.")
        print("Subsequent training runs will load data from disk instantly.")
        sys.exit(1)
    except ImportError as e:
        print(f"\nERROR: {e}")
        print("\nMake sure data_loader.py is in the same directory as this script.")
        sys.exit(1)

    # Get wandb API key from argument or environment variable
    wandb_api_key = args.wandb_api_key or os.environ.get('WANDB_API_KEY')

    # Create trainer and start training
    trainer = Trainer(model, train_config, model_config, wandb_api_key=wandb_api_key)
    trainer.train(dataloader)


if __name__ == "__main__":
    main()
