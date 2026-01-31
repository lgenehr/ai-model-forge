#!/usr/bin/env python3
"""
BitNet-Mamba Hybrid Training Script

A professional, object-oriented implementation combining:
- BitNet b1.58 quantization (weights in {-1, 0, 1})
- Mamba State Space Model architecture

Optimized for: NVIDIA RTX 4070 Ti Super (16GB) + Ryzen 9 7950X
Uses bfloat16 mixed precision throughout.

Author: AI Model Forge
License: MIT
"""

import os
import sys
import csv
import math
import json
import random
import logging
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional, Iterator, Tuple, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    import tiktoken
except ImportError:
    tiktoken = None
    print("Warning: tiktoken not available, will use fallback tokenizer")

try:
    from einops import rearrange, repeat
except ImportError:
    rearrange = None
    print("Warning: einops not available, using manual reshape operations")

try:
    from datasets import load_dataset, IterableDataset
except ImportError:
    raise ImportError("datasets library is required: pip install datasets")

# Optional: mamba-ssm for optimized CUDA kernels
# pip install mamba-ssm (requires CUDA toolkit)
MAMBA_SSM_AVAILABLE = False
try:
    from mamba_ssm import Mamba
    MAMBA_SSM_AVAILABLE = True
except ImportError:
    pass


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
    output_dir: str = "./ai-model-forge/bitnet-mamba-hybrid"
    checkpoint_dir: str = "./ai-model-forge/bitnet-mamba-hybrid/checkpoints"
    log_file: str = "./ai-model-forge/bitnet-mamba-hybrid/training.log"
    csv_file: str = "./ai-model-forge/bitnet-mamba-hybrid/loss_history.csv"

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
    Mamba Block with BitNet quantized projections.

    Implements the selective state space model from "Mamba: Linear-Time
    Sequence Modeling with Selective State Spaces" (Gu & Dao, 2023).

    Uses BitLinear for in_proj and out_proj to achieve 1.58-bit efficiency.
    """

    def __init__(self, config: ModelConfig, use_bitlinear: bool = True):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.d_state = config.d_state
        self.d_conv = config.d_conv
        self.d_inner = config.d_inner
        self.expand = config.expand

        Linear = BitLinear if use_bitlinear else nn.Linear

        # Input projection: d_model -> 2 * d_inner (for x and z paths)
        self.in_proj = Linear(self.d_model, 2 * self.d_inner, bias=config.bias)

        # Convolution for local context
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=config.d_conv,
            padding=config.d_conv - 1,
            groups=self.d_inner,
            bias=True
        )

        # SSM parameters projection
        # Projects to: delta (dt), B, C
        self.x_proj = nn.Linear(
            self.d_inner,
            config.d_state + config.d_state + 1,  # B + C + dt_rank
            bias=False
        )

        # dt (delta) projection
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        # A parameter (structured as log for stability)
        A = repeat(
            torch.arange(1, config.d_state + 1, dtype=torch.float32),
            'n -> d n',
            d=self.d_inner
        ) if rearrange else torch.arange(1, config.d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))

        # D parameter (skip connection)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = Linear(self.d_inner, self.d_model, bias=config.bias)

        # Layer normalization
        self.norm = RMSNorm(self.d_model)

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

    def ssm(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor
    ) -> torch.Tensor:
        """
        Selective State Space computation.

        Args:
            x: Input tensor [batch, seq_len, d_inner]
            dt: Delta (time step) [batch, seq_len, d_inner]
            B: Input-to-state matrix [batch, seq_len, d_state]
            C: State-to-output matrix [batch, seq_len, d_state]

        Returns:
            y: Output tensor [batch, seq_len, d_inner]
        """
        batch_size, seq_len, d_inner = x.shape
        d_state = self.config.d_state

        # Get A from log parameterization
        A = -torch.exp(self.A_log.float())  # [d_inner, d_state]

        # Discretize A and B
        # A_bar = exp(dt * A)
        # B_bar = dt * B (simplified)
        dt = F.softplus(dt)  # Ensure positive

        # Initialize state
        h = torch.zeros(batch_size, d_inner, d_state, device=x.device, dtype=x.dtype)

        outputs = []

        # Sequential SSM computation (can be parallelized with associative scan)
        for t in range(seq_len):
            dt_t = dt[:, t, :, None]  # [batch, d_inner, 1]
            B_t = B[:, t, None, :]     # [batch, 1, d_state]
            C_t = C[:, t, None, :]     # [batch, 1, d_state]
            x_t = x[:, t, :, None]     # [batch, d_inner, 1]

            # State update: h = A_bar * h + B_bar * x
            A_bar = torch.exp(dt_t * A.unsqueeze(0))  # [batch, d_inner, d_state]
            B_bar = dt_t * B_t.expand(-1, d_inner, -1)  # [batch, d_inner, d_state]

            h = A_bar * h + B_bar * x_t.expand(-1, -1, d_state)

            # Output: y = C * h
            y_t = (h * C_t.expand(-1, d_inner, -1)).sum(dim=-1)  # [batch, d_inner]
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)  # [batch, seq_len, d_inner]
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Mamba block.

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape

        # Residual connection
        residual = x
        x = self.norm(x)

        # Input projection splits into x and z paths
        xz = self.in_proj(x)
        x_path, z = xz.chunk(2, dim=-1)  # Each: [batch, seq_len, d_inner]

        # 1D convolution (causal)
        x_path = x_path.transpose(1, 2)  # [batch, d_inner, seq_len]
        x_path = self.conv1d(x_path)[:, :, :seq_len]  # Causal: remove future padding
        x_path = x_path.transpose(1, 2)  # [batch, seq_len, d_inner]

        # Activation
        x_path = F.silu(x_path)

        # SSM parameter projection
        x_dbc = self.x_proj(x_path)  # [batch, seq_len, d_state + d_state + 1]
        dt_rank = 1
        dt, B, C = x_dbc.split([dt_rank, self.config.d_state, self.config.d_state], dim=-1)

        # Project dt
        dt = self.dt_proj(dt)  # [batch, seq_len, d_inner]

        # SSM computation
        y = self.ssm(x_path, dt, B, C)

        # Skip connection with D
        y = y + x_path * self.D

        # Gate with z
        y = y * F.silu(z)

        # Output projection
        output = self.out_proj(y)
        output = self.dropout(output)

        return output + residual


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

        # Apply Mamba blocks
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
# Data Pipeline
# =============================================================================

class TokenizerWrapper:
    """Wrapper for tokenizer with fallback support"""

    def __init__(self, tokenizer_name: str = "gpt2"):
        if tiktoken is not None:
            self.tokenizer = tiktoken.get_encoding("gpt2")
            self.vocab_size = self.tokenizer.n_vocab
            self.eos_token_id = self.tokenizer.eot_token
            self.pad_token_id = self.tokenizer.eot_token
        else:
            # Fallback: simple character-level tokenizer
            print("Using fallback character tokenizer")
            self.tokenizer = None
            self.vocab_size = 50304
            self.eos_token_id = 0
            self.pad_token_id = 0

    def encode(self, text: str) -> List[int]:
        if self.tokenizer is not None:
            return self.tokenizer.encode(text, allowed_special={'<|endoftext|>'})
        else:
            return [ord(c) % self.vocab_size for c in text]

    def decode(self, tokens: List[int]) -> str:
        if self.tokenizer is not None:
            return self.tokenizer.decode(tokens)
        else:
            return ''.join(chr(t) for t in tokens if t < 128)


class BilingualDataPipeline:
    """
    Streaming data pipeline that mixes English and Portuguese data.

    English: HuggingFaceFW/fineweb-edu (sample-10BT)
    Portuguese: Wikipedia (pt)
    """

    def __init__(
        self,
        tokenizer: TokenizerWrapper,
        config: TrainingConfig,
        seed: int = 42
    ):
        self.tokenizer = tokenizer
        self.config = config
        self.seed = seed
        self.rng = random.Random(seed)

        # Track total tokens processed
        self.total_tokens = 0

        # Initialize datasets
        self._init_datasets()

    def _init_datasets(self):
        """Initialize streaming datasets"""
        logging.info("Loading English dataset: HuggingFaceFW/fineweb-edu (sample-10BT)")
        try:
            self.en_dataset = load_dataset(
                "HuggingFaceFW/fineweb-edu",
                name="sample-10BT",
                split="train",
                streaming=True,
                trust_remote_code=True
            )
            self.en_iter = iter(self.en_dataset)
        except Exception as e:
            logging.warning(f"Could not load fineweb-edu: {e}")
            logging.info("Falling back to wikitext-103 for English")
            self.en_dataset = load_dataset(
                "wikitext",
                "wikitext-103-raw-v1",
                split="train",
                streaming=True
            )
            self.en_iter = iter(self.en_dataset)

        logging.info("Loading Portuguese dataset: Wikipedia (pt)")
        try:
            self.pt_dataset = load_dataset(
                "wikipedia",
                "20220301.pt",
                split="train",
                streaming=True,
                trust_remote_code=True
            )
            self.pt_iter = iter(self.pt_dataset)
        except Exception as e:
            logging.warning(f"Could not load Portuguese Wikipedia: {e}")
            logging.info("Falling back to Portuguese Oscar subset")
            try:
                self.pt_dataset = load_dataset(
                    "oscar-corpus/OSCAR-2301",
                    language="pt",
                    split="train",
                    streaming=True,
                    trust_remote_code=True
                )
                self.pt_iter = iter(self.pt_dataset)
            except Exception as e2:
                logging.warning(f"Could not load OSCAR: {e2}")
                self.pt_dataset = None
                self.pt_iter = None

    def _get_text(self, sample: Dict, is_english: bool) -> str:
        """Extract text from dataset sample"""
        if 'text' in sample:
            return sample['text']
        elif 'content' in sample:
            return sample['content']
        else:
            # Try common text field names
            for key in ['sentence', 'document', 'article']:
                if key in sample:
                    return sample[key]
        return ""

    def _get_next_sample(self, is_english: bool) -> Optional[str]:
        """Get next text sample from the appropriate dataset"""
        try:
            if is_english:
                sample = next(self.en_iter)
            else:
                if self.pt_iter is None:
                    # Fallback to English if PT not available
                    sample = next(self.en_iter)
                    is_english = True
                else:
                    sample = next(self.pt_iter)

            return self._get_text(sample, is_english)
        except StopIteration:
            # Reset iterator
            if is_english:
                self.en_iter = iter(self.en_dataset)
                sample = next(self.en_iter)
            else:
                if self.pt_iter is not None:
                    self.pt_iter = iter(self.pt_dataset)
                    sample = next(self.pt_iter)
                else:
                    self.en_iter = iter(self.en_dataset)
                    sample = next(self.en_iter)

            return self._get_text(sample, is_english)

    def _create_batch_from_texts(
        self,
        texts: List[str],
        max_seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize and create a batch from text samples"""
        all_tokens = []

        for text in texts:
            tokens = self.tokenizer.encode(text)
            all_tokens.extend(tokens)
            all_tokens.append(self.tokenizer.eos_token_id)

        # Chunk into sequences of max_seq_len
        sequences = []
        for i in range(0, len(all_tokens) - max_seq_len, max_seq_len):
            seq = all_tokens[i:i + max_seq_len + 1]
            if len(seq) == max_seq_len + 1:
                sequences.append(seq)

        if not sequences:
            # Not enough tokens, create a padded sequence
            seq = all_tokens[:max_seq_len + 1]
            while len(seq) < max_seq_len + 1:
                seq.append(self.tokenizer.pad_token_id)
            sequences.append(seq)

        # Convert to tensors
        batch = torch.tensor(sequences, dtype=torch.long)
        input_ids = batch[:, :-1]
        labels = batch[:, 1:]

        return input_ids, labels

    def generate_batches(
        self,
        batch_size: int,
        max_seq_len: int
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor, int]]:
        """
        Generate training batches with interleaved EN/PT data.

        Yields:
            input_ids: [batch_size, max_seq_len]
            labels: [batch_size, max_seq_len]
            tokens_in_batch: Number of tokens in this batch
        """
        buffer = []
        buffer_tokens = 0
        target_buffer_tokens = batch_size * max_seq_len * 2  # Some overhead

        while self.total_tokens < self.config.max_tokens:
            # Decide language based on ratio
            is_english = self.rng.random() < self.config.en_ratio

            # Get next text sample
            text = self._get_next_sample(is_english)
            if text:
                buffer.append(text)
                buffer_tokens += len(text.split()) * 1.3  # Rough token estimate

            # Create batch when buffer is full enough
            if buffer_tokens >= target_buffer_tokens:
                input_ids, labels = self._create_batch_from_texts(buffer, max_seq_len)

                # Yield batches
                for i in range(0, len(input_ids), batch_size):
                    batch_input = input_ids[i:i + batch_size]
                    batch_labels = labels[i:i + batch_size]

                    if len(batch_input) == batch_size:
                        tokens_in_batch = batch_input.numel()
                        self.total_tokens += tokens_in_batch

                        yield batch_input, batch_labels, tokens_in_batch

                        if self.total_tokens >= self.config.max_tokens:
                            return

                # Reset buffer
                buffer = []
                buffer_tokens = 0


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
    """

    def __init__(
        self,
        model: BitNetMambaLM,
        train_config: TrainingConfig,
        model_config: ModelConfig
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
        self.scaler = GradScaler(enabled=train_config.use_amp and self.dtype == torch.float16)

        # Training state
        self.global_step = 0
        self.total_tokens = 0
        self.best_loss = float('inf')

        # Setup paths
        self._setup_paths()

        # Setup logging
        self._setup_logging()

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
            latest_checkpoint = checkpoints[-1]
            self.logger.info(f"Resuming from checkpoint: {latest_checkpoint}")
            self._load_checkpoint(latest_checkpoint)

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

        # Save regular checkpoint
        checkpoint_path = Path(self.train_config.checkpoint_dir) / f"checkpoint_{self.global_step:08d}.pt"
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")

        # Save best model
        if is_best:
            best_path = Path(self.train_config.output_dir) / "best_model.pt"
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model: {best_path}")

        # Clean up old checkpoints (keep last 3)
        checkpoints = sorted(Path(self.train_config.checkpoint_dir).glob("checkpoint_*.pt"))
        for old_checkpoint in checkpoints[:-3]:
            old_checkpoint.unlink()

    def _load_checkpoint(self, checkpoint_path: Path):
        """Load training checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.total_tokens = checkpoint['total_tokens']
        self.best_loss = checkpoint['best_loss']

        self.logger.info(f"Resumed from step {self.global_step}, tokens: {self.total_tokens:,}")

    def train(self, data_pipeline: BilingualDataPipeline):
        """Main training loop"""
        self.model.train()

        self.logger.info("=" * 60)
        self.logger.info("Starting BitNet-Mamba Hybrid Training")
        self.logger.info(f"Model parameters: {self.model.get_num_params():,}")
        self.logger.info(f"Target tokens: {self.train_config.max_tokens:,}")
        self.logger.info(f"Max steps: {self.train_config.max_steps:,}")
        self.logger.info(f"Effective batch size: {self.train_config.batch_size * self.train_config.gradient_accumulation_steps}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Dtype: {self.dtype}")
        self.logger.info("=" * 60)

        # Training metrics
        accumulated_loss = 0.0
        accumulated_tokens = 0
        step_start_time = datetime.now()

        # Data generator
        batch_generator = data_pipeline.generate_batches(
            batch_size=self.train_config.batch_size,
            max_seq_len=self.train_config.max_seq_len
        )

        # Update data pipeline's total tokens to match checkpoint
        data_pipeline.total_tokens = self.total_tokens

        for batch_idx, (input_ids, labels, tokens_in_batch) in enumerate(batch_generator):
            # Move to device
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)

            # Forward pass with mixed precision
            with autocast(device_type='cuda', dtype=self.dtype, enabled=self.train_config.use_amp):
                outputs = self.model(input_ids, labels)
                loss = outputs['loss'] / self.train_config.gradient_accumulation_steps

            # Backward pass
            if self.train_config.use_amp and self.dtype == torch.float16:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            accumulated_loss += loss.item() * self.train_config.gradient_accumulation_steps
            accumulated_tokens += tokens_in_batch

            # Gradient accumulation step
            if (batch_idx + 1) % self.train_config.gradient_accumulation_steps == 0:
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

                self.optimizer.zero_grad()

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

                    # Track best loss
                    if avg_loss < self.best_loss:
                        self.best_loss = avg_loss

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

        # Final checkpoint and cleanup
        self._save_checkpoint(is_best=True)
        self.csv_file.close()

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

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--max_seq_len", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--max_tokens", type=int, default=4_000_000_000, help="Maximum tokens to train on")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=2000, help="Warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")

    # Data arguments
    parser.add_argument("--en_ratio", type=float, default=0.5, help="English data ratio")
    parser.add_argument("--pt_ratio", type=float, default=0.5, help="Portuguese data ratio")

    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./ai-model-forge/bitnet-mamba-hybrid",
                        help="Output directory")

    # Other arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no_amp", action="store_true", help="Disable automatic mixed precision")

    return parser.parse_args()


def main():
    args = parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Create configurations
    model_config = ModelConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        d_state=args.d_state,
        d_conv=args.d_conv,
        expand=args.expand,
        dropout=args.dropout,
        max_seq_len=args.max_seq_len
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
        en_ratio=args.en_ratio,
        pt_ratio=args.pt_ratio,
        output_dir=args.output_dir,
        checkpoint_dir=f"{args.output_dir}/checkpoints",
        log_file=f"{args.output_dir}/training.log",
        csv_file=f"{args.output_dir}/loss_history.csv"
    )

    # Print configuration
    print("=" * 60)
    print("BitNet-Mamba Hybrid Training Configuration")
    print("=" * 60)
    print(f"Model Config: {asdict(model_config)}")
    print(f"Training Config: {asdict(train_config)}")
    print("=" * 60)

    # Create tokenizer
    tokenizer = TokenizerWrapper()
    model_config.vocab_size = tokenizer.vocab_size

    # Create model
    model = BitNetMambaLM(model_config)
    print(f"Model created with {model.get_num_params():,} parameters")

    # Create data pipeline
    data_pipeline = BilingualDataPipeline(tokenizer, train_config, seed=args.seed)

    # Create trainer and start training
    trainer = Trainer(model, train_config, model_config)
    trainer.train(data_pipeline)


if __name__ == "__main__":
    main()
