#!/usr/bin/env python3
"""
BitNet-Mamba Hybrid Inference Script

Loads a trained BitNet-Mamba model and generates text using
greedy decoding or nucleus (top-p) sampling.

Author: AI Model Forge
License: MIT
"""

import os
import sys
import math
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import tiktoken
except ImportError:
    tiktoken = None
    print("Warning: tiktoken not available, will use fallback tokenizer")

try:
    from einops import repeat
except ImportError:
    repeat = None


# =============================================================================
# Configuration (must match training)
# =============================================================================

@dataclass
class ModelConfig:
    """Configuration for BitNet-Mamba Hybrid Model"""
    vocab_size: int = 50304
    d_model: int = 768
    n_layers: int = 12
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    dropout: float = 0.0  # Disable dropout for inference
    bias: bool = False
    max_seq_len: int = 2048
    bitnet_groups: int = 1

    def __post_init__(self):
        self.d_inner = self.expand * self.d_model


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


def quantize_weights_ternary(weight: torch.Tensor):
    """Quantize weights to {-1, 0, 1} following BitNet b1.58"""
    scale = weight.abs().mean().clamp(min=1e-5)
    weight_normalized = weight / scale
    weight_quant = weight_normalized.clamp(-1, 1).round()
    weight_quant = (weight_quant - weight_normalized).detach() + weight_normalized
    return weight_quant, scale


class BitLinear(nn.Module):
    """BitNet b1.58 Linear Layer"""

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

        self.weight = nn.Parameter(torch.empty(out_features, in_features))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        self.input_norm = RMSNorm(in_features)
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.input_norm(x)
        weight_quant, scale = quantize_weights_ternary(self.weight)
        output = F.linear(x_norm, weight_quant, self.bias)
        output = output * scale
        return output


# =============================================================================
# Mamba SSM Components
# =============================================================================

class MambaBlock(nn.Module):
    """Mamba Block with BitNet quantized projections"""

    def __init__(self, config: ModelConfig, use_bitlinear: bool = True):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.d_state = config.d_state
        self.d_conv = config.d_conv
        self.d_inner = config.d_inner
        self.expand = config.expand

        Linear = BitLinear if use_bitlinear else nn.Linear

        self.in_proj = Linear(self.d_model, 2 * self.d_inner, bias=config.bias)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=config.d_conv,
            padding=config.d_conv - 1,
            groups=self.d_inner,
            bias=True
        )

        self.x_proj = nn.Linear(
            self.d_inner,
            config.d_state + config.d_state + 1,
            bias=False
        )

        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        if repeat is not None:
            A = repeat(
                torch.arange(1, config.d_state + 1, dtype=torch.float32),
                'n -> d n',
                d=self.d_inner
            )
        else:
            A = torch.arange(1, config.d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1).clone()

        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = Linear(self.d_inner, self.d_model, bias=config.bias)
        self.norm = RMSNorm(self.d_model)
        self.dropout = nn.Dropout(config.dropout)

        # Cache for autoregressive generation
        self._conv_cache: Optional[torch.Tensor] = None
        self._ssm_state: Optional[torch.Tensor] = None

    def reset_cache(self):
        """Reset the inference cache"""
        self._conv_cache = None
        self._ssm_state = None

    def ssm_step(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor
    ) -> torch.Tensor:
        """Single step of SSM for autoregressive generation"""
        batch_size = x.shape[0]
        d_state = self.config.d_state

        A = -torch.exp(self.A_log.float())
        dt = F.softplus(dt)

        # Initialize state if needed
        if self._ssm_state is None:
            self._ssm_state = torch.zeros(
                batch_size, self.d_inner, d_state,
                device=x.device, dtype=x.dtype
            )

        # State update
        dt = dt[:, :, None]  # [batch, d_inner, 1]
        B = B[:, None, :]    # [batch, 1, d_state]
        C = C[:, None, :]    # [batch, 1, d_state]
        x = x[:, :, None]    # [batch, d_inner, 1]

        A_bar = torch.exp(dt * A.unsqueeze(0))
        B_bar = dt * B.expand(-1, self.d_inner, -1)

        self._ssm_state = A_bar * self._ssm_state + B_bar * x.expand(-1, -1, d_state)

        y = (self._ssm_state * C.expand(-1, self.d_inner, -1)).sum(dim=-1)
        return y

    def ssm(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor
    ) -> torch.Tensor:
        """Full SSM computation for parallel processing"""
        batch_size, seq_len, d_inner = x.shape
        d_state = self.config.d_state

        A = -torch.exp(self.A_log.float())
        dt = F.softplus(dt)

        h = torch.zeros(batch_size, d_inner, d_state, device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(seq_len):
            dt_t = dt[:, t, :, None]
            B_t = B[:, t, None, :]
            C_t = C[:, t, None, :]
            x_t = x[:, t, :, None]

            A_bar = torch.exp(dt_t * A.unsqueeze(0))
            B_bar = dt_t * B_t.expand(-1, d_inner, -1)

            h = A_bar * h + B_bar * x_t.expand(-1, -1, d_state)
            y_t = (h * C_t.expand(-1, d_inner, -1)).sum(dim=-1)
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)

    def forward(
        self,
        x: torch.Tensor,
        use_cache: bool = False
    ) -> torch.Tensor:
        """Forward pass with optional caching for generation"""
        batch_size, seq_len, _ = x.shape

        residual = x
        x = self.norm(x)

        xz = self.in_proj(x)
        x_path, z = xz.chunk(2, dim=-1)

        # Convolution
        if use_cache and seq_len == 1:
            # Cached single-token generation
            if self._conv_cache is None:
                self._conv_cache = torch.zeros(
                    batch_size, self.d_inner, self.d_conv - 1,
                    device=x.device, dtype=x.dtype
                )

            # Append new input and slide window
            conv_input = torch.cat([self._conv_cache, x_path.transpose(1, 2)], dim=-1)
            self._conv_cache = conv_input[:, :, 1:]

            x_path = self.conv1d(conv_input)[:, :, -1:]
            x_path = x_path.transpose(1, 2)
        else:
            x_path = x_path.transpose(1, 2)
            x_path = self.conv1d(x_path)[:, :, :seq_len]
            x_path = x_path.transpose(1, 2)

        x_path = F.silu(x_path)

        # SSM parameters
        x_dbc = self.x_proj(x_path)
        dt, B, C = x_dbc.split([1, self.config.d_state, self.config.d_state], dim=-1)
        dt = self.dt_proj(dt)

        # SSM computation
        if use_cache and seq_len == 1:
            y = self.ssm_step(x_path.squeeze(1), dt.squeeze(1), B.squeeze(1), C.squeeze(1))
            y = y.unsqueeze(1)
        else:
            y = self.ssm(x_path, dt, B, C)

        # Skip connection and gate
        y = y + x_path * self.D
        y = y * F.silu(z)

        output = self.out_proj(y)
        output = self.dropout(output)

        return output + residual


# =============================================================================
# Full Model
# =============================================================================

class BitNetMambaLM(nn.Module):
    """BitNet-Mamba Language Model"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([
            MambaBlock(config, use_bitlinear=True)
            for _ in range(config.n_layers)
        ])
        self.norm_f = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

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

    def reset_cache(self):
        """Reset all layer caches"""
        for layer in self.layers:
            layer.reset_cache()

    def forward(
        self,
        input_ids: torch.Tensor,
        use_cache: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        x = self.embedding(input_ids)

        for layer in self.layers:
            x = layer(x, use_cache=use_cache)

        x = self.norm_f(x)
        logits = self.lm_head(x)

        return {'logits': logits}


# =============================================================================
# Tokenizer
# =============================================================================

class TokenizerWrapper:
    """Wrapper for tokenizer with fallback support"""

    def __init__(self):
        if tiktoken is not None:
            self.tokenizer = tiktoken.get_encoding("gpt2")
            self.vocab_size = self.tokenizer.n_vocab
            self.eos_token_id = self.tokenizer.eot_token
            self.pad_token_id = self.tokenizer.eot_token
        else:
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
            return ''.join(chr(t) for t in tokens if 32 <= t < 127)


# =============================================================================
# Text Generation
# =============================================================================

class TextGenerator:
    """
    Text generation with various decoding strategies.

    Supports:
    - Greedy decoding
    - Nucleus (top-p) sampling
    - Temperature scaling
    - Top-k sampling
    """

    def __init__(
        self,
        model: BitNetMambaLM,
        tokenizer: TokenizerWrapper,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype

        self.model.eval()
        self.model.to(device)

    def _sample_nucleus(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_p: float
    ) -> torch.Tensor:
        """Nucleus (top-p) sampling"""
        if temperature == 0:
            return logits.argmax(dim=-1)

        # Apply temperature
        logits = logits / temperature

        # Sort logits in descending order
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = float('-inf')

        # Sample from the filtered distribution
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def _sample_top_k(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_k: int
    ) -> torch.Tensor:
        """Top-k sampling"""
        if temperature == 0:
            return logits.argmax(dim=-1)

        # Apply temperature
        logits = logits / temperature

        # Keep only top-k tokens
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 0,
        stop_on_eos: bool = True,
        use_cache: bool = True,
        stream: bool = False
    ) -> Union[str, None]:
        """
        Generate text from a prompt.

        Args:
            prompt: Input text prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0 = greedy)
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling threshold (0 = disabled)
            stop_on_eos: Stop generation on EOS token
            use_cache: Use KV cache for efficient generation
            stream: Print tokens as they are generated

        Returns:
            Generated text string
        """
        # Reset cache
        if use_cache:
            self.model.reset_cache()

        # Tokenize prompt
        input_ids = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], device=self.device, dtype=torch.long)

        generated_tokens = []

        # Initial forward pass with full context
        with torch.autocast(device_type='cuda', dtype=self.dtype):
            outputs = self.model(input_tensor, use_cache=use_cache)

        logits = outputs['logits'][:, -1, :]  # Last token logits

        # Sample first token
        if top_k > 0:
            next_token = self._sample_top_k(logits, temperature, top_k)
        else:
            next_token = self._sample_nucleus(logits, temperature, top_p)

        generated_tokens.append(next_token.item())

        if stream:
            print(self.tokenizer.decode([next_token.item()]), end='', flush=True)

        # Autoregressive generation
        for _ in range(max_tokens - 1):
            # Check for EOS
            if stop_on_eos and next_token.item() == self.tokenizer.eos_token_id:
                break

            # Forward pass with single token
            input_tensor = next_token.unsqueeze(0).unsqueeze(0)

            with torch.autocast(device_type='cuda', dtype=self.dtype):
                outputs = self.model(input_tensor, use_cache=use_cache)

            logits = outputs['logits'][:, -1, :]

            # Sample next token
            if top_k > 0:
                next_token = self._sample_top_k(logits, temperature, top_k)
            else:
                next_token = self._sample_nucleus(logits, temperature, top_p)

            generated_tokens.append(next_token.item())

            if stream:
                print(self.tokenizer.decode([next_token.item()]), end='', flush=True)

        if stream:
            print()  # Newline after streaming

        # Decode generated tokens
        generated_text = self.tokenizer.decode(generated_tokens)
        return generated_text

    def generate_greedy(self, prompt: str, max_tokens: int = 100) -> str:
        """Greedy decoding (temperature=0)"""
        return self.generate(prompt, max_tokens=max_tokens, temperature=0)


# =============================================================================
# Model Loading
# =============================================================================

def load_model(
    checkpoint_path: str,
    device: Optional[torch.device] = None,
    config_override: Optional[Dict[str, Any]] = None
) -> tuple:
    """
    Load a trained BitNet-Mamba model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Target device
        config_override: Optional config overrides

    Returns:
        Tuple of (model, tokenizer, config)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract config from checkpoint
    if 'model_config' in checkpoint:
        config_dict = checkpoint['model_config']
        # Remove computed fields that aren't part of the dataclass definition
        valid_fields = {'vocab_size', 'd_model', 'n_layers', 'd_state', 'd_conv',
                        'expand', 'dropout', 'bias', 'max_seq_len', 'bitnet_groups'}
        filtered_config = {k: v for k, v in config_dict.items() if k in valid_fields}
        config = ModelConfig(**filtered_config)
    else:
        # Use default config
        config = ModelConfig()

    # Apply overrides
    if config_override:
        for key, value in config_override.items():
            if hasattr(config, key):
                setattr(config, key, value)

    # Disable dropout for inference
    config.dropout = 0.0

    # Create model
    model = BitNetMambaLM(config)

    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    # Create tokenizer
    tokenizer = TokenizerWrapper()

    print(f"Model loaded successfully!")
    print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  - Device: {device}")
    print(f"  - Config: d_model={config.d_model}, n_layers={config.n_layers}")

    return model, tokenizer, config


# =============================================================================
# Interactive Mode
# =============================================================================

def interactive_mode(generator: TextGenerator, args: argparse.Namespace):
    """Run an interactive text generation session"""
    print("\n" + "=" * 60)
    print("BitNet-Mamba Hybrid Interactive Generation")
    print("=" * 60)
    print("Commands:")
    print("  /quit or /exit  - Exit the program")
    print("  /temp <value>   - Set temperature (current: {})".format(args.temperature))
    print("  /top_p <value>  - Set top-p (current: {})".format(args.top_p))
    print("  /top_k <value>  - Set top-k (current: {})".format(args.top_k))
    print("  /max <value>    - Set max tokens (current: {})".format(args.max_tokens))
    print("  /greedy         - Use greedy decoding")
    print("  /sample         - Use sampling (default)")
    print("=" * 60)

    temperature = args.temperature
    top_p = args.top_p
    top_k = args.top_k
    max_tokens = args.max_tokens
    greedy = args.greedy

    while True:
        try:
            prompt = input("\nPrompt> ").strip()

            if not prompt:
                continue

            # Handle commands
            if prompt.lower() in ['/quit', '/exit']:
                print("Goodbye!")
                break
            elif prompt.startswith('/temp '):
                temperature = float(prompt.split()[1])
                print(f"Temperature set to {temperature}")
                continue
            elif prompt.startswith('/top_p '):
                top_p = float(prompt.split()[1])
                print(f"Top-p set to {top_p}")
                continue
            elif prompt.startswith('/top_k '):
                top_k = int(prompt.split()[1])
                print(f"Top-k set to {top_k}")
                continue
            elif prompt.startswith('/max '):
                max_tokens = int(prompt.split()[1])
                print(f"Max tokens set to {max_tokens}")
                continue
            elif prompt == '/greedy':
                greedy = True
                print("Using greedy decoding")
                continue
            elif prompt == '/sample':
                greedy = False
                print("Using sampling")
                continue

            # Generate text
            print("\nGenerating...\n")
            print("-" * 40)

            if greedy:
                output = generator.generate_greedy(prompt, max_tokens=max_tokens)
            else:
                output = generator.generate(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    stream=True
                )

            if not args.stream:
                print(output)

            print("-" * 40)

        except KeyboardInterrupt:
            print("\n\nInterrupted. Type /quit to exit.")
            continue
        except Exception as e:
            print(f"Error: {e}")
            continue


# =============================================================================
# Main
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BitNet-Mamba Hybrid Text Generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./ai-model-forge/bitnet-mamba-hybrid/best_model.pt",
        help="Path to model checkpoint"
    )

    # Generation arguments
    parser.add_argument("--prompt", type=str, default=None, help="Input prompt (if not provided, enters interactive mode)")
    parser.add_argument("--max_tokens", type=int, default=100, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Nucleus sampling threshold")
    parser.add_argument("--top_k", type=int, default=0, help="Top-k sampling threshold (0=disabled)")
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding")
    parser.add_argument("--stream", action="store_true", help="Stream output tokens")

    # Other arguments
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"],
                        help="Data type for inference")

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup device and dtype
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32
    }
    dtype = dtype_map[args.dtype]

    # Check if checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        # Try alternate paths
        alternate_paths = [
            Path("./ai-model-forge/bitnet-mamba-hybrid/checkpoints").glob("checkpoint_*.pt"),
        ]

        found = False
        for path_gen in alternate_paths:
            checkpoints = sorted(list(path_gen))
            if checkpoints:
                checkpoint_path = checkpoints[-1]
                print(f"Using latest checkpoint: {checkpoint_path}")
                found = True
                break

        if not found:
            print(f"Error: Checkpoint not found at {args.checkpoint}")
            print("Please train a model first using train_hybrid-mamba-bitnet.py")
            sys.exit(1)

    # Load model
    model, tokenizer, config = load_model(str(checkpoint_path), device=device)

    # Create generator
    generator = TextGenerator(model, tokenizer, device, dtype)

    if args.prompt:
        # Single generation mode
        print("\n" + "=" * 60)
        print("Prompt:", args.prompt)
        print("=" * 60)

        if args.greedy:
            output = generator.generate_greedy(args.prompt, max_tokens=args.max_tokens)
        else:
            output = generator.generate(
                args.prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                stream=args.stream
            )

        if not args.stream:
            print("\nGeneration:")
            print("-" * 40)
            print(output)
            print("-" * 40)
    else:
        # Interactive mode
        interactive_mode(generator, args)


if __name__ == "__main__":
    main()
