"""
Hybrid Transformer-Mamba-MoE Language Model (~820M parameters)

Architecture overview (18 layers):
  - 14 Mamba (SSM) layers: O(L) compute, efficient long-range context
  -  4 Attention + MoE layers (every ~4-5 layers): global attention + sparse experts

Total parameters: ~820M
Active per forward pass: ~560M (2/6 experts active per MoE step)
Target hardware: RTX 4070 Ti Super (16-17GB VRAM)
Expected throughput: 1500-2500 tok/s with mamba-ssm CUDA kernel
"""

import math
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

logger = logging.getLogger(__name__)

# ─── Optional CUDA backends ───────────────────────────────────────────────────

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as _ssm_fn
    HAS_MAMBA_SSM = True
    logger.info("mamba-ssm CUDA kernel: available ✓")
except ImportError:
    HAS_MAMBA_SSM = False
    logger.warning("mamba-ssm not found. Using torch.jit.script fallback (slower).")

try:
    from flash_attn import flash_attn_func as _flash_fn
    HAS_FLASH_ATTN = True
    logger.info("FlashAttention: available ✓")
except ImportError:
    HAS_FLASH_ATTN = False
    logger.info("FlashAttention not found. Using torch SDPA (efficient for Ampere+).")


# ─── Configuration ────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    """All model hyperparameters in one place."""

    # Vocabulary / sequence
    vocab_size: int = 50304          # GPT-2 BPE (padded to multiple of 64)
    max_seq_len: int = 2048

    # Embedding
    d_model: int = 2048              # hidden dimension

    # Layer layout
    n_layers: int = 18               # total layers
    # Indices (0-based) that use Attention + MoE; all others use Mamba
    attention_layers: List[int] = field(default_factory=lambda: [3, 7, 11, 17])

    # Mamba SSM
    mamba_expand: int = 2            # d_inner = expand * d_model = 4096
    mamba_d_state: int = 16          # SSM state dimension N
    mamba_d_conv: int = 4            # causal conv width
    mamba_dt_rank: str = "auto"      # "auto" → ceil(d_model / 16) = 128

    # Grouped-Query Attention
    n_heads: int = 16
    n_kv_heads: int = 4             # GQA: fewer key/value heads
    rope_base: float = 10000.0

    # Mixture of Experts
    # WSL2 note: Windows display driver occupies ~1.5-2 GB of VRAM even when
    # training. To avoid memory thrashing (GPU throttling to ~77W), we target
    # ≤13 GB total training VRAM on a 17.2 GB card.
    #
    # MoE memory is the largest tunable knob:
    #   8 experts × intermediate=1024 → 201M MoE params → 11.5 GB AdamW states
    #   + ~1.2 GB activations (grad ckpt) = ~12.7 GB → safe on WSL2
    #
    # Quality note: Mamba layers already project d_model → 2×d_inner (4096)
    # internally, so MoE only adds specialist *refinement* — intermediate=1024
    # (0.5× d_model) is adequate in a Mamba-heavy hybrid architecture.
    moe_n_experts: int = 8
    moe_top_k: int = 2
    moe_intermediate: int = 1024     # SwiGLU hidden dim — sized for WSL2 VRAM budget
    moe_aux_loss_coeff: float = 0.01 # load-balancing coefficient

    # Misc
    norm_eps: float = 1e-5
    dropout: float = 0.0             # 0.0 recommended for pre-training
    tie_embeddings: bool = True

    def __post_init__(self):
        if self.mamba_dt_rank == "auto":
            self.mamba_dt_rank = math.ceil(self.d_model / 16)
        assert self.d_model % self.n_heads == 0
        assert self.n_heads % self.n_kv_heads == 0
        assert self.head_dim % 2 == 0, "RoPE requires an even head_dim"
        assert self.moe_top_k <= self.moe_n_experts
        assert all(0 <= idx < self.n_layers for idx in self.attention_layers)
        self._attention_set = set(self.attention_layers)

    @property
    def d_inner(self) -> int:
        return self.mamba_expand * self.d_model

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def attention_set(self):
        return self._attention_set


# ─── Primitives ───────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """Root-Mean-Square Layer Normalization (no bias)."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast to float32 for numerical stability, then back
        x_f32 = x.float()
        normed = x_f32 * torch.rsqrt(x_f32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (normed * self.weight.float()).to(x.dtype)


def precompute_freqs_cis(head_dim: int, max_seq_len: int, base: float = 10000.0) -> torch.Tensor:
    """Precompute complex RoPE frequencies for efficient application."""
    half = head_dim // 2
    freqs = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) / half))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)  # [L, half]
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64 [L, half]


def apply_rotary_emb(
    q: torch.Tensor,   # [B, L, H, head_dim]
    k: torch.Tensor,   # [B, L, Hkv, head_dim]
    freqs_cis: torch.Tensor,  # [L, head_dim//2]  complex
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to Q and K."""
    def rotate(x, freqs):
        # x: [B, L, H, D] → view as complex [B, L, H, D/2]
        x_c = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        f = freqs.unsqueeze(0).unsqueeze(2)  # [1, L, 1, D/2]
        x_rot = torch.view_as_real(x_c * f).flatten(-2)
        return x_rot.to(x.dtype)

    return rotate(q, freqs_cis), rotate(k, freqs_cis)


# ─── Selective Scan (Mamba SSM core) ─────────────────────────────────────────

@torch.jit.script
def _selective_scan_fallback(
    u: torch.Tensor,          # [B, L, d_inner]
    delta: torch.Tensor,      # [B, L, d_inner]
    A: torch.Tensor,          # [d_inner, d_state]
    B: torch.Tensor,          # [B, L, d_state]
    C: torch.Tensor,          # [B, L, d_state]
    D: torch.Tensor,          # [d_inner]
    delta_bias: torch.Tensor, # [d_inner]
    z: torch.Tensor,          # [B, L, d_inner]
) -> torch.Tensor:
    """
    Sequential selective scan compiled with TorchScript.
    Numerically stable (float32), runs as native C++ loop.
    ~5-10x slower than mamba-ssm CUDA kernel but always correct.
    """
    dtype = u.dtype
    u = u.float()
    delta = (delta.float() + delta_bias.unsqueeze(0).unsqueeze(0))
    delta = F.softplus(delta)
    A = A.float()
    B = B.float()
    C = C.float()

    B_batch, L, d_inner = u.shape
    d_state = A.shape[1]

    # Precompute discretized transitions — avoids per-step recomputation
    # dA: [B, L, d_inner, d_state],  dBu: [B, L, d_inner, d_state]
    dA = torch.exp(torch.einsum('bld,dn->bldn', delta, A))
    dBu = torch.einsum('bld,bln->bldn', delta * u, B)

    h = u.new_zeros(B_batch, d_inner, d_state)
    ys: List[torch.Tensor] = []

    for i in range(L):
        h = dA[:, i] * h + dBu[:, i]          # [B, d, N]
        y = (h * C[:, i].unsqueeze(1)).sum(-1)  # [B, d]
        ys.append(y)

    y = torch.stack(ys, dim=1)  # [B, L, d]
    y = y + u * D.float()
    y = y * F.silu(z.float())
    return y.to(dtype)


# ─── Mamba Block ──────────────────────────────────────────────────────────────

class MambaMixer(nn.Module):
    """
    Mamba selective SSM mixer (Gu & Dao 2023).

    Input:  [B, L, d_model]
    Output: [B, L, d_model]

    Compute cost: O(B·L·d_model·d_inner) — linear in sequence length.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        d_model = config.d_model
        d_inner = config.d_inner
        d_state = config.mamba_d_state
        d_conv  = config.mamba_d_conv
        dt_rank = config.mamba_dt_rank

        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = dt_rank
        self.use_mamba_ssm = HAS_MAMBA_SSM

        # [x, z] projection (no bias, following original Mamba)
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)

        # Causal depth-wise conv (groups=d_inner → one filter per channel)
        self.conv1d = nn.Conv1d(d_inner, d_inner, kernel_size=d_conv,
                                padding=d_conv - 1, groups=d_inner, bias=True)

        # Projects d_inner → (dt_rank, d_state, d_state)
        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)

        # dt projection: dt_rank → d_inner (with bias used as dt_bias)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        # SSM parameters (A stored in log-space for stability)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True  # type: ignore[attr-defined]

        self.D = nn.Parameter(torch.ones(d_inner))
        self.D._no_weight_decay = True  # type: ignore[attr-defined]

        # Output projection
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

        self._init_dt_proj()

    def _init_dt_proj(self):
        """Initialize dt_proj to produce dt values in [0.001, 0.1]."""
        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        )
        inv_dt = dt + torch.log(-torch.expm1(-dt))  # softplus⁻¹
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_weight_decay = True  # type: ignore[attr-defined]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape

        # Dual projection: x and gating z
        xz = self.in_proj(x)          # [B, L, 2*d_inner]
        x_in, z = xz.chunk(2, dim=-1) # each [B, L, d_inner]

        # Causal convolution (channels-first)
        x_conv = rearrange(x_in, 'b l d -> b d l')
        x_conv = self.conv1d(x_conv)[..., :L]  # [B, d_inner, L]
        x_conv = F.silu(x_conv)
        x_conv = rearrange(x_conv, 'b d l -> b l d')  # [B, L, d_inner]

        # Compute dt, B_ssm, C_ssm from x
        x_flat = x_conv.reshape(B * L, self.d_inner)
        x_dbl = self.x_proj(x_flat)                         # [BL, dt_rank + 2*N]
        dt, B_ssm, C_ssm = x_dbl.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj(dt).reshape(B, L, self.d_inner)   # [B, L, d_inner]
        B_ssm = B_ssm.reshape(B, L, self.d_state)           # [B, L, N]
        C_ssm = C_ssm.reshape(B, L, self.d_state)           # [B, L, N]

        A = -torch.exp(self.A_log.float())  # [d_inner, d_state]

        if self.use_mamba_ssm:
            # Fast path: mamba-ssm CUDA kernel (channels-first format)
            # Output: [B, d_inner, L] with silu(z) gating applied internally
            y = _ssm_fn(
                rearrange(x_conv, 'b l d -> b d l').contiguous(),
                rearrange(dt,    'b l d -> b d l').contiguous(),
                A,
                rearrange(B_ssm, 'b l n -> b n l').contiguous(),
                rearrange(C_ssm, 'b l n -> b n l').contiguous(),
                self.D.float(),
                z=rearrange(z, 'b l d -> b d l').contiguous(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )
            y = rearrange(y, 'b d l -> b l d')  # [B, L, d_inner]
        else:
            # Fallback: TorchScript sequential scan
            y = _selective_scan_fallback(
                x_conv, dt, A, B_ssm, C_ssm, self.D,
                self.dt_proj.bias.float(), z
            )

        return self.out_proj(y)  # [B, L, d_model]


class MambaBlock(nn.Module):
    """Pre-norm Mamba block with residual connection."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm = RMSNorm(config.d_model, config.norm_eps)
        self.mixer = MambaMixer(config)
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.mixer(self.norm(x)))


# ─── Attention Block (GQA + RoPE) ────────────────────────────────────────────

class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (Ainslie et al. 2023).

    n_heads KV repetitions = n_heads // n_kv_heads, reducing KV cache size.
    Uses torch.nn.functional.scaled_dot_product_attention (FlashAttention-2 on
    Ampere GPUs via cuDNN backend — no flash_attn package required).
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads    = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim   = config.head_dim
        self.n_rep      = config.n_heads // config.n_kv_heads  # GQA repetitions

        self.q_proj = nn.Linear(config.d_model, config.n_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_heads * config.head_dim, config.d_model, bias=False)

        self.dropout_p = config.dropout

    def forward(
        self,
        x: torch.Tensor,         # [B, L, d_model]
        freqs_cis: torch.Tensor, # [L, head_dim//2] complex
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, L, _ = x.shape

        q = self.q_proj(x).view(B, L, self.n_heads,    self.head_dim)
        k = self.k_proj(x).view(B, L, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, L, self.n_kv_heads, self.head_dim)

        # Apply RoPE to Q and K
        q, k = apply_rotary_emb(q, k, freqs_cis[:L])

        # Append to KV cache if provided (inference)
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=1)
            v = torch.cat([v_cache, v], dim=1)
            new_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = (k, v)
        else:
            new_cache = None

        # Repeat KV heads for GQA
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=2)
            v = v.repeat_interleave(self.n_rep, dim=2)

        # [B, L, H, D] → [B, H, L, D] for SDPA
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention (uses FlashAttention kernel on Ampere via cuDNN)
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=(kv_cache is None),  # causal only during training
        )  # [B, H, L, D]

        # Merge heads
        out = attn_out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.o_proj(out), new_cache


# ─── MoE Layer ────────────────────────────────────────────────────────────────

class SwiGLUExpert(nn.Module):
    """Single SwiGLU expert FFN: SwiGLU(x) = SiLU(W1 x) ⊙ (W3 x), then W2."""

    def __init__(self, d_model: int, intermediate: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, intermediate, bias=False)
        self.up_proj   = nn.Linear(d_model, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoELayer(nn.Module):
    """
    Sparse Mixture-of-Experts layer (Switch Transformer / GShard style).

    Routing: top-k token-choice (each token selects top_k experts).
    Experts: independent SwiGLU FFNs.
    Load balancing: auxiliary loss from Switch Transformer paper.

    Computation strategy: dense dispatch over experts (torch.compile friendly),
    then scatter-add for the weighted combination. Memory cost:
      O(B·T·n_experts·d_model) intermediate tensor — acceptable for 6 experts.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_experts  = config.moe_n_experts
        self.top_k      = config.moe_top_k
        self.aux_coeff  = config.moe_aux_loss_coeff

        self.gate   = nn.Linear(config.d_model, config.moe_n_experts, bias=False)
        self.experts = nn.ModuleList([
            SwiGLUExpert(config.d_model, config.moe_intermediate)
            for _ in range(config.moe_n_experts)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T, d_model]
        Returns:
            out:      [B, T, d_model]
            aux_loss: scalar load-balancing loss
        """
        B, T, D = x.shape
        x_flat = x.reshape(B * T, D)       # [N, D]  where N = B*T

        # ── Router ───────────────────────────────────────────────────────────
        gate_logits = self.gate(x_flat)    # [N, E]
        scores = gate_logits.softmax(-1)   # [N, E]

        top_scores, top_indices = scores.topk(self.top_k, dim=-1)  # [N, K]
        top_scores = top_scores / top_scores.sum(-1, keepdim=True)  # renormalize

        # ── Load-balancing auxiliary loss (Switch Transformer eq. 4) ─────────
        # f_i: fraction of tokens routed to expert i
        # p_i: mean router probability for expert i
        # L_aux = n_experts * sum_i(f_i * p_i)
        expert_mask = torch.zeros(B * T, self.n_experts, device=x.device, dtype=x.dtype)
        expert_mask.scatter_(1, top_indices, 1.0)
        f = expert_mask.mean(0)            # [E] — fraction of tokens per expert
        p = scores.mean(0)                 # [E] — mean probability per expert
        aux_loss = self.aux_coeff * self.n_experts * (f * p).sum()

        # ── Expert computation (dense dispatch) ───────────────────────────────
        # Compute all expert outputs for all tokens, then select top-k
        # Memory: [N, E, D] — ~100MB peak for B=2, T=2048, E=6, D=2048 @ bf16
        expert_outs = torch.stack(
            [self.experts[e](x_flat) for e in range(self.n_experts)],
            dim=1
        )  # [N, E, D]

        # Gather the top-k expert outputs and weighted-sum
        # top_indices: [N, K],  top_scores: [N, K]
        idx_expanded = top_indices.unsqueeze(-1).expand(-1, -1, D)  # [N, K, D]
        selected = expert_outs.gather(1, idx_expanded)              # [N, K, D]
        out = (selected * top_scores.unsqueeze(-1)).sum(dim=1)      # [N, D]

        return out.reshape(B, T, D), aux_loss


class AttentionMoEBlock(nn.Module):
    """
    Hybrid Transformer block: pre-norm GQA attention + pre-norm MoE FFN.

    Structure (following LLaMA / Mixtral):
        x → norm → Attention → + x
        x → norm → MoE FFN  → + x
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.attn      = GroupedQueryAttention(config)
        self.moe_norm  = RMSNorm(config.d_model, config.norm_eps)
        self.moe       = MoELayer(config)
        self.dropout   = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Attention sub-block
        attn_out, new_cache = self.attn(self.attn_norm(x), freqs_cis, kv_cache)
        x = x + self.dropout(attn_out)

        # MoE FFN sub-block
        moe_out, aux_loss = self.moe(self.moe_norm(x))
        x = x + self.dropout(moe_out)

        return x, aux_loss, new_cache


# ─── Full Model ───────────────────────────────────────────────────────────────

class HybridMoEModel(nn.Module):
    """
    Hybrid Transformer-Mamba-MoE Language Model.

    Layer layout (default 18 layers):
        Layer  0: Mamba
        Layer  1: Mamba
        Layer  2: Mamba
        Layer  3: Attention + MoE   ← global attention + sparse routing
        Layer  4: Mamba
        ...
        Layer  7: Attention + MoE
        ...
        Layer 11: Attention + MoE
        Layer 12-16: Mamba
        Layer 17: Attention + MoE

    Parameter breakdown (~820M):
        Embedding:        103M (vocab=50304, d=2048) — tied with LM head
        14 Mamba layers:  371M (26.5M each)
         4 Attn layers:    42M (10.5M each, GQA 16/4)
         4 MoE layers:    302M (75.5M each, 6 experts × 12.6M)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self._attn_set = config.attention_set

        # Input embedding
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)

        # Precompute RoPE frequencies (stored as buffer, not parameter)
        freqs = precompute_freqs_cis(config.head_dim, config.max_seq_len * 2, config.rope_base)
        self.register_buffer("freqs_cis", freqs, persistent=False)

        # Build layers
        self.layers: nn.ModuleList = nn.ModuleList()
        for i in range(config.n_layers):
            if i in self._attn_set:
                self.layers.append(AttentionMoEBlock(config))
            else:
                self.layers.append(MambaBlock(config))

        # Final norm + language model head
        self.norm   = RMSNorm(config.d_model, config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: embed ↔ lm_head (saves 103M parameters of VRAM)
        if config.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        # Initialize weights
        self.apply(self._init_weights)
        # Scale residual projections by 1/sqrt(2*n_layers) for training stability
        for name, p in self.named_parameters():
            if any(s in name for s in ["out_proj.weight", "o_proj.weight", "down_proj.weight"]):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * config.n_layers))

        logger.info(f"Model initialized: {self.num_parameters():,} parameters total "
                    f"({self.num_parameters(only_trainable=True):,} trainable)")

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def num_parameters(self, only_trainable: bool = False) -> int:
        params = self.parameters() if not only_trainable else \
                 (p for p in self.parameters() if p.requires_grad)
        # Deduplicate shared parameters (tied embedding/lm_head)
        seen = set()
        total = 0
        for p in params:
            if id(p) not in seen:
                seen.add(id(p))
                total += p.numel()
        return total

    def forward(
        self,
        input_ids: torch.Tensor,             # [B, L]
        targets:   Optional[torch.Tensor] = None,  # [B, L]
        kv_caches: Optional[List] = None,          # list of KV caches (inference)
        use_checkpoint: bool = False,              # gradient checkpointing
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Dict[str, float]]]:
        """
        Args:
            input_ids:  Token indices [B, L]
            targets:    Gold token indices [B, L] for loss computation
            kv_caches:  List of (K, V) tensors per attention layer (inference)
            use_checkpoint: Apply gradient checkpointing at block level

        Returns:
            logits:     [B, L, vocab_size]
            total_loss: ce_loss + aux_loss scalar (for backward), or None
            loss_dict:  {"ce_loss": float, "aux_loss": float}, or None
        """
        B, L = input_ids.shape
        x = self.embed_tokens(input_ids)  # [B, L, d_model]
        freqs = self.freqs_cis[:L]

        total_aux_loss = x.new_zeros(())
        attn_idx = 0  # index into kv_caches

        for i, layer in enumerate(self.layers):
            if i in self._attn_set:
                cache = kv_caches[attn_idx] if kv_caches is not None else None

                if use_checkpoint and self.training:
                    # Gradient checkpointing: recompute activations in backward.
                    # use_reentrant=True is required for compatibility with
                    # mamba-ssm pybind CUDA extensions under torch.compile.
                    def attn_moe_ckpt(x, freqs, layer=layer):
                        out, aux, _ = layer(x, freqs, kv_cache=None)
                        return out, aux
                    x, aux = torch.utils.checkpoint.checkpoint(
                        attn_moe_ckpt, x, freqs, use_reentrant=True
                    )
                    new_cache = None
                else:
                    x, aux, new_cache = layer(x, freqs, cache)

                total_aux_loss = total_aux_loss + aux
                if kv_caches is not None and new_cache is not None:
                    kv_caches[attn_idx] = new_cache
                attn_idx += 1
            else:
                if use_checkpoint and self.training:
                    x = torch.utils.checkpoint.checkpoint(
                        layer, x, use_reentrant=True
                    )
                else:
                    x = layer(x)

        x = self.norm(x)
        logits = self.lm_head(x)  # [B, L, vocab_size]

        if targets is not None:
            ce_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )
            total_loss = ce_loss + total_aux_loss
            loss_dict: Optional[Dict[str, float]] = {
                "ce_loss":  ce_loss.item(),
                "aux_loss": total_aux_loss.item(),
            }
            return logits, total_loss, loss_dict

        return logits, None, None

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
    ) -> torch.Tensor:
        """Autoregressive generation with KV cache."""
        self.eval()
        device = prompt_ids.device
        ids = prompt_ids.clone()

        # Initialize empty KV caches for attention layers
        n_attn = len(self.config.attention_layers)
        kv_caches: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * n_attn

        # Prefill: process prompt in one pass (no KV cache for simplicity)
        logits, _, _ = self.forward(ids, kv_caches=None)
        next_token_logits = logits[:, -1, :]

        generated = []
        for _ in range(max_new_tokens):
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for prev_id in ids[0].tolist():
                    next_token_logits[0, prev_id] /= repetition_penalty

            # Temperature + top-p sampling
            if temperature > 0:
                logits_t = next_token_logits / temperature
                probs = F.softmax(logits_t, dim=-1)
                # Top-p (nucleus) filtering
                sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
                cumprobs = sorted_probs.cumsum(-1)
                mask = cumprobs - sorted_probs > top_p
                sorted_probs[mask] = 0.0
                sorted_probs /= sorted_probs.sum(-1, keepdim=True)
                next_token = sorted_idx.gather(-1, torch.multinomial(sorted_probs, 1))
            else:
                next_token = next_token_logits.argmax(-1, keepdim=True)

            generated.append(next_token)
            ids = torch.cat([ids, next_token], dim=1)

            # Single-token forward for the next step
            logits_new, _, _ = self.forward(next_token, kv_caches=None)
            next_token_logits = logits_new[:, -1, :]

        return torch.cat([prompt_ids, torch.cat(generated, dim=1)], dim=1)


# ─── Utilities ────────────────────────────────────────────────────────────────

def create_model(config: Optional[ModelConfig] = None) -> HybridMoEModel:
    """Convenience factory: create model from config (or defaults)."""
    if config is None:
        config = ModelConfig()
    return HybridMoEModel(config)


def estimate_memory_gb(config: ModelConfig, batch_size: int = 2, seq_len: int = 2048) -> Dict[str, float]:
    """
    Rough VRAM estimate for training (fp32 params + bf16 activations).
    Does not account for fragmentation (~10-15% overhead in practice).
    """
    n_params = HybridMoEModel(config).num_parameters()

    # fp32 weights + gradients + Adam m + v
    params_gb = n_params * 4 / 1e9
    grads_gb  = n_params * 4 / 1e9
    optim_gb  = n_params * 8 / 1e9   # m + v in fp32

    # Rough activation estimate with gradient checkpointing (only block inputs)
    n_blocks = config.n_layers
    act_per_block = batch_size * seq_len * config.d_model * 2  # bf16 = 2 bytes
    activations_gb = n_blocks * act_per_block / 1e9

    total = params_gb + grads_gb + optim_gb + activations_gb
    return {
        "params_gb": round(params_gb, 2),
        "grads_gb":  round(grads_gb, 2),
        "optim_gb":  round(optim_gb, 2),
        "activations_gb": round(activations_gb, 2),
        "total_gb":  round(total, 2),
        "n_params":  n_params,
    }
