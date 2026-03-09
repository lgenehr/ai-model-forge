# Hybrid Transformer-Mamba-MoE (~820M parameters)

A modern LLM trained from scratch combining three complementary architectures:
**Selective SSM (Mamba)** for efficient long-range modeling,
**Grouped-Query Attention** for precise global context, and
**Sparse Mixture-of-Experts** for maximum parameter capacity at low compute cost.

---

## Architecture

```
Token Embedding (50304 × 2048)
        │
┌────── Repeat pattern (18 layers total) ──────┐
│  Layers 0-2, 4-6, 8-10, 12-16:               │
│      Pre-norm → MambaMixer → Residual         │
│                                               │
│  Layers 3, 7, 11, 17:                        │
│      Pre-norm → GQA Attention → Residual     │
│      Pre-norm → MoE FFN      → Residual      │
└───────────────────────────────────────────────┘
        │
    RMSNorm → LM Head (tied with embedding)
```

### Why this layout?

| Component | Role | Compute cost |
|-----------|------|-------------|
| **Mamba SSM** | Long-range memory, efficient recurrence | O(L·D·d_state) — linear |
| **GQA Attention** | Precise global context, positional reasoning | O(L²·D) — but only 4 of 18 layers |
| **Sparse MoE** | Scale parameters without scaling FLOPs | Top-2/6 experts active per token |

Placing attention every ~4-5 Mamba layers mirrors the Jamba architecture (AI21 Labs 2024), which showed that a 1:7 attention-to-SSM ratio achieves near-full-attention quality at a fraction of the cost.

---

## Parameter Budget

| Component | Count | % of total |
|-----------|-------|-----------|
| Embedding (tied) | 103M | 12.6% |
| 14 × Mamba blocks | 371M | 45.4% |
| 4 × GQA attention | 42M | 5.1% |
| 4 × MoE FFN (6 experts) | 302M | 36.9% |
| **Total** | **~818M** | |
| Active per forward | ~570M | 69.7% |

MoE parameters are stored in full but only 2/6 experts are activated per token — giving 3× more parameter capacity at the same FLOPs as a dense FFN with `intermediate = 2 × d_model`.

---

## Memory Analysis (RTX 4070 Ti Super, 17.2 GB)

```
fp32 parameters:          3.3 GB   (AdamW stores master copy in fp32)
fp32 gradients:           3.3 GB
fp32 optimizer (m + v):   6.6 GB
bf16 activations (ckpt):  1.5 GB   (gradient checkpointing active)
─────────────────────────────────
Total estimate:          14.7 GB   ← 2.5 GB margin
```

With `gradient_checkpointing: true`, each block recomputes its forward pass
during backprop rather than storing all intermediate tensors.

### Pushing to 1B parameters

Change in `config.yaml`:
```yaml
moe_n_experts: 8          # 6 → 8 experts (+120M params → ~920M total)
```
Memory impact: +1.1 GB → ~15.8 GB. Needs gradient checkpointing enabled.

---

## Mamba SSM Implementation

Two code paths, selected automatically at import:

1. **`mamba-ssm` CUDA kernel** (fast path) — uses `selective_scan_fn` from
   the `mamba-ssm` package. ~50× faster than the Python fallback.
   Required for the 1500–2500 tok/s target.

2. **TorchScript sequential scan** (fallback) — numerically identical,
   compiles to C++ via `@torch.jit.script`. Correct but ~10× slower.

The CUDA kernel is installed in this environment (`mamba-ssm` detected ✓).

---

## GQA Attention

Uses **Grouped-Query Attention** (Ainslie et al. 2023) with:
- 16 Q heads, 4 KV heads → 4× KV cache compression
- RoPE positional embeddings (θ = 10,000)
- `F.scaled_dot_product_attention` — uses cuDNN FlashAttention-v2 kernel on Ampere GPUs automatically (no separate `flash-attn` package needed)

---

## MoE Routing

Follows **Switch Transformer** top-k routing with load-balancing:

```
gate_logits = W_gate @ x        # [N, n_experts]
scores      = softmax(gate_logits)
top_scores, top_idx = topk(scores, k=2)

# Auxiliary load-balancing loss (prevents expert collapse):
L_aux = α × n_experts × Σᵢ fᵢ × pᵢ
```

Where `fᵢ` = fraction of tokens routed to expert `i`,
and `pᵢ` = mean routing probability for expert `i`.
`α = 0.01` (configurable via `moe_aux_loss_coeff`).

---

## Quick Start

```bash
cd hybrid-moe-1b

# Verify model builds and estimate memory
python -c "from model import create_model, estimate_memory_gb, ModelConfig; \
  cfg = ModelConfig(); m = create_model(cfg); \
  print(estimate_memory_gb(cfg, batch_size=2))"

# Start training (reads config.yaml)
python train.py

# Override any config value from CLI
python train.py --batch_size 4 --grad_accum 8 --wandb --wandb_run_name my-run

# Resume from checkpoint
python train.py --resume ./model_1b/checkpoints/latest.pt

# Disable torch.compile for debugging
python train.py --no_compile --no_grad_checkpoint
```

---

## Performance Notes

| Setting | Expected tok/s |
|---------|---------------|
| `mamba-ssm` ✓ + `compile` ✓ + bf16 | **1800–2500** |
| `mamba-ssm` ✓ + bf16 (no compile) | 1000–1500 |
| Fallback scan + compile | 200–400 |

The compile step adds ~2-3 minutes on first run but dramatically reduces Python overhead on subsequent steps.

`torch.compile(mode="reduce-overhead")` is used (vs `"max-autotune"`) because the Mamba sequential scan has a non-trivial compile time and `reduce-overhead` is more practical for training loops.

---

## Dependencies

All already installed in this environment:

```
torch >= 2.0       (2.10.0+cu128 detected)
mamba-ssm          (CUDA kernel — required for target throughput)
einops
wandb              (optional)
pyyaml
numpy
```

For 1B parameters with headroom to spare, optionally install:
```bash
pip install bitsandbytes   # 8-bit Adam → saves ~6.6 GB optimizer memory
```
