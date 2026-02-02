# Memory Optimization Guide for BitNet-Mamba Hybrid Training

## Problem
Training the BitNet-Mamba hybrid model (128M parameters) with the original configuration caused CUDA out of memory errors on RTX 4070 Ti SUPER (17.2 GB VRAM).

## Root Cause Analysis

### Original Configuration Memory Usage
```
Model Parameters: 128M × 4 bytes = 512 MB
Optimizer States (AdamW): 128M × 8 bytes = 1 GB
Gradients: 128M × 4 bytes = 512 MB
Activations (batch=8, seq=2048, d_model=1024, layers=12): ~12-15 GB
---
Total: ~17-18 GB (exceeds 17.2 GB available)
```

The main memory bottleneck is **activation memory** during forward/backward passes.

## Solutions Implemented

### 1. Gradient Checkpointing (40% memory reduction)
- **What**: Recomputes activations during backward pass instead of storing them
- **Trade-off**: ~10-20% slower training, but saves ~40% activation memory
- **How to enable**: Add `--gradient_checkpointing` flag
- **Implementation**: Uses `torch.utils.checkpoint` to wrap each Mamba layer

### 2. Reduced Batch Size (50% memory reduction)
- **Original**: batch_size=8
- **Optimized**: batch_size=4
- **Impact**: Halves activation memory requirements

### 3. Increased Gradient Accumulation (maintains effective batch size)
- **Original**: grad_accum=4 (effective batch = 32)
- **Optimized**: grad_accum=8 (effective batch = 32)
- **Impact**: No change to training dynamics, same convergence behavior

### 4. Optional: Reduced Sequence Length (50% additional memory reduction)
- **Original**: max_seq_len=2048
- **Ultra-safe**: max_seq_len=1024
- **Impact**: Less context per sample, but fits safely in memory

## Training Scripts

### Recommended: `train_memory_optimized.sh`
```bash
./bitnet-mamba-hybrid/train_memory_optimized.sh
```
- **Expected memory**: ~12-14 GB
- **Speed**: ~10-20% slower than original (due to checkpointing)
- **Reliability**: Should work on 16GB+ GPUs
- **Configuration**:
  - batch_size=4, grad_accum=8
  - max_seq_len=2048
  - gradient_checkpointing=True

### Fallback: `train_ultra_memory_safe.sh`
```bash
./bitnet-mamba-hybrid/train_ultra_memory_safe.sh
```
- **Expected memory**: ~8-10 GB
- **Speed**: ~30% slower than original
- **Reliability**: Works on 12GB+ GPUs
- **Configuration**:
  - batch_size=4, grad_accum=16
  - max_seq_len=1024
  - gradient_checkpointing=True

## Memory Usage Comparison

| Configuration | Batch Size | Seq Len | Grad Ckpt | Memory | Speed |
|---------------|------------|---------|-----------|--------|-------|
| Original      | 8          | 2048    | No        | ~18 GB | 1.0x  |
| Optimized     | 4          | 2048    | Yes       | ~13 GB | 0.85x |
| Ultra Safe    | 4          | 1024    | Yes       | ~9 GB  | 0.70x |

## How Gradient Checkpointing Works

```python
# Without checkpointing (stores all activations)
for layer in self.layers:
    x = layer(x)  # Activations kept in memory for backward

# With checkpointing (trades compute for memory)
for layer in self.layers:
    x = checkpoint(layer, x, use_reentrant=False)
    # Activations discarded, recomputed during backward
```

## Monitoring Memory Usage

To monitor GPU memory during training:
```bash
watch -n 1 nvidia-smi
```

Or add this to your training script:
```python
torch.cuda.max_memory_allocated() / 1e9  # GB
```

## Additional Tips

1. **Empty CUDA cache before training**:
   ```bash
   python -c "import torch; torch.cuda.empty_cache()"
   ```

2. **Close all other GPU applications** (browsers, electron apps, etc.)

3. **Use `nvidia-smi` to check for lingering processes**:
   ```bash
   nvidia-smi
   ```

4. **Monitor memory during first few steps**:
   - If memory usage is stable after 10 steps, you're safe
   - If it keeps growing, reduce batch_size or seq_len further

## Fixed Command Line Bug

The original command had a **missing backslash** after `--data_dir`:
```bash
# WRONG (causes shell parsing errors)
--data_dir "bitnet-mamba-hybrid/data/tokenized"
--wandb \

# CORRECT
--data_dir "bitnet-mamba-hybrid/data/tokenized" \
--wandb \
```

This is fixed in both training scripts.

## Performance Expectations

With gradient checkpointing enabled:
- **Memory**: Fits comfortably in 17 GB
- **Speed**: 10-20% slower than without checkpointing
- **Convergence**: Identical to original (effective batch size unchanged)
- **Throughput**: ~1000-1500 tokens/sec on RTX 4070 Ti SUPER

## References

- [PyTorch Gradient Checkpointing](https://pytorch.org/docs/stable/checkpoint.html)
- [Reducing Memory Usage in Neural Networks](https://huggingface.co/docs/transformers/perf_train_gpu_one)
- [BitNet Paper](https://arxiv.org/abs/2402.17764)
- [Mamba Paper](https://arxiv.org/abs/2312.00752)
