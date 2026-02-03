#!/bin/bash
# BitNet-Mamba Hybrid Training Script (HIGH THROUGHPUT - Maximum GPU Utilization)
#
# Use this script when you have spare GPU memory (e.g., only using 70% of VRAM)
# and want to maximize tokens/second throughput.
#
# Key Optimizations:
# 1. torch.compile() with reduce-overhead mode (JIT compilation for faster execution)
# 2. Larger batch_size (12) to fill GPU memory up to ~90-95%
# 3. Reduced grad_accum (3) to maintain reasonable effective batch size
# 4. Increased num_workers (8) and prefetch_factor (4) for faster data loading
# 5. NO gradient checkpointing (prioritizes speed over memory)
#
# Expected Memory Usage: ~90-95% of GPU VRAM
# Expected Throughput: +30-50% tokens/second compared to memory_optimized config
#
# Requirements:
# - PyTorch 2.0+ for torch.compile()
# - GPU with sufficient VRAM (16GB+ recommended)
#
# If you get OOM errors, try reducing batch_size to 10 or 8

cd "$(dirname "$0")"

python train_hybrid-mamba-bitnet.py \
    --high_throughput \
    --compile \
    --compile_mode "reduce-overhead" \
    --d_model 1024 \
    --n_layers 12 \
    --d_state 16 \
    --d_conv 4 \
    --expand 2 \
    --batch_size 12 \
    --grad_accum 3 \
    --max_seq_len 2048 \
    --max_tokens 4000000000 \
    --lr 3e-4 \
    --warmup_steps 2000 \
    --weight_decay 0.1 \
    --num_workers 8 \
    --prefetch_factor 4 \
    --en_ratio 0.5 \
    --pt_ratio 0.5 \
    --output_dir "model" \
    --data_dir "data/tokenized" \
    --wandb \
    --wandb_project "bitnet-mamba-hybrid" \
    --wandb_run_name "d1024-high-throughput-$(date +%Y%m%d-%H%M%S)"
