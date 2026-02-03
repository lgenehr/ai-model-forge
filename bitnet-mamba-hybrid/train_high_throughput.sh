#!/bin/bash
# BitNet-Mamba Hybrid Training Script (HIGH THROUGHPUT - Maximum GPU Utilization)
#
# Use this script when you have spare GPU memory (e.g., only using 70% of VRAM)
# and want to maximize tokens/second throughput.
#
# Key Optimizations:
# 1. torch.compile() with default mode (JIT compilation - reduce-overhead has issues with Mamba)
# 2. Optimized batch_size (8) - balanced for 16-17GB GPUs with torch.compile()
# 3. Increased num_workers (8) and prefetch_factor (4) for faster data loading
# 4. NO gradient checkpointing (prioritizes speed over memory)
#
# Expected Memory Usage: ~85-90% of GPU VRAM
# Expected Throughput: +20-40% tokens/second compared to memory_optimized config
#
# Requirements:
# - PyTorch 2.0+ for torch.compile()
# - GPU with sufficient VRAM (16GB+ recommended)
#
# If you get OOM errors, try reducing batch_size to 6

cd "$(dirname -- "$0")"

python train_hybrid-mamba-bitnet.py \
    --compile \
    --compile_mode "default" \
    --d_model 1024 \
    --n_layers 12 \
    --d_state 16 \
    --d_conv 4 \
    --expand 2 \
    --batch_size 8 \
    --grad_accum 4 \
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
