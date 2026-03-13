#!/bin/bash
# BitNet-Mamba Hybrid Training Script (HIGH THROUGHPUT - Maximum GPU Utilization)
#
# Use this script when you have spare GPU memory (e.g., only using 70% of VRAM)
# and want to maximize tokens/second throughput.
#
# Key Optimizations:
# 1. Larger batch_size (8) to better utilize GPU memory
# 2. Increased num_workers (8) and prefetch_factor (4) for faster data loading
# 3. NO gradient checkpointing (prioritizes speed over memory)
#
# NOTE: torch.compile() is NOT used because Mamba's selective_scan CUDA kernel
# is incompatible with PyTorch Dynamo, causing graph breaks and overhead.
#
# Expected Memory Usage: ~85-90% of GPU VRAM
# Expected Throughput: ~2500-3000 tokens/sec on RTX 4070 Ti SUPER
#
# If you get OOM errors, try reducing batch_size to 6

cd "$(dirname -- "$0")/.."

python train_hybrid-mamba-bitnet.py \
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
    --data_dir "/home/lgene/meu_modelo_temp/ai-model-forge/datasets/tokenized" \
    --wandb \
    --wandb_project "bitnet-mamba-hybrid" \
    --wandb_run_name "d1024-high-throughput-$(date +%Y%m%d-%H%M%S)"
