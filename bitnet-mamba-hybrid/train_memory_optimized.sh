#!/bin/bash
# BitNet-Mamba Hybrid Training Script (Memory Optimized for RTX 4070 Ti SUPER 17GB)
#
# Key Optimizations:
# 1. Gradient checkpointing enabled (trades compute for ~40% memory savings)
# 2. Reduced batch_size from 8 to 4 (50% activation memory reduction)
# 3. Increased grad_accum from 4 to 8 (maintains effective batch size = 32)
# 4. Uses bfloat16 for memory efficiency
#
# Expected Memory Usage: ~12-14 GB (within 17GB budget)
# Training Speed: ~10-20% slower due to gradient checkpointing (worth it!)

python train_hybrid-mamba-bitnet.py \
    --d_model 1024 \
    --n_layers 12 \
    --d_state 16 \
    --d_conv 4 \
    --expand 2 \
    --gradient_checkpointing \
    --batch_size 4 \
    --grad_accum 8 \
    --max_seq_len 2048 \
    --max_tokens 4000000000 \
    --lr 3e-4 \
    --warmup_steps 2000 \
    --weight_decay 0.1 \
    --en_ratio 0.5 \
    --pt_ratio 0.5 \
    --output_dir "model" \
    --data_dir "data/tokenized" \
    --wandb \
    --wandb_project "bitnet-mamba-hybrid" \
    --wandb_run_name "d1024-seq2048-run-pt-en-optimized"
