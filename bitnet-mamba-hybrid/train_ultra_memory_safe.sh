#!/bin/bash
# BitNet-Mamba Hybrid Training Script (Ultra Memory Safe for RTX 4070 Ti SUPER 17GB)
#
# Use this if train_memory_optimized.sh still runs out of memory.
#
# Additional Optimizations:
# 1. Reduced max_seq_len from 2048 to 1024 (another 50% memory reduction)
# 2. Batch size 4 with grad_accum 16 (effective batch size still = 64)
# 3. Gradient checkpointing enabled
#
# Expected Memory Usage: ~8-10 GB (very safe for 17GB GPU)
# Training Speed: ~30% slower, but guaranteed to fit in memory

python bitnet-mamba-hybrid/train_hybrid-mamba-bitnet.py \
    --d_model 1024 \
    --n_layers 12 \
    --d_state 16 \
    --d_conv 4 \
    --expand 2 \
    --gradient_checkpointing \
    --batch_size 4 \
    --grad_accum 16 \
    --max_seq_len 1024 \
    --max_tokens 4000000000 \
    --lr 3e-4 \
    --warmup_steps 2000 \
    --weight_decay 0.1 \
    --en_ratio 0.5 \
    --pt_ratio 0.5 \
    --output_dir "bitnet-mamba-hybrid/model" \
    --data_dir "bitnet-mamba-hybrid/data/tokenized" \
    --wandb \
    --wandb_api_key "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" \
    --wandb_project "bitnet-mamba-hybrid" \
    --wandb_run_name "d1024-seq1024-run-pt-en-ultra-safe"
