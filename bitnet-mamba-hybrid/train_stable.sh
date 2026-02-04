#!/bin/bash
# Stable BitNet-Mamba Hybrid Pretraining Script (Recommended Defaults)
#
# Goals:
# - Stable pretraining from scratch with BitNet + Mamba
# - Conservative learning rate and warmup
# - Gradient clipping and checkpointing enabled
# - Balanced EN/PT ratio
#
# Notes:
# - Use WANDB_API_KEY env var if you enable --wandb
# - Adjust batch_size/grad_accum for your GPU memory budget

set -euo pipefail

cd "$(dirname -- "$0")"

python train_hybrid-mamba-bitnet.py \
    --d_model 768 \
    --n_layers 12 \
    --d_state 16 \
    --d_conv 4 \
    --expand 2 \
    --dropout 0.1 \
    --gradient_checkpointing \
    --batch_size 6 \
    --grad_accum 8 \
    --max_seq_len 2048 \
    --max_tokens 4000000000 \
    --lr 2e-4 \
    --min_lr 2e-5 \
    --warmup_steps 4000 \
    --weight_decay 0.1 \
    --max_grad_norm 1.0 \
    --en_ratio 0.5 \
    --pt_ratio 0.5 \
    --num_workers 6 \
    --prefetch_factor 2 \
    --output_dir "model" \
    --data_dir "data/tokenized"
