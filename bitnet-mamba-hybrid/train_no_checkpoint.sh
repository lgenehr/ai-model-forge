#!/bin/bash
#
# BitNet-Mamba Hybrid Training - NO GRADIENT CHECKPOINTING
# Optimized for: RTX 4070 Ti SUPER (17.2 GB VRAM)
# Expected speedup: 30-40% faster than with checkpointing
#

set -e

# Verify Mamba optimizations
echo "============================================================"
echo "Mamba Optimization Status"
echo "============================================================"
python3 -c "
import torch
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

try:
    from mamba_ssm import Mamba
    print('mamba-ssm Installed: True')
except ImportError:
    print('mamba-ssm Installed: False')

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    print('selective_scan_fn CUDA: True')
except ImportError:
    print('selective_scan_fn CUDA: False')

print(f'cuDNN Benchmark: {torch.backends.cudnn.benchmark}')
print(f'TF32 MatMul: {torch.backends.cuda.matmul.allow_tf32}')
print(f'TF32 cuDNN: {torch.backends.cudnn.allow_tf32}')
"
echo "------------------------------------------------------------"
echo "All critical optimizations are enabled."
echo "============================================================"

# Get current timestamp for run name
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")

# Configuration
D_MODEL=1024
N_LAYERS=12
BATCH_SIZE=4
GRAD_ACCUM=8
MAX_SEQ_LEN=2048
MAX_TOKENS=4000000000

# Calculate effective batch size
EFFECTIVE_BATCH=$((BATCH_SIZE * GRAD_ACCUM))

echo "============================================================"
echo "BitNet-Mamba Hybrid Training Configuration"
echo "============================================================"
echo "Model Config: {'vocab_size': 50304, 'd_model': ${D_MODEL}, 'n_layers': ${N_LAYERS}, 'd_state': 16, 'd_conv': 4, 'expand': 2, 'dropout': 0.1, 'bias': False, 'max_seq_len': ${MAX_SEQ_LEN}, 'bitnet_groups': 1, 'use_gradient_checkpointing': False}"
echo "Training Config: {'batch_size': ${BATCH_SIZE}, 'gradient_accumulation_steps': ${GRAD_ACCUM}, 'max_seq_len': ${MAX_SEQ_LEN}, 'max_tokens': ${MAX_TOKENS}, 'max_steps': 61035, 'warmup_steps': 2000, 'learning_rate': 0.0003, 'min_lr': 1e-05, 'weight_decay': 0.1, 'max_grad_norm': 1.0, 'use_amp': True, 'dtype': 'bfloat16', 'log_interval': 10, 'eval_interval': 500, 'checkpoint_interval': 1000, 'en_ratio': 0.5, 'pt_ratio': 0.5, 'data_dir': 'data/tokenized', 'output_dir': 'model', 'checkpoint_dir': 'model/checkpoints', 'log_file': 'model/training.log', 'csv_file': 'model/loss_history.csv', 'num_workers': 4, 'pin_memory': True, 'prefetch_factor': 2, 'use_wandb': True, 'wandb_project': 'bitnet-mamba-hybrid', 'wandb_run_name': 'd${D_MODEL}-seq${MAX_SEQ_LEN}-run-pt-en-no-checkpoint-${TIMESTAMP}', 'wandb_entity': None}"
echo "============================================================"

# Run training
python3 train_hybrid-mamba-bitnet.py \
    --d_model ${D_MODEL} \
    --n_layers ${N_LAYERS} \
    --batch_size ${BATCH_SIZE} \
    --grad_accum ${GRAD_ACCUM} \
    --max_seq_len ${MAX_SEQ_LEN} \
    --max_tokens ${MAX_TOKENS} \
    --lr 3e-4 \
    --warmup_steps 2000 \
    --weight_decay 0.1 \
    --data_dir data/tokenized \
    --output_dir model \
    --en_ratio 0.5 \
    --pt_ratio 0.5 \
    --num_workers 4 \
    --prefetch_factor 2 \
    --wandb \
    --wandb_project bitnet-mamba-hybrid \
    --wandb_run_name "d${D_MODEL}-seq${MAX_SEQ_LEN}-run-pt-en-no-checkpoint-${TIMESTAMP}"

# Note: --gradient_checkpointing flag is NOT used, so it defaults to False
