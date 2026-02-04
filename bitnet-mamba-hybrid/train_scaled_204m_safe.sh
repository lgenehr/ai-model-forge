#!/bin/bash
# =============================================================================
# BitNet-Mamba Hybrid Training Script (SCALED 204M - MEMORY SAFE)
# =============================================================================
#
# FALLBACK CONFIGURATION for when train_scaled_204m.sh causes OOM.
#
# DIFFERENCES FROM PRIMARY SCRIPT:
# --------------------------------
# - batch_size: 4 → 3 (-25% memory per step)
# - grad_accum: 8 → 11 (maintains ~33 effective batch)
# - Expected VRAM: 15-17 GB → 13-15 GB
# - Throughput: ~550-824 → ~410-620 tok/sec
#
# USE THIS SCRIPT IF:
# - You see "CUDA out of memory" errors with train_scaled_204m.sh
# - Your GPU has exactly 16 GB (not 17+ GB)
# - You're running other VRAM-consuming processes
#
# HYPERPARAMETERS:
# ----------------
# Same as primary script, only batch configuration changed.
#
# GPU MEMORY: ~13-15 GB (conservative estimate)
# THROUGHPUT: ~410-620 tokens/sec (reduced due to smaller batch)
#
# =============================================================================

set -e  # Exit on error

# Change to script directory
cd "$(dirname -- "$0")"

# Print configuration summary
echo "============================================================"
echo "BitNet-Mamba SCALED Training (204M) - MEMORY SAFE MODE"
echo "============================================================"
echo ""
echo "This is the MEMORY-SAFE fallback configuration."
echo "Use train_scaled_204m.sh for higher throughput if VRAM allows."
echo ""
echo "ARCHITECTURE:"
echo "  d_model:          1280"
echo "  n_layers:         14"
echo "  Parameters:       ~204M"
echo ""
echo "BATCH CONFIGURATION (MEMORY SAFE):"
echo "  Batch Size:       3 (reduced from 4)"
echo "  Grad Accumulation: 11 (increased to compensate)"
echo "  Effective Batch:  33"
echo "  Sequence Length:  2048"
echo ""
echo "MEMORY ESTIMATE:    13-15 GB"
echo "THROUGHPUT TARGET:  ≥410 tokens/sec"
echo ""
echo "============================================================"
echo ""

# Verify data exists
if [ ! -d "data/tokenized/en" ] && [ ! -d "data/tokenized/pt" ]; then
    echo "ERROR: Preprocessed data not found!"
    echo "Please run: python preprocess_datasets.py --output_dir data/tokenized"
    exit 1
fi

echo "Starting training (memory-safe mode)..."
echo ""

# Main training command (memory-safe configuration)
python train_hybrid-mamba-bitnet.py \
    \
    `# === MODEL ARCHITECTURE (SCALED TO 204M) ===` \
    --d_model 1280 \
    --n_layers 14 \
    --d_state 16 \
    --d_conv 4 \
    --expand 2 \
    --dropout 0.1 \
    \
    `# === MEMORY OPTIMIZATION (REQUIRED) ===` \
    --gradient_checkpointing \
    \
    `# === BATCH CONFIGURATION (MEMORY SAFE) ===` \
    --batch_size 3 \
    --grad_accum 11 \
    --max_seq_len 2048 \
    \
    `# === HYPERPARAMETERS (SAME AS PRIMARY) ===` \
    --lr 8e-5 \
    --min_lr 5e-7 \
    --warmup_steps 5000 \
    --weight_decay 0.04 \
    --max_grad_norm 0.5 \
    --max_tokens 4000000000 \
    \
    `# === DATA CONFIGURATION ===` \
    --data_dir "data/tokenized" \
    --en_ratio 0.5 \
    --pt_ratio 0.5 \
    --num_workers 4 \
    --prefetch_factor 2 \
    \
    `# === OUTPUT CONFIGURATION ===` \
    --output_dir "model_204m" \
    \
    `# === EXPERIMENT TRACKING ===` \
    --wandb \
    --wandb_project "bitnet-mamba-hybrid" \
    --wandb_run_name "scaled-204m-safe-$(date +%Y%m%d-%H%M%S)" \
    \
    `# === REPRODUCIBILITY ===` \
    --seed 42

echo ""
echo "============================================================"
echo "Training completed or interrupted."
echo "Check model_204m/ directory for checkpoints."
echo "============================================================"
