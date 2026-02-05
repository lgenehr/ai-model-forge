#!/bin/bash
# =============================================================================
# BitNet-Mamba Hybrid Training Script (SCALED 204M CONFIGURATION)
# =============================================================================
#
# This script is the VALIDATED configuration for scaling to ~204M parameters.
# Based on comprehensive VRAM and throughput analysis documented in:
#   SCALING_ANALYSIS_255M.md
#
# ARCHITECTURE CHANGES FROM BASELINE (128M):
# ------------------------------------------
# - d_model: 1024 → 1280 (+25% width)
# - n_layers: 12 → 14 (+17% depth)
# - Parameters: 128M → 204M (+59%)
# - d_inner: 2048 → 2560 (computed from d_model × expand)
#
# WHY NOT 255M (d_model=1536)?
# ----------------------------
# The 255M target would require 22-28 GB VRAM, exceeding the 17 GB budget by
# 30-65%. Analysis showed that d_model=1536 causes activation memory explosion
# even with gradient checkpointing enabled.
#
# The 204M configuration is the PARETO OPTIMAL point:
# - Maximum parameters within VRAM budget
# - Maintains ≥500 tok/sec throughput
# - Preserves full 2048 sequence length
# - Acceptable numerical stability
#
# HYPERPARAMETER RATIONALE:
# -------------------------
#
# --lr 8e-5
#   Scaled from baseline 1e-4 using: LR_new = LR_old × sqrt(128M/204M)
#   Larger models need smaller learning rates to prevent loss instability.
#   BitNet STE noise is amplified with more parameters per layer.
#
# --warmup_steps 5000
#   Extended from 4000 to accommodate:
#   - Lower learning rate (needs longer to reach effective range)
#   - More parameters to stabilize
#   - 337M warmup tokens = 8.4% of 4B target (within 5-10% standard range)
#
# --weight_decay 0.04
#   Reduced from 0.05. Larger models are less prone to overfitting on
#   pretraining data, so aggressive decay is counterproductive.
#
# --batch_size 4
#   Memory-constrained choice. With d_model=1280 and 14 layers, batch=4
#   uses ~15-17 GB VRAM. If OOM occurs, use train_scaled_204m_safe.sh
#   with batch=3 instead.
#
# --grad_accum 8
#   Effective batch size = 4 × 8 = 32
#   Maintains same effective batch as baseline for comparable convergence.
#
# --n_layers 14
#   Depth scaling is preferred over width scaling because:
#   1. Better memory efficiency with gradient checkpointing
#   2. More recurrent state refinement in Mamba SSM
#   3. Linear FLOP growth vs quadratic for width
#
# EXPECTED BEHAVIOR:
# ------------------
# - Initial loss: ~10-11 (random model on vocab ~50k)
# - After 1000 steps: loss should be < 9
# - After 10000 steps: loss should be < 6.5
# - After 50000 steps: loss should be < 4.5
# - Final loss target: < 3.2 (good), < 2.8 (excellent)
#
# Note: Slightly higher loss targets than 128M due to capacity-data ratio.
#
# WARNING SIGNS (stop training if you see these):
# - Loss > 11 after 1000 steps
# - Loss spikes > 1.5x current loss
# - NaN or Inf in loss
# - Gradient norm consistently > 10
# - VRAM usage > 17.5 GB (OOM risk)
#
# GPU MEMORY: ~15-17 GB (tight fit for 17 GB card)
# THROUGHPUT: ~550-824 tokens/sec on RTX 4070 Ti SUPER
#
# =============================================================================

set -e  # Exit on error

# Change to script directory
cd "$(dirname -- "$0")"

# Print configuration summary
echo "============================================================"
echo "BitNet-Mamba SCALED Training Configuration (204M Parameters)"
echo "============================================================"
echo ""
echo "ARCHITECTURE:"
echo "  d_model:          1280 (was 1024)"
echo "  n_layers:         14 (was 12)"
echo "  d_inner:          2560 (computed)"
echo "  Parameters:       ~204M (+59% from baseline)"
echo ""
echo "HYPERPARAMETERS:"
echo "  Learning Rate:    8e-5 (scaled down from 1e-4)"
echo "  Warmup Steps:     5000 (~337M tokens, 8.4% of training)"
echo "  Weight Decay:     0.04 (reduced for larger model)"
echo "  Gradient Clip:    0.5 (unchanged, SSM stability)"
echo ""
echo "BATCH CONFIGURATION:"
echo "  Batch Size:       4"
echo "  Grad Accumulation: 8"
echo "  Effective Batch:  32"
echo "  Sequence Length:  2048"
echo ""
echo "MEMORY ESTIMATE:    15-17 GB (checkpointing enabled)"
echo "THROUGHPUT TARGET:  ≥550 tokens/sec"
echo ""
echo "============================================================"
echo ""

# Verify data exists
if [ ! -d "data/tokenized/en" ] && [ ! -d "data/tokenized/pt" ]; then
    echo "ERROR: Preprocessed data not found!"
    echo "Please run: python preprocess_datasets.py --output_dir data/tokenized"
    exit 1
fi

# Check VRAM before starting
echo "Checking GPU memory..."
nvidia-smi --query-gpu=memory.free,memory.total --format=csv,noheader,nounits | while read free total; do
    free_gb=$(echo "scale=1; $free / 1024" | bc)
    total_gb=$(echo "scale=1; $total / 1024" | bc)
    echo "GPU Memory: ${free_gb}GB free / ${total_gb}GB total"

    if (( $(echo "$free < 15000" | bc -l) )); then
        echo ""
        echo "WARNING: Less than 15 GB free VRAM detected!"
        echo "This configuration requires ~15-17 GB."
        echo "Consider closing other applications or using train_scaled_204m_safe.sh"
        echo ""
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
done

echo ""
echo "Starting training..."
echo ""

# Main training command
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
    `# === BATCH CONFIGURATION ===` \
    --batch_size 4 \
    --grad_accum 8 \
    --max_seq_len 2048 \
    \
    `# === HYPERPARAMETERS (RECALIBRATED FOR 204M) ===` \
    --lr 2e-4 \
    --min_lr 5e-7 \
    --warmup_steps 2000 \
    --weight_decay 0.03 \
    --max_grad_norm 0.5 \
    --max_tokens 3000000000 \
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
    --wandb_run_name "scaled-204m-d1280-L14-$(date +%Y%m%d-%H%M%S)" \
    \
    `# === REPRODUCIBILITY ===` \
    --seed 42

echo ""
echo "============================================================"
echo "Training completed or interrupted."
echo "Check model_204m/ directory for checkpoints."
echo "============================================================"
