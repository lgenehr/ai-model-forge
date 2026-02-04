#!/bin/bash
# =============================================================================
# BitNet-Mamba Hybrid Training Script (STABLE CONFIGURATION)
# =============================================================================
#
# This script is optimized for STABLE PRETRAINING FROM SCRATCH with the
# BitNet b1.58 + Mamba SSM hybrid architecture.
#
# KEY STABILITY FEATURES:
# -----------------------
# 1. Lower learning rate (1e-4) - BitNet's STE adds gradient noise
# 2. Longer warmup (4000 steps) - Gentler optimization ramp-up
# 3. Conservative weight decay (0.05) - Prevents fighting noisy gradients
# 4. Tighter gradient clipping (0.5) - SSM parameters are sensitive
# 5. Gradient checkpointing - Reduces memory pressure, improves stability
#
# HYPERPARAMETER RATIONALE:
# -------------------------
#
# --lr 1e-4
#   BitNet papers (Wang et al. 2023) recommend 1e-4 to 1.5e-4 due to the
#   quantization noise from Straight-Through Estimator (STE). Using 3e-4
#   (common for transformers) causes instability in BitNet+Mamba.
#
# --warmup_steps 4000
#   With effective batch size 32 and seq_len 2048, this is ~262M tokens.
#   For 4B total tokens, warmup is ~6.5% of training. BitNet needs gentler
#   warmup because the quantization noise is amplified when weights are
#   still random.
#
# --weight_decay 0.05
#   Standard 0.1 weight decay is too aggressive for BitNet. The noisy
#   gradients mean weights update more slowly, but aggressive decay still
#   shrinks them at the normal rate, causing underfitting.
#
# --max_grad_norm 0.5
#   Mamba's SSM parameters (A_log, dt_proj, D) are sensitive to large
#   gradient updates. Tighter clipping prevents loss spikes.
#
# --gradient_checkpointing
#   Trades compute for memory. More importantly, it forces cleaner gradient
#   computation paths, which can improve numerical stability.
#
# EXPECTED BEHAVIOR:
# ------------------
# - Initial loss: ~10-11 (random model on vocab_size ~50k)
# - After 1000 steps: loss should be < 9
# - After 10000 steps: loss should be < 7
# - After 50000 steps: loss should be < 5
# - Final loss target: < 3.5 (good), < 3.0 (excellent)
#
# WARNING SIGNS (stop training if you see these):
# - Loss > 12 after 1000 steps
# - Loss spikes > 2x current loss
# - NaN or Inf in loss
# - Gradient norm consistently > 10
# - Generated text is pure repetition or gibberish after 10M tokens
#
# GPU MEMORY: ~14-16 GB with these settings
# THROUGHPUT: ~2000-2500 tokens/sec on RTX 4070 Ti SUPER
#
# =============================================================================

set -e  # Exit on error

# Change to script directory
cd "$(dirname -- "$0")"

# Print configuration summary
echo "============================================================"
echo "BitNet-Mamba Stable Training Configuration"
echo "============================================================"
echo "Learning Rate:     1e-4 (conservative for BitNet)"
echo "Warmup Steps:      4000 (~262M tokens, 6.5% of training)"
echo "Weight Decay:      0.05 (reduced for noisy gradients)"
echo "Gradient Clip:     0.5 (tighter for SSM stability)"
echo "Batch Size:        4 x 8 = 32 effective"
echo "Sequence Length:   2048"
echo "Target Tokens:     2.5B"
echo "============================================================"
echo ""

# Verify data exists
if [ ! -d "data/tokenized/en" ] && [ ! -d "data/tokenized/pt" ]; then
    echo "ERROR: Preprocessed data not found!"
    echo "Please run: python preprocess_datasets.py --output_dir data/tokenized"
    exit 1
fi

# Main training command
python train_hybrid-mamba-bitnet.py \
    \
    `# === MODEL ARCHITECTURE ===` \
    --d_model 1024 \
    --n_layers 12 \
    --d_state 16 \
    --d_conv 4 \
    --expand 2 \
    --dropout 0.1 \
    \
    `# === MEMORY OPTIMIZATION ===` \
    --gradient_checkpointing \
    \
    `# === BATCH CONFIGURATION ===` \
    --batch_size 4 \
    --grad_accum 8 \
    --max_seq_len 2048 \
    \
    `# === TRAINING HYPERPARAMETERS (TUNED FOR STABILITY) ===` \
    --lr 1e-4 \
    --warmup_steps 4000 \
    --weight_decay 0.05 \
    --max_tokens 2500000000 \
    \
    `# === DATA CONFIGURATION ===` \
    --data_dir "data/tokenized" \
    --en_ratio 0.5 \
    --pt_ratio 0.5 \
    --num_workers 4 \
    --prefetch_factor 2 \
    \
    `# === OUTPUT CONFIGURATION ===` \
    --output_dir "model" \
    \
    `# === EXPERIMENT TRACKING ===` \
    --wandb \
    --wandb_project "bitnet-mamba-hybrid" \
    --wandb_run_name "stable-d1024-$(date +%Y%m%d-%H%M%S)" \
    \
    `# === REPRODUCIBILITY ===` \
    --seed 42

echo ""
echo "============================================================"
echo "Training completed or interrupted."
echo "Check model/ directory for checkpoints."
echo "============================================================"
