#!/bin/bash
# =============================================================================
# BitNet-Mamba RECOVERY PHASE 3 - Per-Group Learning Rates for BitLinear
# =============================================================================
#
# ROOT CAUSE IDENTIFIED:
# After 30k steps and 2B tokens, BitLinear weight parameters (in_proj,
# out_proj across all 14 layers) have NOT moved from random initialization.
# The STE (Straight-Through Estimator) for ternary quantization adds
# gradient noise that the base LR cannot overcome.
#
# FIX: Per-group learning rates where BitLinear gets 5x higher LR.
#
# MUDANÇAS DA FASE 2 → FASE 3:
# --lr 3e-4 → 5e-4               (higher base LR)
# --bitlinear_lr_scale N/A → 5.0  (NEW: BitLinear effective LR = 2.5e-3)
# --dropout 0.1 → 0.0             (model is underfitting, not overfitting)
# --weight_decay 0.005 → 0.005    (minimal regularization)
# --max_grad_norm 2.5 → 3.0       (relaxed for STE noise)
# --warmup_steps 1000 → 500       (short, model already trained)
# --weights_only (reset optimizer state for new param groups)
#
# JUSTIFICATIVA:
# - BitLinear weights frozen at init (std ~0.02, ternary ~31/37/31%)
# - Only embedding and SSM parameters were learning
# - STE gradient noise requires higher LR for BitLinear layers
# - Per-group LR is the standard fix for this problem
#
# =============================================================================

set -e

cd "$(dirname -- "$0")/.."

echo "============================================================"
echo "BitNet-Mamba RECOVERY PHASE 3 - Per-Group Learning Rates"
echo "============================================================"
echo ""
echo "ROOT CAUSE:"
echo "  BitLinear weights NOT learning (frozen at random init)"
echo "  STE gradient noise too high for base LR to overcome"
echo ""
echo "FIX: Per-group learning rates"
echo "  Base LR (SSM/embedding):  5e-4"
echo "  BitLinear LR scale:       5.0x"
echo "  Effective BitLinear LR:   2.5e-3"
echo ""
echo "OTHER CHANGES:"
echo "  Dropout:          0.1 → 0.0 (underfitting, not overfitting)"
echo "  Gradient Clip:    2.5 → 3.0 (relaxed for STE noise)"
echo "  Warmup Steps:     1000 → 500 (short, already trained)"
echo "  Optimizer:        RESET (weights_only, new param groups)"
echo ""
echo "TARGETS:"
echo "  500 steps:        Loss < 4.5"
echo "  1000 steps:       Loss < 4.0"
echo "  BitLinear grads:  Should increase vs. before"
echo ""
echo "============================================================"
echo ""

# Backup current state
echo "Creating backup..."
BACKUP_DIR="model_204m/checkpoints/backup_phase3_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

LATEST_CKPT=$(ls -t model_204m/checkpoints/checkpoint_*.pt 2>/dev/null | head -1)
if [ -n "$LATEST_CKPT" ]; then
    cp "$LATEST_CKPT" "$BACKUP_DIR/"
    echo "Backed up $LATEST_CKPT to $BACKUP_DIR/"
fi

echo ""
echo "Starting Phase 3 training with per-group learning rates..."
echo ""

python train_hybrid-mamba-bitnet.py \
    \
    `# === MODEL ARCHITECTURE (UNCHANGED) ===` \
    --d_model 1280 \
    --n_layers 14 \
    --d_state 16 \
    --d_conv 4 \
    --expand 2 \
    --dropout 0.0 \
    \
    `# === MEMORY OPTIMIZATION (UNCHANGED) ===` \
    --gradient_checkpointing \
    \
    `# === BATCH CONFIGURATION (UNCHANGED) ===` \
    --batch_size 4 \
    --grad_accum 8 \
    --max_seq_len 2048 \
    \
    `# === PHASE 3: PER-GROUP LEARNING RATES ===` \
    --lr 5e-4 \
    --min_lr 1e-6 \
    --warmup_steps 500 \
    --weight_decay 0.005 \
    --max_grad_norm 3.0 \
    --bitlinear_lr_scale 5.0 \
    #--weights_only \
    --max_tokens 4000000000 \
    \
    `# === DATA CONFIGURATION (UNCHANGED) ===` \
    --data_dir "data/tokenized" \
    --en_ratio 0.3 \
    --pt_ratio 0.7 \
    --num_workers 4 \
    --prefetch_factor 2 \
    \
    `# === OUTPUT CONFIGURATION (UNCHANGED) ===` \
    --output_dir "model_204m" \
    \
    `# === EXPERIMENT TRACKING ===` \
    --wandb \
    --wandb_project "bitnet-mamba-hybrid" \
    --wandb_run_name "recovery-phase3-bitlinear-lr5x-$(date +%Y%m%d-%H%M%S)" \
    \
    `# === REPRODUCIBILITY ===` \
    --seed 42

echo ""
echo "============================================================"
echo "Phase 3 training completed or interrupted."
echo ""
echo "VERIFICATION:"
echo "1. Check loss curve: should drop below 4.0 within 1000 steps"
echo "2. Check BitLinear gradients: should be higher than before"
echo "3. Run inference: python inference_hybrid.py"
echo "============================================================"
