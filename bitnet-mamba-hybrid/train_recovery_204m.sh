#!/bin/bash
# =============================================================================
# BitNet-Mamba RECOVERY Training Script
# =============================================================================
#
# This script implements the recovery plan from DIAGNOSTIC_REPORT.md
#
# KEY CHANGES FROM ORIGINAL:
# --------------------------
# --lr 1e-4                    (was 3e-5) - 3.3x increase
# --max_grad_norm 1.5          (was 0.3)  - 5x relaxation
# --warmup_steps 2000          (was 400)  - 5x increase
# --weight_decay 0.01          (was 0.03) - reduced
# --weights_only               NEW FLAG   - reset optimizer, keep weights
#
# EXPECTED RESULTS:
# -----------------
# After 1000 steps: Loss < 4.0 (from current 4.77)
# After 5000 steps: Loss < 3.5
# GradNorm: 0.5-1.5 (not stuck at 0.3)
# No loss spikes > 50%
#
# =============================================================================

set -e  # Exit on error

# Change to script directory
cd "$(dirname -- "$0")"

# Print configuration summary
echo "============================================================"
echo "BitNet-Mamba RECOVERY Training - 204M Parameters"
echo "============================================================"
echo ""
echo "RECOVERY CHANGES:"
echo "  Learning Rate:    3e-5 → 1e-4 (3.3x increase)"
echo "  Gradient Clip:    0.3 → 1.5 (5x relaxation)"
echo "  Warmup Steps:     400 → 2000 (5x increase)"
echo "  Weight Decay:     0.03 → 0.01 (reduced)"
echo "  Mode:             Weights-only resume (reset optimizer)"
echo ""
echo "CURRENT STATE:"
echo "  Step:             28,438"
echo "  Tokens:           1.86B / 4B (46.6%)"
echo "  Current Loss:     ~4.77"
echo ""
echo "TARGETS:"
echo "  1000 steps:       Loss < 4.0"
echo "  5000 steps:       Loss < 3.5"
echo "  GradNorm:         0.5-1.5 (healthy range)"
echo ""
echo "============================================================"
echo ""

# Backup current state
echo "Creating backups..."
BACKUP_DIR="model_204m/checkpoints/backup_before_recovery_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

if [ -f "model_204m/checkpoints/checkpoint_interrupt_00028438.pt" ]; then
    cp model_204m/checkpoints/checkpoint_interrupt_00028438.pt "$BACKUP_DIR/"
    echo "✅ Backed up checkpoint to $BACKUP_DIR/"
fi

if [ -f "model_204m/training.log" ]; then
    cp model_204m/training.log "$BACKUP_DIR/training_before_recovery.log"
fi

if [ -f "model_204m/loss_history.csv" ]; then
    cp model_204m/loss_history.csv "$BACKUP_DIR/loss_history_before_recovery.csv"
fi

echo ""
echo "Backups complete. Starting recovery training..."
echo ""

# Verify data exists
if [ ! -d "data/tokenized/en" ] && [ ! -d "data/tokenized/pt" ]; then
    echo "ERROR: Preprocessed data not found!"
    echo "Please run: python preprocess_datasets.py --output_dir data/tokenized"
    exit 1
fi

# Main recovery training command
python train_hybrid-mamba-bitnet.py \
    \
    `# === MODEL ARCHITECTURE (UNCHANGED) ===` \
    --d_model 1280 \
    --n_layers 14 \
    --d_state 16 \
    --d_conv 4 \
    --expand 2 \
    --dropout 0.1 \
    \
    `# === MEMORY OPTIMIZATION (UNCHANGED) ===` \
    --gradient_checkpointing \
    \
    `# === BATCH CONFIGURATION (UNCHANGED) ===` \
    --batch_size 4 \
    --grad_accum 8 \
    --max_seq_len 2048 \
    \
    `# === RECOVERY HYPERPARAMETERS (MODIFIED) ===` \
    --lr 1e-4 \
    --min_lr 1e-6 \
    --warmup_steps 2000 \
    --weight_decay 0.01 \
    --max_grad_norm 1.5 \
    --weights_only \
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
    --wandb_run_name "recovery-204m-lr1e4-gradnorm1.5-$(date +%Y%m%d-%H%M%S)" \
    \
    `# === REPRODUCIBILITY ===` \
    --seed 42

echo ""
echo "============================================================"
echo "Training completed or interrupted."
echo ""
echo "NEXT STEPS:"
echo "1. Check loss_history.csv for improvement"
echo "2. Verify GradNorm is in 0.5-1.5 range (not 0.3)"
echo "3. Look for loss < 4.0 within 1000 steps"
echo "4. Continue monitoring for 5000 steps total"
echo ""
echo "If recovery fails after 5000 steps, see DIAGNOSTIC_REPORT.md"
echo "for alternative strategies."
echo "============================================================"
