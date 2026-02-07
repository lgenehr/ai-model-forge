#!/bin/bash
# =============================================================================
# BitNet-Mamba RECOVERY PHASE 2 - Hyperparameters Mais Agressivos
# =============================================================================
#
# A fase 1 não mostrou melhoria após 242 steps pós-warmup.
# Esta fase 2 usa hyperparameters MAIS AGRESSIVOS:
#
# MUDANÇAS DA FASE 1 → FASE 2:
# --lr 1e-4 → 3e-4           (3x aumento - AGRESSIVO!)
# --max_grad_norm 1.5 → 2.5  (66% relaxamento)
# --warmup_steps 2000 → 1000 (warmup mais curto pois já passou)
# --weight_decay 0.01 → 0.005 (ainda mais baixo)
#
# JUSTIFICATIVA:
# - Loss não caiu após 242 steps pós-warmup
# - Modelo pode precisar de LR mais alto para "desbloquear"
# - GradNorm saudável sugere que podemos aumentar LR com segurança
# - Teste de overfitting provou que arquitetura funciona
#
# =============================================================================

set -e

cd "$(dirname -- "$0")/.."

echo "============================================================"
echo "BitNet-Mamba RECOVERY PHASE 2 - Aggressive Hyperparameters"
echo "============================================================"
echo ""
echo "PHASE 1 RESULTS:"
echo "  Steps pós-warmup: 242"
echo "  Loss improvement: NONE (4.77 → 4.80)"
echo "  Decision:         TRY MORE AGGRESSIVE SETTINGS"
echo ""
echo "PHASE 2 CHANGES:"
echo "  Learning Rate:    1e-4 → 3e-4 (3x increase!)"
echo "  Gradient Clip:    1.5 → 2.5 (66% relaxation)"
echo "  Warmup Steps:     2000 → 1000 (shorter, already warmed up)"
echo "  Weight Decay:     0.01 → 0.005 (reduced)"
echo ""
echo "CURRENT STATE:"
echo "  Step:             30,680"
echo "  Loss:             4.81"
echo "  Status:           Stagnant"
echo ""
echo "NEW TARGETS:"
echo "  500 steps:        Loss < 4.5"
echo "  1000 steps:       Loss < 4.0"
echo "  GradNorm:         0.5-3.0 (wider range for higher LR)"
echo ""
echo "============================================================"
echo ""

# Backup current state
echo "Creating backup..."
BACKUP_DIR="model_204m/checkpoints/backup_phase2_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

if [ -f "model_204m/checkpoints/checkpoint_interrupt_00030680.pt" ]; then
    cp model_204m/checkpoints/checkpoint_interrupt_00030680.pt "$BACKUP_DIR/"
    echo "✅ Backed up checkpoint to $BACKUP_DIR/"
fi

echo ""
echo "Starting Phase 2 training with AGGRESSIVE settings..."
echo ""

# Phase 2 training with more aggressive hyperparameters
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
    `# === PHASE 2: AGGRESSIVE HYPERPARAMETERS ===` \
    --lr 3e-4 \
    --min_lr 1e-6 \
    --warmup_steps 1000 \
    --weight_decay 0.005 \
    --max_grad_norm 2.5 \
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
    --wandb_run_name "recovery-phase2-lr3e4-gradnorm2.5-$(date +%Y%m%d-%H%M%S)" \
    \
    `# === REPRODUCIBILITY ===` \
    --seed 42

echo ""
echo "============================================================"
echo "Phase 2 training completed or interrupted."
echo ""
echo "NEXT STEPS:"
echo "1. Monitor loss for improvement within 500 steps"
echo "2. If loss < 4.5 within 500 steps: SUCCESS"
echo "3. If loss > 4.5 after 1000 steps: Consider architecture review"
echo "============================================================"
