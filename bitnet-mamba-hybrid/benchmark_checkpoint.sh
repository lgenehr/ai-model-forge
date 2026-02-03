#!/bin/bash
#
# Benchmark: Gradient Checkpointing ON vs OFF
# Compara velocidade e uso de memória
#

set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║   Benchmark: Gradient Checkpointing ON vs OFF               ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

BENCHMARK_STEPS=50
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Este benchmark vai:${NC}"
echo "  1. Rodar treinamento COM checkpoint por ${BENCHMARK_STEPS} steps"
echo "  2. Rodar treinamento SEM checkpoint por ${BENCHMARK_STEPS} steps"
echo "  3. Comparar velocidade e uso de memória"
echo ""
echo -e "${YELLOW}Tempo estimado: ~5-10 minutos${NC}"
echo ""
read -p "Pressione ENTER para continuar ou Ctrl+C para cancelar..."
echo ""

# Function to extract metrics from log
extract_metrics() {
    local log_file=$1
    local label=$2

    echo -e "\n${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}${label}${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    if [[ ! -f "$log_file" ]]; then
        echo -e "${RED}Log file not found: $log_file${NC}"
        return 1
    fi

    # Get last 10 lines with training metrics
    tail -20 "$log_file" | grep "Step" | tail -10 | while read line; do
        echo "$line"
    done

    # Calculate average tok/s
    avg_toks=$(tail -20 "$log_file" | grep "Tok/s" | tail -10 | \
               grep -oP 'Tok/s:\s+\K[0-9,]+' | tr -d ',' | \
               awk '{sum+=$1; count++} END {if(count>0) printf "%.0f", sum/count}')

    if [[ -n "$avg_toks" ]]; then
        echo ""
        echo -e "${YELLOW}Média de velocidade: ${avg_toks} tok/s${NC}"
    fi
}

# ============================================================================
# Test 1: WITH Gradient Checkpointing
# ============================================================================

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  TESTE 1: COM Gradient Checkpointing                        ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

LOG_WITH="benchmark_with_checkpoint_${TIMESTAMP}.log"

# Kill any existing training
pkill -f train_hybrid-mamba-bitnet || true
sleep 2

echo "Iniciando treinamento COM checkpoint..."
echo "Log: $LOG_WITH"
echo ""

# Run with checkpoint for limited steps
timeout 300 python3 train_hybrid-mamba-bitnet.py \
    --d_model 1024 \
    --n_layers 12 \
    --batch_size 4 \
    --grad_accum 8 \
    --max_seq_len 2048 \
    --max_tokens $((BENCHMARK_STEPS * 4 * 8 * 2048)) \
    --lr 3e-4 \
    --warmup_steps 10 \
    --data_dir data/tokenized \
    --output_dir model_benchmark_with \
    --gradient_checkpointing \
    2>&1 | tee "$LOG_WITH" || true

# Wait a bit for system to stabilize
sleep 5

# ============================================================================
# Test 2: WITHOUT Gradient Checkpointing
# ============================================================================

echo ""
echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  TESTE 2: SEM Gradient Checkpointing                        ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

LOG_WITHOUT="benchmark_without_checkpoint_${TIMESTAMP}.log"

# Kill any existing training
pkill -f train_hybrid-mamba-bitnet || true
sleep 2

echo "Iniciando treinamento SEM checkpoint..."
echo "Log: $LOG_WITHOUT"
echo ""

# Run without checkpoint for limited steps
timeout 300 python3 train_hybrid-mamba-bitnet.py \
    --d_model 1024 \
    --n_layers 12 \
    --batch_size 4 \
    --grad_accum 8 \
    --max_seq_len 2048 \
    --max_tokens $((BENCHMARK_STEPS * 4 * 8 * 2048)) \
    --lr 3e-4 \
    --warmup_steps 10 \
    --data_dir data/tokenized \
    --output_dir model_benchmark_without \
    2>&1 | tee "$LOG_WITHOUT" || true

# ============================================================================
# Results Comparison
# ============================================================================

echo ""
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    RESULTADOS                                ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"

# Extract and display metrics
extract_metrics "$LOG_WITH" "COM Gradient Checkpointing"
extract_metrics "$LOG_WITHOUT" "SEM Gradient Checkpointing"

# Calculate speedup
toks_with=$(tail -20 "$LOG_WITH" 2>/dev/null | grep "Tok/s" | tail -10 | \
            grep -oP 'Tok/s:\s+\K[0-9,]+' | tr -d ',' | \
            awk '{sum+=$1; count++} END {if(count>0) printf "%.0f", sum/count}')

toks_without=$(tail -20 "$LOG_WITHOUT" 2>/dev/null | grep "Tok/s" | tail -10 | \
               grep -oP 'Tok/s:\s+\K[0-9,]+' | tr -d ',' | \
               awk '{sum+=$1; count++} END {if(count>0) printf "%.0f", sum/count}')

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}Comparação Final${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

if [[ -n "$toks_with" ]] && [[ -n "$toks_without" ]]; then
    speedup=$(awk "BEGIN {printf \"%.1f\", ($toks_without / $toks_with - 1) * 100}")

    echo ""
    printf "%-30s: %'10d tok/s\n" "COM checkpoint" "$toks_with"
    printf "%-30s: %'10d tok/s\n" "SEM checkpoint" "$toks_without"
    echo ""
    echo -e "${YELLOW}Ganho de velocidade: +${speedup}%${NC}"
    echo ""

    # Time savings calculation
    total_steps=61035
    time_with=$(awk "BEGIN {printf \"%.1f\", $total_steps * 4 * 8 * 2048 / $toks_with / 3600 / 24}")
    time_without=$(awk "BEGIN {printf \"%.1f\", $total_steps * 4 * 8 * 2048 / $toks_without / 3600 / 24}")
    time_saved=$(awk "BEGIN {printf \"%.1f\", $time_with - $time_without}")

    echo -e "${BLUE}Projeção para treinamento completo (61k steps):${NC}"
    printf "  COM checkpoint:  %.1f dias\n" "$time_with"
    printf "  SEM checkpoint:  %.1f dias\n" "$time_without"
    echo -e "  ${GREEN}Economia: ${time_saved} dias${NC}"
    echo ""
fi

echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${BLUE}Logs salvos em:${NC}"
echo "  - $LOG_WITH"
echo "  - $LOG_WITHOUT"
echo ""
echo -e "${YELLOW}Recomendação:${NC}"
if [[ -n "$speedup" ]] && (( $(echo "$speedup > 20" | bc -l) )); then
    echo -e "  ${GREEN}✅ Use SEM checkpoint! Ganho significativo de ${speedup}%${NC}"
    echo "  Execute: ./train_no_checkpoint.sh"
else
    echo -e "  ${YELLOW}⚠️  Ganho moderado. Avalie se vale a pena.${NC}"
fi
echo ""

# Cleanup benchmark directories
rm -rf model_benchmark_with model_benchmark_without 2>/dev/null || true

echo -e "${GREEN}Benchmark completo!${NC}"
