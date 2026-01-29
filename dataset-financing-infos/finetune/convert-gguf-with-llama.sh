#!/bin/bash

# ================= CONFIGURAÇÃO =================
# Caminho onde você copiou os arquivos .safetensors (dentro do WSL)
MODEL_DIR=~/meu_modelo_temp

# Caminho onde o llama.cpp está (ou será clonado)
LLAMA_CPP_DIR=~/llama.cpp

# Nome do arquivo final
OUTPUT_FINAL="$MODEL_DIR/modelo_qwen14b_q4_k_m.gguf"
# ================================================

# Cores para logs
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

set -e # Para o script se der erro em qualquer comando

echo -e "${YELLOW}🚀 Iniciando Processo de Conversão Manual para GGUF${NC}"

# 1. Verifica/Instala Dependências Python necessárias para o script de conversão
echo -e "\n${YELLOW}📦 Verificando dependências Python...${NC}"
pip install gguf protobuf sentencepiece torch --upgrade --quiet
echo -e "${GREEN}✅ Dependências ok.${NC}"

# 2. Verifica se o llama.cpp existe, se não, clona e compila
if [ ! -d "$LLAMA_CPP_DIR" ]; then
    echo -e "\n${YELLOW}⚠️  llama.cpp não encontrado em $LLAMA_CPP_DIR. Clonando...${NC}"
    git clone https://github.com/ggerganov/llama.cpp $LLAMA_CPP_DIR
    cd $LLAMA_CPP_DIR
    echo -e "${YELLOW}🔨 Compilando llama.cpp (isso pode demorar 1-2 min)...${NC}"
    make -j$(nproc)
    cd - > /dev/null
else
    # Se já existe, garante que está compilado
    if [ ! -f "$LLAMA_CPP_DIR/llama-quantize" ] && [ ! -f "$LLAMA_CPP_DIR/build/bin/llama-quantize" ]; then
        echo -e "\n${YELLOW}🔨 llama.cpp encontrado mas não compilado. Compilando...${NC}"
        cd $LLAMA_CPP_DIR
        make -j$(nproc)
        cd - > /dev/null
    fi
fi

# Localiza o binário correto do quantize (pode variar dependendo de como foi compilado)
if [ -f "$LLAMA_CPP_DIR/llama-quantize" ]; then
    QUANTIZE_BIN="$LLAMA_CPP_DIR/llama-quantize"
elif [ -f "$LLAMA_CPP_DIR/build/bin/llama-quantize" ]; then
    QUANTIZE_BIN="$LLAMA_CPP_DIR/build/bin/llama-quantize"
else
    echo -e "${RED}❌ Erro: Não foi possível encontrar o binário 'llama-quantize'. A compilação falhou?${NC}"
    exit 1
fi

echo -e "${GREEN}✅ llama.cpp pronto para uso.${NC}"

# 3. Conversão de Safetensors para GGUF (FP16)
# Isso cria um arquivo gigante temporário (~28GB)
TEMP_FP16="$MODEL_DIR/temp_model_fp16.gguf"

echo -e "\n${YELLOW}🔄 Passo 1/2: Convertendo pesos HF para GGUF FP16...${NC}"
python3 "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" "$MODEL_DIR" \
    --outfile "$TEMP_FP16" \
    --outtype f16

echo -e "${GREEN}✅ Conversão FP16 concluída.${NC}"

# 4. Quantização para Q4_K_M (Comprimindo)
echo -e "\n${YELLOW}jmf 📉 Passo 2/2: Quantizando para Q4_K_M (Comprimindo)...${NC}"
"$QUANTIZE_BIN" "$TEMP_FP16" "$OUTPUT_FINAL" q4_k_m

echo -e "${GREEN}✅ Quantização concluída! Arquivo salvo em: $OUTPUT_FINAL${NC}"

# 5. Limpeza (Opcional)
echo -e "\n${YELLOW}🧹 Deseja apagar o arquivo temporário FP16 de ~28GB? (s/n)${NC}"
read -r response
if [[ "$response" =~ ^([sS][iI]|[sS])$ ]]; then
    rm "$TEMP_FP16"
    echo -e "${GREEN}🗑️  Arquivo temporário removido. Espaço liberado.${NC}"
else
    echo -e "${YELLOW}Arquivo FP16 mantido em: $TEMP_FP16${NC}"
fi

echo -e "\n${GREEN}🎉 SUCESSO! Seu modelo está pronto para uso.${NC}"