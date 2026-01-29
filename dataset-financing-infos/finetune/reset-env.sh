#!/usr/bin/env bash
set -e # Para o script se houver qualquer erro

VENV_NAME="unsloth311"
PYTHON_VERSION="3.11.8"

# --- 1. Configuração do Ambiente (Pyenv) ---

echo "🔻 Desativando ambiente ativo (se existir)..."
# Tenta desativar, ignora erro se não tiver nada ativo
pyenv deactivate 2>/dev/null || true 

echo "🗑️ Removendo virtualenv '$VENV_NAME' (se existir)..."
if pyenv versions --bare | grep -q "^${VENV_NAME}$"; then
    pyenv uninstall -f "$VENV_NAME"
fi

echo "🧱 Recriando virtualenv '$VENV_NAME' com Python $PYTHON_VERSION..."
# Garante que a versão base do python está instalada
pyenv install -s "$PYTHON_VERSION"
pyenv virtualenv "$PYTHON_VERSION" "$VENV_NAME"

echo "✅ Ativando virtualenv..."
# Necessário configurar o shell para o pyenv localmente neste script
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
pyenv activate "$VENV_NAME"

echo "⬆️ Atualizando pip e ferramentas de build..."
pip install --upgrade pip setuptools wheel

# --- 2. Instalação das Dependências ---

echo "🔥 Instalando PyTorch com suporte a CUDA (GPU)..."
# CRÍTICO: Usamos --index-url para garantir que baixe a versão com CUDA e não CPU.
# Nota: Se a versão 2.10.0 não existir no index cu121, remova o index ou ajuste a versão.
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0

echo "⚡ Instalando xFormers e Bitsandbytes..."
pip install xformers bitsandbytes

echo "🚀 Instalando Unsloth..."
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install unsloth-zoo

echo "🤗 Instalando Ecossistema Hugging Face (Transformers, TRL, PEFT)..."
pip install transformers trl accelerate datasets sentencepiece protobuf peft

echo "🛠️ Instalando Utilitários (Pandas, WandB, etc)..."
pip install pandas scipy wandb

echo "🎉 Instalação concluída com sucesso!"
echo "Teste final de GPU:"
python -c "import torch; print(f'GPU: {torch.cuda.is_available()} | Torch: {torch.__version__}')"