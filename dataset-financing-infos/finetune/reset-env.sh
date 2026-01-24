#!/usr/bin/env bash
set -e

VENV_NAME="unsloth311"
PYTHON_VERSION="3.11.8"

echo "🔻 Desativando ambiente ativo (se existir)..."
pyenv deactivate 2>/dev/null || true

echo "🗑️ Removendo virtualenv '$VENV_NAME' (se existir)..."
if pyenv versions --bare | grep -q "^${VENV_NAME}$"; then
    pyenv uninstall -f "$VENV_NAME"
fi

echo "🧱 Recriando virtualenv '$VENV_NAME' com Python $PYTHON_VERSION..."
pyenv virtualenv "$PYTHON_VERSION" "$VENV_NAME"

echo "✅ Ativando virtualenv..."
pyenv activate "$VENV_NAME"

echo "⬆️ Atualizando ferramentas base..."
pip install --upgrade pip setuptools wheel

echo "🎉 Virtualenv '$VENV_NAME' resetada com sucesso."
