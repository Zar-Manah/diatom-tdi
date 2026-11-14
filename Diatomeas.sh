#!/usr/bin/env bash
# zar manah diatom tdi - Autonomous Linux Launcher

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

PYTHON_BIN="$(command -v python3 || command -v python)"
if [ -z "$PYTHON_BIN" ]; then
    echo "❌ No se encontró Python en el sistema."
    exit 1
fi

if "$PYTHON_BIN" -c "import torch, PIL" >/dev/null 2>&1; then
    RUNNER="$PYTHON_BIN"
else
    if [ ! -d ".venv" ]; then
        echo "[*] Preparando entorno local..."
        "$PYTHON_BIN" -m venv .venv
    fi
    RUNNER="$DIR/.venv/bin/python3"
    if ! "$RUNNER" -c "import torch, PIL" >/dev/null 2>&1; then
        echo "[*] Instalando dependencias (PyTorch, Pillow)..."
        "$RUNNER" -m pip install --quiet --upgrade pip
        "$RUNNER" -m pip install --quiet torch torchvision pillow
    fi
fi

echo "=========================================================="
echo "zar manah diatom tdi"
echo "=========================================================="

exec "$RUNNER" app/server.py --lan "$@"
