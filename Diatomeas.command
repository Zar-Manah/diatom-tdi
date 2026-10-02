#!/bin/bash
# zar manah diatom tdi - Autonomous Launcher

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

export PATH="/opt/homebrew/bin:/usr/local/bin:$HOME/.local/bin:$PATH"

# 1. Buscar Python en el sistema
PYTHON_BIN=""
for PY in "/opt/homebrew/bin/python3" "/usr/local/bin/python3" "$HOME/.local/bin/python3" "python3" "python"; do
    if command -v "$PY" >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v "$PY")"
        break
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo "=========================================================="
    echo "❌ No se encontró Python 3 en el sistema."
    echo "=========================================================="
    read -p "Presiona Enter para cerrar esta ventana..."
    exit 1
fi

# 2. Si el python ya tiene torch, usarlo directamente
if "$PYTHON_BIN" -c "import torch, PIL" >/dev/null 2>&1; then
    RUNNER="$PYTHON_BIN"
else
    # 3. Si no tiene torch, preparar un .venv local silenciosamente
    if [ ! -d ".venv" ]; then
        echo "[*] Preparando entorno local para Diatomeas..."
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

"$RUNNER" app/server.py --lan

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "⚠️ La aplicación terminó con código $EXIT_CODE."
    read -p "Presiona Enter para cerrar esta ventana..."
fi
