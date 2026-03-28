#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="$(basename "$PROJECT_DIR")"
VENV_DIR="${HOME}/venv/${PROJECT_NAME}"
VENV_PYTHON="${VENV_DIR}/bin/python"
REQ_FILE="${PROJECT_DIR}/requirements.txt"

cd "$PROJECT_DIR"

if ! command -v uv >/dev/null 2>&1; then
  echo "[INFO] Installing uv via pip"
  python3 -m pip install --user --upgrade uv
fi

mkdir -p "${HOME}/venv"
uv venv "$VENV_DIR"

# Refresh base packaging tools first
uv pip install --python "$VENV_PYTHON" --upgrade pip setuptools wheel

# Upgrade project requirements
uv pip install --python "$VENV_PYTHON" --upgrade -r "$REQ_FILE"

echo "[INFO] Upgrade complete for ${PROJECT_NAME} (${VENV_DIR})"
