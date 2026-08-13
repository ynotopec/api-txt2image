#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="$(basename "$PROJECT_DIR")"
VENV_DIR="${HOME}/venv/${PROJECT_NAME}"
VENV_PYTHON="${VENV_DIR}/bin/python"
REQ_FILE="${PROJECT_DIR}/requirements.txt"
REQ_HASH_FILE="${VENV_DIR}/.requirements.sha256"

resolve_uv() {
  command -v uv 2>/dev/null || {
    [[ -x "${HOME}/.local/bin/uv" ]] && printf '%s\n' "${HOME}/.local/bin/uv"
  }
}

install_uv() {
  echo "[INFO] Installing uv"
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="${HOME}/.local/bin" sh
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="${HOME}/.local/bin" sh
  else
    echo "[ERROR] curl or wget is required to install uv." >&2
    exit 1
  fi
}

UV_BIN="$(resolve_uv || true)"
if [[ -z "$UV_BIN" ]]; then
  install_uv
  UV_BIN="$(resolve_uv || true)"
fi
if [[ -z "$UV_BIN" ]]; then
  echo "[ERROR] uv installation did not produce an executable." >&2
  exit 1
fi

mkdir -p "${HOME}/venv"
if [[ ! -x "$VENV_PYTHON" ]]; then
  if [[ -e "$VENV_DIR" ]]; then
    echo "[WARN] Recreating incomplete environment: ${VENV_DIR}"
    "$UV_BIN" venv --clear --system-site-packages "$VENV_DIR"
  else
    # Preserve NVIDIA's platform-tuned PyTorch when a DGX image provides it.
    "$UV_BIN" venv --system-site-packages "$VENV_DIR"
  fi
fi

# uv resolves the correct PyTorch wheels for both x86_64 H100 hosts and
# aarch64 DGX Spark hosts. NVIDIA's installed driver supplies CUDA at runtime.
echo "[INFO] Installing compatible dependency upgrades"
"$UV_BIN" pip install --python "$VENV_PYTHON" --upgrade -r "$REQ_FILE"
sha256sum "$REQ_FILE" | awk '{printf "%s", $1}' > "$REQ_HASH_FILE"

echo "[INFO] Ready: ${VENV_DIR}"
