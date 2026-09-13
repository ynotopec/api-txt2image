#!/usr/bin/env bash
set -euo pipefail

IP="${1:-${HOST:-0.0.0.0}}"
PORT="${2:-${PORT:-}}"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="$(basename "$PROJECT_DIR")"
VENV_DIR="${HOME}/venv/${PROJECT_NAME}"
VENV_PYTHON="${VENV_DIR}/bin/python"
REQ_FILE="${PROJECT_DIR}/requirements.txt"
REQ_HASH_FILE="${VENV_DIR}/.requirements.sha256"

cd "$PROJECT_DIR"

resolve_uv() {
  if command -v uv >/dev/null 2>&1; then
    command -v uv
    return 0
  fi

  if [[ -x "${HOME}/.local/bin/uv" ]]; then
    printf '%s\n' "${HOME}/.local/bin/uv"
    return 0
  fi

  return 1
}

if ! UV_BIN="$(resolve_uv)"; then
  echo "[INFO] uv is not installed; running install.sh"
  "${PROJECT_DIR}/install.sh"
  UV_BIN="$(resolve_uv)"
fi

mkdir -p "${HOME}/venv"

if [[ ! -d "$VENV_DIR" ]]; then
  "$UV_BIN" venv --system-site-packages "$VENV_DIR" >/dev/null
elif [[ ! -x "$VENV_PYTHON" ]]; then
  echo "[WARN] Existing virtualenv at ${VENV_DIR} is incomplete. Recreating it."
  "$UV_BIN" venv --clear --system-site-packages "$VENV_DIR" >/dev/null
fi

CURRENT_HASH="$(sha256sum "$REQ_FILE" | awk '{print $1}')"
INSTALLED_HASH=""
if [[ -f "$REQ_HASH_FILE" ]]; then
  INSTALLED_HASH="$(cat "$REQ_HASH_FILE")"
fi

if [[ "$CURRENT_HASH" != "$INSTALLED_HASH" ]]; then
  "${PROJECT_DIR}/install.sh"
else
  echo "[INFO] Dependencies already up to date (idempotent run)"
fi

if [[ -z "$PORT" ]]; then
  PORT="$($VENV_PYTHON -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')"
  echo "[INFO] Selected free port ${PORT}"
fi

if [[ ! "$PORT" =~ ^[0-9]+$ ]] || (( PORT < 1 || PORT > 65535 )); then
  echo "[ERROR] PORT must be an integer from 1 to 65535." >&2
  exit 2
fi

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
else
  echo "[WARN] .env file not found. Continuing with current environment variables."
  echo "[WARN] Copy .env.example to .env and set OPENAI_API_KEY or OPENAI_API_KEYS."
fi

if [[ -z "${OPENAI_API_KEY:-}" && -z "${OPENAI_API_KEYS:-}" ]]; then
  echo "[ERROR] OPENAI_API_KEY or OPENAI_API_KEYS is required. Set one in environment or .env." >&2
  exit 1
fi

exec "$VENV_PYTHON" -m uvicorn app:app \
  --host "$IP" \
  --port "$PORT"
