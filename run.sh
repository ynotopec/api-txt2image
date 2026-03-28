#!/usr/bin/env bash
set -euo pipefail

IP="${1:-0.0.0.0}"
PORT="${2:-8000}"

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
  echo "[ERROR] uv is required but was not found in PATH or ~/.local/bin/uv." >&2
  echo "[ERROR] Run ./upgrade.sh first." >&2
  exit 1
fi

mkdir -p "${HOME}/venv"

if [[ ! -d "$VENV_DIR" ]]; then
  "$UV_BIN" venv "$VENV_DIR" >/dev/null
elif [[ ! -x "$VENV_PYTHON" ]]; then
  echo "[WARN] Existing virtualenv at ${VENV_DIR} is incomplete. Recreating it."
  "$UV_BIN" venv --clear "$VENV_DIR" >/dev/null
fi

CURRENT_HASH="$(sha256sum "$REQ_FILE" | awk '{print $1}')"
INSTALLED_HASH=""
if [[ -f "$REQ_HASH_FILE" ]]; then
  INSTALLED_HASH="$(cat "$REQ_HASH_FILE")"
fi

if [[ "$CURRENT_HASH" != "$INSTALLED_HASH" ]]; then
  echo "[INFO] Installing/updating dependencies from requirements.txt"
  "$UV_BIN" pip install --python "$VENV_PYTHON" -r "$REQ_FILE"
  printf '%s' "$CURRENT_HASH" > "$REQ_HASH_FILE"
else
  echo "[INFO] Dependencies already up to date (idempotent run)"
fi

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
else
  echo "[WARN] .env file not found. Continuing with current environment variables."
  echo "[WARN] Copy .env.example to .env and set OPENAI_API_KEY."
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "[ERROR] OPENAI_API_KEY is required. Set it in environment or .env." >&2
  exit 1
fi

exec "$UV_BIN" run --python "$VENV_PYTHON" \
  uvicorn app:app \
  --host "$IP" \
  --port "$PORT"
