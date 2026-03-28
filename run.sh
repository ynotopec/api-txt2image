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

if ! command -v uv >/dev/null 2>&1; then
  echo "[ERROR] uv is required but not installed. Run ./upgrade.sh first." >&2
  exit 1
fi

mkdir -p "${HOME}/venv"

if [[ ! -d "$VENV_DIR" ]]; then
  uv venv "$VENV_DIR" >/dev/null
elif [[ ! -x "$VENV_PYTHON" ]]; then
  echo "[WARN] Existing virtualenv at ${VENV_DIR} is incomplete. Recreating it."
  uv venv --clear "$VENV_DIR" >/dev/null
fi

CURRENT_HASH="$(sha256sum "$REQ_FILE" | awk '{print $1}')"
INSTALLED_HASH=""
if [[ -f "$REQ_HASH_FILE" ]]; then
  INSTALLED_HASH="$(cat "$REQ_HASH_FILE")"
fi

if [[ "$CURRENT_HASH" != "$INSTALLED_HASH" ]]; then
  echo "[INFO] Installing/updating dependencies from requirements.txt"
  uv pip install --python "$VENV_PYTHON" -r "$REQ_FILE"
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
  echo "[WARN] Copy .env.example to .env and adjust values for production usage."
fi

exec uv run --python "$VENV_PYTHON" \
  uvicorn app:app \
  --host "$IP" \
  --port "$PORT"
