#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible name. install.sh handles both first install and upgrades.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${PROJECT_DIR}/install.sh" "$@"
