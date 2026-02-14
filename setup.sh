#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

SKIP_INSTALL=0
SKIP_EDITABLE=0
FORCE_CONFIG=0
WITH_DEV=0

print_usage() {
  cat <<'EOF'
Usage: bash setup.sh [options]

Options:
  --skip-install   Skip `python -m pip install -r requirements.txt`
  --skip-editable  Skip `python -m pip install -e .`
  --force-config   Overwrite `.env` and `config/profiles.yaml` from example files
  --with-dev       Install editable package with dev dependencies (`-e .[dev]`)
  -h, --help       Show this help message
EOF
}

log_step() {
  printf '\n==> %s\n' "$1"
}

log_ok() {
  printf '[OK] %s\n' "$1"
}

log_warn() {
  printf '[WARN] %s\n' "$1"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-install)
      SKIP_INSTALL=1
      shift
      ;;
    --skip-editable)
      SKIP_EDITABLE=1
      shift
      ;;
    --force-config)
      FORCE_CONFIG=1
      shift
      ;;
    --with-dev)
      WITH_DEV=1
      shift
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    *)
      printf 'Unknown option: %s\n\n' "$1"
      print_usage
      exit 1
      ;;
  esac
done

if ! command -v python >/dev/null 2>&1; then
  printf 'python not found in PATH.\n'
  exit 1
fi

if ! command -v pip >/dev/null 2>&1; then
  printf 'pip not found in PATH.\n'
  exit 1
fi

log_step "Checking Python version"
python - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit("Python >= 3.10 is required.")
print(f"Python version: {sys.version.split()[0]}")
PY

log_step "Preparing folders"
mkdir -p storage/conversations storage/logs config
log_ok "Created storage and config folders if missing"

log_step "Preparing configuration files"
if [[ ${FORCE_CONFIG} -eq 1 ]]; then
  cp .env.example .env
  cp config/profiles.example.yaml config/profiles.yaml
  log_ok "Overwritten .env and config/profiles.yaml from example files"
else
  if [[ ! -f .env ]]; then
    cp .env.example .env
    log_ok "Created .env from .env.example"
  else
    log_warn "Kept existing .env"
  fi

  if [[ ! -f config/profiles.yaml ]]; then
    cp config/profiles.example.yaml config/profiles.yaml
    log_ok "Created config/profiles.yaml from config/profiles.example.yaml"
  else
    log_warn "Kept existing config/profiles.yaml"
  fi
fi

if [[ ${SKIP_INSTALL} -eq 0 ]]; then
  log_step "Installing requirements"
  python -m pip install -r requirements.txt
  log_ok "Installed requirements.txt"
else
  log_warn "Skipped requirements installation"
fi

if [[ ${SKIP_EDITABLE} -eq 0 ]]; then
  log_step "Installing project package"
  if [[ ${WITH_DEV} -eq 1 ]]; then
    python -m pip install -e ".[dev]"
    log_ok "Installed editable package with dev dependencies"
  else
    python -m pip install -e .
    log_ok "Installed editable package"
  fi
else
  log_warn "Skipped editable installation"
fi

cat <<'EOF'

Setup completed.

Next commands:
  llm-lab chat
  llm-lab web --ui streamlit
  llm-lab web --ui gradio

EOF
