#!/usr/bin/env bash
set -e
REPO_URL="${REPO_URL:-https://github.com/zots0127/Iridium-emitters.git}"
REPO_DIR="${REPO_DIR:-Iridium-emitters}"
BRANCH="${BRANCH:-main}"
ts() { date '+%Y-%m-%d %H:%M:%S'; }
info() { echo "[$(ts)] INFO: $*"; }
warn() { echo "[$(ts)] WARNING: $*"; }
err() { echo "[$(ts)] ERROR: $*" >&2; }
if ! command -v bash >/dev/null; then err "bash not found"; exit 1; fi
OS="$(uname)"
install_with_brew() {
  if ! command -v brew >/dev/null; then /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"; fi
  brew update || true
  for pkg in "$@"; do brew install "$pkg" || true; done
}
install_with_apt() { sudo apt-get update -y || true; sudo apt-get install -y "$@" || true; }
install_with_yum() { sudo yum install -y "$@" || true; }
install_with_pacman() { sudo pacman -Sy --noconfirm "$@" || true; }
ensure_tool() {
  local t="$1"
  if command -v "$t" >/dev/null; then return 0; fi
  info "Installing $t"
  if [ "$OS" = "Darwin" ]; then
    install_with_brew "$t"
  elif command -v apt-get >/dev/null; then
    install_with_apt "$t"
  elif command -v yum >/dev/null; then
    install_with_yum "$t"
  elif command -v pacman >/dev/null; then
    install_with_pacman "$t"
  else
    warn "Unknown package manager; please install $t manually"
  fi
}
ensure_tool curl
ensure_tool git
if [ -d "$REPO_DIR/.git" ]; then
  info "Updating repository"
  cd "$REPO_DIR"
  git fetch --depth 1 origin "$BRANCH"
  git checkout "$BRANCH"
  git pull --ff-only --depth 1
else
  info "Cloning repository"
  git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$REPO_DIR"
  cd "$REPO_DIR"
fi
if [ -f "uv.sh" ]; then
  info "Setting up environment via uv.sh"
  bash uv.sh
else
  if command -v python3 >/dev/null; then PY=python3; elif command -v python >/dev/null; then PY=python; else PY=""; fi
  if [ -n "$PY" ] && [ -f "requirements.txt" ]; then
    info "Creating venv and installing requirements"
    "$PY" -m venv .venv || true
    if [ -f ".venv/bin/activate" ]; then . ".venv/bin/activate"; elif [ -f ".venv/Scripts/activate" ]; then . ".venv/Scripts/activate"; fi
    pip install -r requirements.txt || true
  fi
fi
if [ -f "run_workflow.sh" ]; then
  info "Running workflow"
  bash run_workflow.sh
elif [ -f "project_workflow.sh" ]; then
  info "Running workflow"
  bash project_workflow.sh
else
  err "No workflow script found"; exit 1
fi
