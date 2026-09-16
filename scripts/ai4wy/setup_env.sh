#!/usr/bin/env bash
# Create the wytrap virtual environment on ARCC (ai4wy) with uv.
#
# Run once from the repo root on a LOGIN node (compute nodes may lack
# outbound network):
#
#   bash scripts/ai4wy/setup_env.sh
#
# Environment overrides:
#   PROJECT_ROOT  default /project/uwyo-0007/software
#   VENV          default $PROJECT_ROOT/.venv-wytrap
#   PYTHON        default 3.11 (uv downloads a matching CPython if none is loaded)
#
# Everything big (uv cache, model weights) is kept under $PROJECT_ROOT so it
# does not hit the $HOME quota and is shared across users of the project.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/project/uwyo-0007/software}"
VENV="${VENV:-$PROJECT_ROOT/.venv-wytrap}"
PYTHON="${PYTHON:-3.11}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export UV_CACHE_DIR="${UV_CACHE_DIR:-$PROJECT_ROOT/.uv-cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$PROJECT_ROOT/.uv-python}"
mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$PROJECT_ROOT/models"

echo "[setup] host: $(hostname)  arch: $(uname -m)"
echo "[setup] repo: $REPO_ROOT"
echo "[setup] venv: $VENV"

# 1. uv (single static binary, no root needed). Installs to ~/.local/bin.
if ! command -v uv >/dev/null 2>&1; then
    echo "[setup] installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

# 2. Virtual environment.
if [[ ! -x "$VENV/bin/python" ]]; then
    uv venv --python "$PYTHON" "$VENV"
fi
export VIRTUAL_ENV="$VENV"

# 3. Dependencies, then the two local packages without their pinned deps.
uv pip install -r "$REPO_ROOT/requirements-ai4wy.txt"
uv pip install --no-deps -e "$REPO_ROOT/wytrap" -e "$REPO_ROOT/koger_detection"

# 4. Sanity check (no weights downloaded yet).
"$VENV/bin/python" - <<'PY'
import platform, torch
print(f"python {platform.python_version()} on {platform.machine()}")
print(f"torch {torch.__version__}, cuda build {torch.version.cuda}, cuda available {torch.cuda.is_available()}")
import PytorchWildlife, bioclip, wytrap, koger_detection.utils.json  # noqa
print("imports ok: PytorchWildlife, bioclip, wytrap, koger_detection")
PY
"$VENV/bin/wytrap" --version
"$VENV/bin/wytrap" species --list wyoming_all --count-only

cat <<MSG

[setup] done. Activate with:
    source $VENV/bin/activate
Then pre-download weights on the login node (network) before the first GPU job:
    HF_HOME=$PROJECT_ROOT/models TORCH_HOME=$PROJECT_ROOT/models/torch \\
        bash scripts/ai4wy/prefetch_weights.sh
MSG
