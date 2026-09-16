#!/usr/bin/env bash
# Download MegaDetector v6 (yolov9-e) and BioCLIP 2 weights into the shared
# project cache. Run on a login node, where outbound network works. Compute
# nodes then load from cache. Idempotent.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/project/uwyo-0007/software}"
VENV="${VENV:-$PROJECT_ROOT/.venv-wytrap}"
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/models}"
export TORCH_HOME="${TORCH_HOME:-$PROJECT_ROOT/models/torch}"
mkdir -p "$HF_HOME" "$TORCH_HOME"

"$VENV/bin/python" - <<'PY'
import os
print("HF_HOME   =", os.environ["HF_HOME"])
print("TORCH_HOME=", os.environ["TORCH_HOME"])

# BioCLIP 2 via huggingface_hub (same files pybioclip / open_clip load).
from huggingface_hub import snapshot_download
p = snapshot_download("imageomics/bioclip-2",
                      allow_patterns=["open_clip_model.safetensors", "open_clip_config.json",
                                      "tokenizer*", "*.json", "*.txt"])
print("BioCLIP 2 cached at", p)

# MegaDetector v6 via wytrap's Detector on CPU (weights land in TORCH_HOME / cwd cache
# per PytorchWildlife's downloader).
from wytrap.detector import Detector
Detector(device="cpu")
print("MegaDetector v6 weights cached")
PY
