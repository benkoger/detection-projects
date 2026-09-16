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

# MegaDetector v6 yolov9-e checkpoint. PytorchWildlife fetches it from Zenodo
# with a single wget call and no retry; Zenodo returns 504s often enough that
# we download it ourselves first, with retries, into the exact path
# PytorchWildlife checks (torch.hub dir / checkpoints / MODEL_NAME).
CKPT_DIR="$TORCH_HOME/hub/checkpoints"
CKPT="$CKPT_DIR/MDV6-yolov9-e-1280.pt"
MDV6_URL="https://zenodo.org/records/15398270/files/MDV6-yolov9-e-1280.pt?download=1"
mkdir -p "$CKPT_DIR"
if [[ ! -s "$CKPT" || $(stat -c%s "$CKPT") -lt 10000000 ]]; then
    for attempt in $(seq 1 20); do
        echo "[prefetch] MDv6 checkpoint attempt $attempt"
        if curl -L -sS --max-time 900 -o "$CKPT.part" "$MDV6_URL" \
           && [[ $(stat -c%s "$CKPT.part") -gt 10000000 ]]; then
            mv "$CKPT.part" "$CKPT"; break
        fi
        rm -f "$CKPT.part"; sleep 30
    done
fi
[[ -s "$CKPT" ]] || { echo "[prefetch] could not download $CKPT"; exit 1; }
ls -la "$CKPT"

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
