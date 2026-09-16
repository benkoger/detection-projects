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
MDV6_TRIES="${MDV6_TRIES:-5}"
if [[ ! -s "$CKPT" || $(stat -c%s "$CKPT") -lt 10000000 ]]; then
    for attempt in $(seq 1 "$MDV6_TRIES"); do
        echo "[prefetch] MDv6 checkpoint attempt $attempt"
        if curl -L -sS --max-time 900 -o "$CKPT.part" "$MDV6_URL" \
           && [[ $(stat -c%s "$CKPT.part") -gt 10000000 ]]; then
            mv "$CKPT.part" "$CKPT"; break
        fi
        rm -f "$CKPT.part"; sleep 30
    done
fi
if [[ -s "$CKPT" ]]; then
    ls -la "$CKPT"
else
    if [[ "$MDV6_TRIES" -gt 0 ]]; then
        echo "[prefetch] Zenodo unreachable; MDv6 not cached. Jobs will use MDV1000-redwood."
    else
        echo "[prefetch] MDV6_TRIES=0, skipping Zenodo."
    fi
fi

# MegaDetector v1000 "redwood" from the Hugging Face mirror. Always cached,
# so a job can run when Zenodo is down (DETECTOR=auto picks it up).
"$VENV/bin/python" - <<'PY'
from huggingface_hub import hf_hub_download
p = hf_hub_download("agentmorris/megadetector", "md_v1000.0.0-redwood.pt")
print("MDv1000-redwood cached at", p)
PY

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

# Load whichever detector is cached through wytrap's wrapper so a bad
# checkpoint fails here, on the login node, not in the job.
import os
from wytrap.detector import Detector
ckpt = os.path.join(os.environ["TORCH_HOME"], "hub", "checkpoints", "MDV6-yolov9-e-1280.pt")
version = "MDV6-yolov9-e" if os.path.exists(ckpt) else "MDV1000-redwood"
Detector(device="cpu", version=version)
print(f"detector load ok: {version}")
PY
