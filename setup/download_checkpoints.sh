#!/usr/bin/env bash
# Download every off-the-shelf model used by the Decomposer and the Composer base model.
#
#   HF_HOME=/bulk/hf_cache bash setup/download_checkpoints.sh [--no-cogvideox]
#
# Hugging Face models go to $HF_HOME (default: data/hf_cache); the two non-HF checkpoints
# (BootsTAPIR, SAM2) go to data/ckpts; DINOv2 is fetched by torch.hub into data/ckpts/torch_hub.
set -euo pipefail
STM_HOME=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export HF_HOME=${HF_HOME:-$STM_HOME/data/hf_cache}
export TORCH_HOME=${TORCH_HOME:-$STM_HOME/data/ckpts/torch_hub}
PYTHON=${PYTHON:-python}
CKPT=$STM_HOME/data/ckpts
mkdir -p "$CKPT" "$HF_HOME" "$TORCH_HOME"

[ -f "$CKPT/bootstapir_checkpoint_v2.pt" ] || curl -L -o "$CKPT/bootstapir_checkpoint_v2.pt" https://storage.googleapis.com/dm-tapnet/bootstap/bootstapir_checkpoint_v2.pt
[ -f "$CKPT/sam2_hiera_large.pt" ] || curl -L -o "$CKPT/sam2_hiera_large.pt" https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt

MODELS="OpenGVLab/InternVL2_5-1B depth-anything/Depth-Anything-V2-Large-hf Changearthmore/moseg"
[[ "${1:-}" == "--no-cogvideox" ]] || MODELS="$MODELS THUDM/CogVideoX-5b-I2V"
"$PYTHON" - <<EOF
from huggingface_hub import snapshot_download
for m in "$MODELS".split():
    print(m, "->", snapshot_download(m))
print("zibojia/minimax-remover ->", snapshot_download("zibojia/minimax-remover", allow_patterns=["vae/*", "transformer/*", "scheduler/*"]))
import torch
torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")   # cached under TORCH_HOME
print("dinov2_vitb14 cached")
EOF
echo "All checkpoints ready (HF_HOME=$HF_HOME, ckpts=$CKPT)."
