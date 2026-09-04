#!/usr/bin/env bash
# One-shot environment setup for StM (data generation + training).
#
#   bash scripts/setup_env.sh [venv_dir]        # default: ~/venvs/stm
#
# Requires: a CUDA GPU driver, `uv` (https://docs.astral.sh/uv/) or python3.12 + venv, ffmpeg.
set -euo pipefail
STM_HOME=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VENV=${1:-$HOME/venvs/stm}
PY=3.12

if command -v uv >/dev/null 2>&1; then
  uv venv "$VENV" --python "$PY"
  PIP=(uv pip install --python "$VENV/bin/python")
else
  python$PY -m venv "$VENV"
  PIP=("$VENV/bin/python" -m pip install)
fi
"${PIP[@]}" torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
"${PIP[@]}" xformers==0.0.32.post2 --index-url https://download.pytorch.org/whl/cu128
"${PIP[@]}" -r "$STM_HOME/requirements.txt"
SAM2_BUILD_CUDA=0 "${PIP[@]}" -e "$STM_HOME/third_party/SegAnyMo/sam2"
"${PIP[@]}" -e "$STM_HOME"   # `import stm` from anywhere

command -v ffmpeg >/dev/null 2>&1 || echo "WARNING: ffmpeg not found - install it (apt-get install ffmpeg); video I/O needs it."
mkdir -p "$STM_HOME/data/ckpts" "$STM_HOME/data/raw" "$STM_HOME/data/clips" "$STM_HOME/data/datasets"
echo "Done. Activate with: source $VENV/bin/activate"
echo "Next: bash setup/download_checkpoints.sh   (Decomposer + CogVideoX-5b-I2V weights)"
