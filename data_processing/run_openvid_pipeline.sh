#!/usr/bin/env bash
# Unattended driver for the StM Decomposer on OpenVid-1M (Panda-70M-derived slice).
# For every zip part (ranked by yield, see stm.data_generation.sources.openvid_index): stream only the
# filtered videos via HTTP range requests (no 40 GB zip download), chunk them into 49x480x720 clips, run
# caption -> motion segmentation -> layers -> inpainting with $GPUS workers, rebuild the manifest, and
# stop once $TARGET_DONE clips are fully decomposed.
#
#   PYTHON=~/venvs/stm/bin/python TARGET_DONE=100000 bash data_processing/run_openvid_pipeline.sh
set -uo pipefail
STM_HOME=${STM_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PYTHON=${PYTHON:-python}
GPUS=${GPUS:-"0 0 0 0"}          # one worker per listed GPU id (repeat an id for several workers, ~15 GB each)
TARGET_DONE=${TARGET_DONE:-100000}
MIN_MOTION=${MIN_MOTION:-2.0}
CLIPS_PER_VIDEO=${CLIPS_PER_VIDEO:-4}
FAST=${FAST:-0}                  # 1 = bf16 TAPIR + fp16 depth (lowers DAVIS IoU; off)
SEG_EXTRA=${SEG_EXTRA:-}         # e.g. "--depth-model depth-anything/Depth-Anything-V2-Small-hf --tracks-per-query-frame 1100"
RAW=$STM_HOME/data/raw/openvid
RANKING=${RANKING:-$RAW/part_ranking_m${MIN_MOTION}.json}
[ -f "$RANKING" ] || RANKING=$RAW/part_ranking.json
PARTS=${PARTS:-$($PYTHON -c "import json; print(' '.join(str(r['part']) for r in json.load(open('$RANKING')) if r['selected'] > 50))")}
FAST_FLAG=""; [ "$FAST" = "1" ] && FAST_FLAG="--fast"
export HF_HOME=${HF_HOME:-$STM_HOME/data/hf_cache} WANDB_MODE=disabled PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$STM_HOME"

done_count() { "$PYTHON" data_processing/generate_data.py status 2>/dev/null | "$PYTHON" -c "import sys,re; m=re.search(r\"'done': (\d+)\", sys.stdin.read()); print(m.group(1) if m else 0)"; }

echo "[driver] $(date) target=$TARGET_DONE fast=$FAST min_motion=$MIN_MOTION clips/video=$CLIPS_PER_VIDEO parts: $PARTS"
for p in $PARTS; do
  n=$(done_count)
  if [ "$n" -ge "$TARGET_DONE" ]; then echo "[driver] $(date) target reached ($n done)"; break; fi
  marker=$RAW/.part$p.m$MIN_MOTION.imported
  if [ ! -f "$marker" ]; then
    echo "[driver] $(date) fetching + chunking part$p (done so far: $n)"
    "$PYTHON" data_processing/generate_data.py import-openvid-remote --parts "$p" --min-motion "$MIN_MOTION" --clips-per-video "$CLIPS_PER_VIDEO" && touch "$marker"
    echo "[driver] $(date) part$p imported; clips=$(ls data/clips/openvid/*.mp4 2>/dev/null | wc -l)"
  fi
  echo "[driver] $(date) decomposing pending openvid clips (gpus $GPUS)"
  "$PYTHON" data_processing/generate_data.py process --sources openvid --gpus $GPUS --workers-per-gpu 1 $FAST_FLAG $SEG_EXTRA
  "$PYTHON" data_processing/generate_data.py manifest | tail -2
  "$PYTHON" data_processing/generate_data.py verify --max-clips 500 | tail -1   # 49x480x720@16fps sanity check on a sample
done
"$PYTHON" data_processing/generate_data.py stats
echo "[driver] $(date) finished ($(done_count) done)"
