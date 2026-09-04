#!/usr/bin/env bash
# Launch StM Composer training from a config file.
#
#   bash training/train.sh training/configs/recipes/stm_main.conf      (DRY_RUN=1 prints the command only)
#
# Multi-node: export NUM_NODES, NODE_RANK, MASTER_ADDR, MASTER_PORT (and NUM_GPUS)
# before calling; single node otherwise.  All paths in the config may use
# $STM_HOME (repo root) and $STM_DATA (bulk data root, default $STM_HOME/data).
set -euo pipefail
export TOKENIZERS_PARALLELISM=false

STM_HOME=${STM_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
STM_DATA=${STM_DATA:-$STM_HOME/data}
export STM_HOME STM_DATA
CONFIG_FILE="${1:-}"
if [[ -z "$CONFIG_FILE" || ! -f "$CONFIG_FILE" ]]; then
  echo "usage: bash training/train.sh <config.conf>   (see training/configs/recipes/)" >&2
  exit 1
fi
source "$CONFIG_FILE"

MODEL_PATH=${MODEL_PATH:-THUDM/CogVideoX-5b-I2V}          # HF id or local path
OUTPUT_DIR=${OUTPUT_DIR:-$STM_HOME/outputs/$(basename "${CONFIG_FILE%.conf}")}
ACCELERATE_CONFIG_FILE=${ACCELERATE_CONFIG_FILE:-$STM_HOME/training/configs/accelerate/fsdp2_1node_4gpu.yaml}
LEARNING_RATE=${LEARNING_RATE:-5e-6}
LR_WARMUP_STEPS=${LR_WARMUP_STEPS:-1000}
LR_SCHEDULER=${LR_SCHEDULER:-cosine}
TOTAL_TRAIN_STEPS=${TOTAL_TRAIN_STEPS:-20000}
NUM_WORKERS=${NUM_WORKERS:-0}

ARGS=(
  --model_path "$MODEL_PATH" --model_name "${MODEL_NAME:-cogvideox-i2v}" --model_type "${MODEL_TYPE:-i2v}"
  --training_type "${TRAINING_TYPE:-sft}"
  --output_dir "$OUTPUT_DIR" --report_to "${REPORT_TO:-tensorboard}"
  --data_root "$TRAIN_DIR" --caption_column "${TRAIN_PROMPT_PATH:-prompts.txt}" --video_column "${TRAIN_VIDEO_PATH:-videos.txt}"
  --train_resolution "${TRAIN_RESOLUTION:-49x480x720}"
  --train_steps "$TOTAL_TRAIN_STEPS" --learning_rate "$LEARNING_RATE" --lr_scheduler "$LR_SCHEDULER" --lr_warmup_steps "$LR_WARMUP_STEPS"
  --seed "${SEED:-42}" --batch_size "${BATCH_SIZE:-2}" --gradient_accumulation_steps "${GRAD_ACC_STEPS:-1}" --mixed_precision "${MIXED_PRECISION:-bf16}"
  --num_workers "$NUM_WORKERS" --pin_memory True --nccl_timeout "${NCCL_TIMEOUT:-1800}"
  --checkpointing_steps "${CHECKPOINT_STEPS:-500}" --checkpointing_limit "${CHECKPOINT_LIMIT:-2}"
  # StM Composer
  --i2v_training_type "${I2V_TRAINING_TYPE:-fg_plain_bg_noisy}"
  --filtering_method "${FILTERING_METHOD:-main_train_foreground_volume}"
  --aug_config_path "${AUG_CONFIG_PATH:-$STM_HOME/training/configs/augmentation/paper.yaml}"
  --fg_column "${FG_COLUMN:-fg-black.txt}" --bg_column "${BG_COLUMN:-bg.txt}"
  --prompt_drop_ratio "${PROMPT_DROP_RATIO:-0.0}" --load_depth "${LOAD_DEPTH:-False}"
  --weight_fg "${WEIGHT_FG:-0.0}" --weight_bg "${WEIGHT_BG:-0.0}" --weight_orig "${WEIGHT_ORIG:-1.0}"
  --apply_fg_noise "${APPLY_FG_NOISE:-True}" --noise_mean_fg "${NOISE_MEAN_FG:--1.0}"
  --apply_bg_noise "${APPLY_BG_NOISE:-True}" --noise_mean_bg "${NOISE_MEAN_BG:--3.0}"
  --apply_mask_on_latents_fg "${APPLY_MASK_ON_LATENTS_FG:-True}" --apply_mask_on_latents_bg "${APPLY_MASK_ON_LATENTS_BG:-False}"
  --apply_mask_on_orig_loss "${APPLY_MASK_ON_ORIG_LOSS:-True}" --fg_weight_on_orig_loss "${FG_WEIGHT_ON_ORIG_LOSS:-0.5}"
  --apply_adaptive_mask_on_orig_loss "${APPLY_ADAPTIVE_MASK_ON_ORIG_LOSS:-False}"
  --apply_mask_on_fg_bg_loss "${APPLY_MASK_ON_FG_BG_LOSS:-True}"
  --train_data_type "${TRAIN_DATA_TYPE:-with-fg-bg}" --choose_dataset "${CHOOSE_DATASET:-fgbg}" --pos_embed_is_4d False
)
if [[ "${RESUME:-false}" == "true" ]]; then ARGS+=(--resume_from_checkpoint "$OUTPUT_DIR"); fi
if [[ "${TRAINING_TYPE:-sft}" == "lora" ]]; then ARGS+=(--rank "${LORA_RANK:-128}" --lora_alpha "${LORA_ALPHA:-64}"); fi
if [[ "${DO_VALIDATION:-true}" == "true" ]]; then
  ARGS+=(--do_validation true --validation_dir "$VAL_DIR" --validation_steps "${VAL_STEPS:-${CHECKPOINT_STEPS:-500}}"
         --validation_prompts "${VAL_PROMPT_PATH:-prompts.txt}" --validation_images "${VAL_IMG_PATH:-images.txt}"
         --validation_guidance_scales ${VAL_GUIDANCE_SCALES:-6.0} --inference_steps "${INFERENCE_STEPS:-50}" --gen_fps "${GEN_FPS:-16}")
  if [[ -n "${VAL_NUM_SAMPLES:-}" ]]; then ARGS+=(--validation_num_samples "$VAL_NUM_SAMPLES"); fi
fi

PYTHON=${PYTHON:-python}
LAUNCH=("$PYTHON" -m accelerate.commands.launch --config_file "$ACCELERATE_CONFIG_FILE")
if [[ -n "${NUM_NODES:-}" && "${NUM_NODES}" -gt 1 ]]; then
  LAUNCH+=(--main_process_ip "$MASTER_ADDR" --main_process_port "${MASTER_PORT:-29500}" --machine_rank "$NODE_RANK"
           --num_machines "$NUM_NODES" --num_processes $(( ${NUM_GPUS:-8} * NUM_NODES )))
fi
echo "${LAUNCH[@]} $STM_HOME/training/train.py ${ARGS[*]}"
if [[ "${DRY_RUN:-0}" == "1" ]]; then exit 0; fi
"${LAUNCH[@]}" "$STM_HOME/training/train.py" "${ARGS[@]}"
