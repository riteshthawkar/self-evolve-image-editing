#!/usr/bin/env bash
#SBATCH --job-name=qedit-data
#SBATCH --output=outputs/logs/slurm_%x_%j.out
#SBATCH --error=outputs/logs/slurm_%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-${SCRATCH:-$PWD/.cache}/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TOKENIZERS_PARALLELISM=false

DOWNLOAD_LIMIT="${DOWNLOAD_LIMIT:-20000}"
FILTER_LIMIT="${FILTER_LIMIT:-$DOWNLOAD_LIMIT}"
MAX_SELECTED="${MAX_SELECTED:-5000}"
VLM_BACKEND="${VLM_BACKEND:-qwen_vl}"
VLM_MODEL_ID="${VLM_MODEL_ID:-Qwen/Qwen3-VL-8B-Instruct}"

bash scripts/prepare_remote_data.sh \
  --stage all \
  --download-limit "$DOWNLOAD_LIMIT" \
  --filter-limit "$FILTER_LIMIT" \
  --max-selected "$MAX_SELECTED" \
  --vlm-backend "$VLM_BACKEND" \
  --vlm-model-id "$VLM_MODEL_ID" \
  --pilot-count "${PILOT_COUNT:-128}" \
  --main-count "${MAIN_COUNT:-1024}" \
  --heldout-count "${HELDOUT_COUNT:-128}" \
  --dataset-path "${DATASET_PATH:-regisss/coco_2017}" \
  --dataset-split "${DATASET_SPLIT:-train}" \
  --image-column "${IMAGE_COLUMN:-image}" \
  --caption-column "${CAPTION_COLUMN:-caption}" \
  --id-column "${ID_COLUMN:-image_id}"
