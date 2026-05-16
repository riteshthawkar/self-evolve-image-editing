#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

usage() {
  cat <<'EOF'
Usage: bash scripts/prepare_remote_data.sh [options]

Stages a bounded source-image pool for one-GPU self-evolving experiments:
  download HF images -> open-VLM source filtering -> deterministic pilot/main/heldout splits.

Options:
  --stage STAGE                all, download, filter, split. Default: all
  --dataset-path ID            HF dataset ID. Default from configs/data/remote_source_pool.yaml
  --dataset-name NAME          Optional HF dataset config name.
  --dataset-split SPLIT        HF split. Default: train
  --image-column NAME          HF image column. Default: image
  --caption-column NAME        Optional HF caption column.
  --id-column NAME             Optional stable ID column.
  --download-limit N           Number of raw source images to save. Default: 20000
  --filter-limit N             Number of raw images to score. Default: download limit
  --max-selected N             Number of selected images to keep. Default: 5000
  --raw-dir PATH               Raw image output directory. Default: data/unlabeled/raw/coco2017
  --selected-dir PATH          Selected manifest output directory. Default: data/unlabeled/selected/coco2017
  --split-dir PATH             Split output directory. Default: data/unlabeled/splits/coco2017
  --vlm-backend NAME           qwen_vl or heuristic. Default: qwen_vl
  --vlm-model-id ID            Open VLM model ID. Default: Qwen/Qwen3-VL-8B-Instruct
  --vlm-max-new-tokens N       Max JSON output tokens from VLM. Default: 192
  --progress-every N           Print filter progress every N scored images. Default: 10
  --pilot-count N              Pilot split size. Default: 128
  --main-count N               Main split size. Default: 1024
  --heldout-count N            Heldout split size. Default: 128
  --seed N                     Split/download seed. Default: 123
  -h, --help                   Show this message.

Environment:
  HF_HOME, HF_DATASETS_CACHE, TRANSFORMERS_CACHE can point to scratch storage.
EOF
}

STAGE="all"
DATASET_PATH=""
DATASET_NAME=""
DATASET_SPLIT="train"
IMAGE_COLUMN="image"
CAPTION_COLUMN=""
ID_COLUMN=""
DOWNLOAD_LIMIT=20000
FILTER_LIMIT=""
MAX_SELECTED=5000
RAW_DIR="data/unlabeled/raw/coco2017"
SELECTED_DIR="data/unlabeled/selected/coco2017"
SPLIT_DIR="data/unlabeled/splits/coco2017"
VLM_BACKEND="qwen_vl"
VLM_MODEL_ID="Qwen/Qwen3-VL-8B-Instruct"
VLM_MAX_NEW_TOKENS=192
PROGRESS_EVERY=10
PILOT_COUNT=128
MAIN_COUNT=1024
HELDOUT_COUNT=128
SEED=123

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage) STAGE="$2"; shift ;;
    --dataset-path) DATASET_PATH="$2"; shift ;;
    --dataset-name) DATASET_NAME="$2"; shift ;;
    --dataset-split) DATASET_SPLIT="$2"; shift ;;
    --image-column) IMAGE_COLUMN="$2"; shift ;;
    --caption-column) CAPTION_COLUMN="$2"; shift ;;
    --id-column) ID_COLUMN="$2"; shift ;;
    --download-limit) DOWNLOAD_LIMIT="$2"; shift ;;
    --filter-limit) FILTER_LIMIT="$2"; shift ;;
    --max-selected) MAX_SELECTED="$2"; shift ;;
    --raw-dir) RAW_DIR="$2"; shift ;;
    --selected-dir) SELECTED_DIR="$2"; shift ;;
    --split-dir) SPLIT_DIR="$2"; shift ;;
    --vlm-backend) VLM_BACKEND="$2"; shift ;;
    --vlm-model-id) VLM_MODEL_ID="$2"; shift ;;
    --vlm-max-new-tokens) VLM_MAX_NEW_TOKENS="$2"; shift ;;
    --progress-every) PROGRESS_EVERY="$2"; shift ;;
    --pilot-count) PILOT_COUNT="$2"; shift ;;
    --main-count) MAIN_COUNT="$2"; shift ;;
    --heldout-count) HELDOUT_COUNT="$2"; shift ;;
    --seed) SEED="$2"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
  shift
done

if [[ -z "$FILTER_LIMIT" ]]; then
  FILTER_LIMIT="$DOWNLOAD_LIMIT"
fi

mkdir -p "$RAW_DIR" "$SELECTED_DIR" "$SPLIT_DIR" outputs/logs

run_download() {
  cmd=(
    "${PYTHON:-python3}" -m qwen_edit_project.data.download_hf_images
    --config configs/data/remote_source_pool.yaml
    --limit "$DOWNLOAD_LIMIT"
    --set "runtime.seed=$SEED"
    --set "dataset.split=$DATASET_SPLIT"
    --set "dataset.image_column=$IMAGE_COLUMN"
    --set "output.images_dir=$RAW_DIR"
    --set "output.metadata_jsonl=${RAW_DIR%/}_metadata.jsonl"
    --set "output.summary_json=${RAW_DIR%/}_download_summary.json"
  )
  if [[ -n "$DATASET_PATH" ]]; then
    cmd+=(--set "dataset.path=$DATASET_PATH")
  fi
  if [[ -n "$DATASET_NAME" ]]; then
    cmd+=(--set "dataset.name=$DATASET_NAME")
  fi
  if [[ -n "$CAPTION_COLUMN" ]]; then
    cmd+=(--set "dataset.caption_column=$CAPTION_COLUMN")
  fi
  if [[ -n "$ID_COLUMN" ]]; then
    cmd+=(--set "dataset.id_column=$ID_COLUMN")
  fi
  "${cmd[@]}"
}

run_filter() {
  "${PYTHON:-python3}" -m qwen_edit_project.data.select_unlabeled_images \
    --config configs/data/source_image_filter.yaml \
    --limit "$FILTER_LIMIT" \
    --set "input.images_dir=$RAW_DIR" \
    --set "vlm.backend=$VLM_BACKEND" \
    --set "vlm.model_id=$VLM_MODEL_ID" \
    --set "vlm.max_new_tokens=$VLM_MAX_NEW_TOKENS" \
    --set "selection.max_selected=$MAX_SELECTED" \
    --set "selection.progress_every=$PROGRESS_EVERY" \
    --set "output.selected_manifest_jsonl=$SELECTED_DIR/manifest.jsonl" \
    --set "output.rejected_manifest_jsonl=$SELECTED_DIR/rejected.jsonl" \
    --set "output.score_jsonl=$SELECTED_DIR/scores.jsonl" \
    --set "output.summary_json=$SELECTED_DIR/selection_summary.json"
}

run_split() {
  "${PYTHON:-python3}" -m qwen_edit_project.data.split_source_manifest \
    --input "$SELECTED_DIR/manifest.jsonl" \
    --output-dir "$SPLIT_DIR" \
    --pilot-count "$PILOT_COUNT" \
    --main-count "$MAIN_COUNT" \
    --heldout-count "$HELDOUT_COUNT" \
    --seed "$SEED"
}

case "$STAGE" in
  all)
    run_download
    run_filter
    run_split
    ;;
  download) run_download ;;
  filter) run_filter ;;
  split) run_split ;;
  *)
    echo "Unsupported stage: $STAGE" >&2
    usage >&2
    exit 1
    ;;
esac
