#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

usage() {
  cat <<'EOF'
Usage: bash scripts/run_gedit_export_shards.sh [options]

Runs GEdit export in contiguous shards. This is useful for resumable long runs:
each shard skips images that already exist unless --no-resume is passed through.

Options:
  --model-type TYPE        base, lora, or full. Default: base
  --model-name NAME        Output model name. Default: qwen_edit_2509_base
  --checkpoint PATH        Required for lora/full
  --shard-size N           Number of records per shard. Default: 128
  --total N                Total records in GEdit split. Default: 1212
  --steps N                Override generation.num_inference_steps
  --device DEVICE          auto, cuda, etc. Default: auto
  --set KEY=VALUE          Extra export override. Repeatable.
  -h, --help               Show this message.
EOF
}

MODEL_TYPE="base"
MODEL_NAME="qwen_edit_2509_base"
CHECKPOINT=""
SHARD_SIZE=128
TOTAL=1212
STEPS=""
DEVICE="auto"
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-type) MODEL_TYPE="$2"; shift ;;
    --model-name) MODEL_NAME="$2"; shift ;;
    --checkpoint) CHECKPOINT="$2"; shift ;;
    --shard-size) SHARD_SIZE="$2"; shift ;;
    --total) TOTAL="$2"; shift ;;
    --steps) STEPS="$2"; shift ;;
    --device) DEVICE="$2"; shift ;;
    --set) EXTRA+=(--set "$2"); shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
  shift
done

if [[ "$MODEL_TYPE" != "base" && -z "$CHECKPOINT" ]]; then
  echo "--checkpoint is required for model-type=$MODEL_TYPE" >&2
  exit 1
fi

offset=0
while (( offset < TOTAL )); do
  current_limit="$SHARD_SIZE"
  if (( offset + current_limit > TOTAL )); then
    current_limit=$(( TOTAL - offset ))
  fi
  cmd=(
    bash scripts/export_gedit.sh
    --device "$DEVICE"
    --offset "$offset"
    --limit "$current_limit"
    --set "model.model_type=$MODEL_TYPE"
    --set "model.model_name=$MODEL_NAME"
  )
  if [[ -n "$CHECKPOINT" ]]; then
    cmd+=(--set "model.checkpoint_path=$CHECKPOINT")
  fi
  if [[ -n "$STEPS" ]]; then
    cmd+=(--set "generation.num_inference_steps=$STEPS")
  fi
  cmd+=("${EXTRA[@]}")
  echo "+ ${cmd[*]}"
  "${cmd[@]}"
  offset=$(( offset + current_limit ))
done
