#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
. "$ROOT/scripts/experiment_common.sh"

usage() {
  cat <<'EOF'
Usage: bash scripts/run_edit_model_suite.sh [options]

Options:
  --model-type TYPE             base, lora, or full. Default: lora
  --train                       Run training before evaluation for lora or full.
  --resume                      Resume training from an existing checkpoint.
  --checkpoint PATH             Evaluate this checkpoint for lora or full.
  --checkpoint-dir PATH         Directory used to auto-discover the latest checkpoint.
  --model-name NAME             Override benchmark model_name.
  --limit N                     Limit export count for GEdit and ImgEdit.
  --skip-validate               Skip validation image generation.
  --skip-export                 Skip benchmark export.
  --skip-score                  Skip benchmark scoring.
  --dry-run                     Print the composed commands without executing them.
  --resume-arg ARG              Raw upstream resume argument for full training. Repeatable.
  --set-train KEY=VALUE         Extra training override. Repeatable.
  --set-eval KEY=VALUE          Extra evaluation override. Repeatable.
  -h, --help                    Show this message.

Examples:
  bash scripts/run_edit_model_suite.sh --model-type lora --train --limit 64
  bash scripts/run_edit_model_suite.sh --model-type lora --train --resume --checkpoint-dir outputs/checkpoints/Qwen-Image-Edit-2509_lora
  bash scripts/run_edit_model_suite.sh --model-type full --checkpoint outputs/checkpoints/foo/latest.safetensors
  bash scripts/run_edit_model_suite.sh --model-type base --dry-run --limit 8
EOF
}

default_checkpoint_dir() {
  case "$1" in
    lora) echo "outputs/checkpoints/Qwen-Image-Edit-2509_lora" ;;
    full) echo "outputs/checkpoints/Qwen-Image-Edit-2509_full" ;;
    base) echo "" ;;
    *)
      echo "Unsupported model type: $1" >&2
      exit 1
      ;;
  esac
}

train_script_for_type() {
  case "$1" in
    lora) echo "scripts/train_lora_2509.sh" ;;
    full) echo "scripts/train_full_2509.sh" ;;
    *)
      echo "" ;;
  esac
}

validate_script_for_type() {
  case "$1" in
    lora) echo "scripts/validate_lora_2509.sh" ;;
    full) echo "scripts/validate_full_2509.sh" ;;
    *)
      echo "" ;;
  esac
}

RUN_TRAIN=0
RUN_VALIDATE=1
RUN_EXPORT=1
RUN_SCORE=1
DRY_RUN=0
RESUME_TRAIN=0
MODEL_TYPE="lora"
CHECKPOINT=""
CHECKPOINT_DIR=""
MODEL_NAME=""
LIMIT=""
RESUME_ARGS=()
TRAIN_OVERRIDES=()
EVAL_OVERRIDES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-type)
      MODEL_TYPE="$2"
      shift
      ;;
    --train)
      RUN_TRAIN=1
      ;;
    --resume)
      RESUME_TRAIN=1
      ;;
    --checkpoint)
      CHECKPOINT="$2"
      shift
      ;;
    --checkpoint-dir)
      CHECKPOINT_DIR="$2"
      shift
      ;;
    --model-name)
      MODEL_NAME="$2"
      shift
      ;;
    --limit)
      LIMIT="$2"
      shift
      ;;
    --skip-validate)
      RUN_VALIDATE=0
      ;;
    --skip-export)
      RUN_EXPORT=0
      ;;
    --skip-score)
      RUN_SCORE=0
      ;;
    --dry-run)
      DRY_RUN=1
      ;;
    --resume-arg)
      RESUME_ARGS+=("$2")
      shift
      ;;
    --set-train)
      TRAIN_OVERRIDES+=("$2")
      shift
      ;;
    --set-eval)
      EVAL_OVERRIDES+=("$2")
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
  shift
done

if (( DRY_RUN )); then
  experiment_enable_dry_run
fi

if [[ -z "$CHECKPOINT_DIR" ]]; then
  CHECKPOINT_DIR="$(default_checkpoint_dir "$MODEL_TYPE")"
fi

if (( RUN_TRAIN )) && [[ "$MODEL_TYPE" == "base" ]]; then
  echo "--train is not supported for model_type=base" >&2
  exit 1
fi

if (( RESUME_TRAIN )) && [[ "$MODEL_TYPE" == "base" ]]; then
  echo "--resume is not supported for model_type=base" >&2
  exit 1
fi

if (( RESUME_TRAIN )) && [[ "$MODEL_TYPE" != "base" && -z "$CHECKPOINT" && -n "$CHECKPOINT_DIR" ]]; then
  CHECKPOINT="$(experiment_latest_checkpoint "$CHECKPOINT_DIR" || true)"
fi

if (( RESUME_TRAIN )) && [[ "$MODEL_TYPE" != "base" && -z "$CHECKPOINT" ]]; then
  if (( DRY_RUN )); then
    CHECKPOINT="${CHECKPOINT_DIR:-outputs/checkpoints}/dry_run_checkpoint.safetensors"
  else
    echo "--resume requires an existing checkpoint or a populated --checkpoint-dir." >&2
    exit 1
  fi
fi

if (( RESUME_TRAIN )) && [[ "$MODEL_TYPE" != "base" ]]; then
  experiment_require_file "$CHECKPOINT"
fi

if (( RUN_TRAIN )); then
  train_script="$(train_script_for_type "$MODEL_TYPE")"
  train_cmd=(bash "$train_script")
  if (( DRY_RUN )); then
    train_cmd+=(--dry-run)
  fi
  if (( RESUME_TRAIN )); then
    case "$MODEL_TYPE" in
      lora)
        train_cmd+=(--set "lora.lora_checkpoint=$CHECKPOINT")
        ;;
      full)
        if [[ ${#RESUME_ARGS[@]} -eq 0 ]]; then
          echo "Full training resume requires at least one --resume-arg value." >&2
          exit 1
        fi
        resume_json="$("${PYTHON:-python3}" -c 'import json, sys; print(json.dumps(sys.argv[1:]))' "${RESUME_ARGS[@]}")"
        train_cmd+=(--set "resume.enabled=true" --set "resume.extra_args=$resume_json")
        ;;
    esac
  fi
  for override in "${TRAIN_OVERRIDES[@]:-}"; do
    [[ -n "$override" ]] || continue
    train_cmd+=(--set "$override")
  done
  experiment_run "${train_cmd[@]}"
fi

if [[ "$MODEL_TYPE" != "base" ]]; then
  if [[ -z "$CHECKPOINT" && -n "$CHECKPOINT_DIR" ]]; then
    CHECKPOINT="$(experiment_latest_checkpoint "$CHECKPOINT_DIR" || true)"
  fi

  if [[ -z "$CHECKPOINT" ]]; then
    if (( DRY_RUN )); then
      CHECKPOINT="${CHECKPOINT_DIR:-outputs/checkpoints}/dry_run_checkpoint.safetensors"
    else
      echo "No checkpoint found. Use --train or provide --checkpoint." >&2
      exit 1
    fi
  fi

  experiment_require_file "$CHECKPOINT"
fi

if [[ -z "$MODEL_NAME" ]]; then
  if [[ "$MODEL_TYPE" == "base" ]]; then
    MODEL_NAME="qwen_edit_2509_base"
  else
    checkpoint_stem="$(basename "${CHECKPOINT%.safetensors}")"
    MODEL_NAME="qwen_edit_2509_${MODEL_TYPE}_${checkpoint_stem}"
  fi
fi

COMMON_EVAL_ARGS=(
  --set "model.model_type=$MODEL_TYPE"
  --set "model.model_name=$MODEL_NAME"
)
if [[ "$MODEL_TYPE" != "base" ]]; then
  COMMON_EVAL_ARGS+=(--set "model.checkpoint_path=$CHECKPOINT")
fi
for override in "${EVAL_OVERRIDES[@]:-}"; do
  [[ -n "$override" ]] || continue
  COMMON_EVAL_ARGS+=(--set "$override")
done

if (( RUN_VALIDATE )); then
  validate_script="$(validate_script_for_type "$MODEL_TYPE")"
  if [[ -n "$validate_script" ]]; then
    validate_cmd=(bash "$validate_script" "$CHECKPOINT")
    for override in "${EVAL_OVERRIDES[@]:-}"; do
      [[ -n "$override" ]] || continue
      validate_cmd+=(--set "$override")
    done
    experiment_run "${validate_cmd[@]}"
  else
    echo "Skipping validation for model_type=$MODEL_TYPE because no checkpoint-based validator is defined."
  fi
fi

if (( RUN_EXPORT )); then
  gedit_cmd=(bash scripts/export_gedit.sh "${COMMON_EVAL_ARGS[@]}")
  imgedit_cmd=(bash scripts/export_imgedit.sh "${COMMON_EVAL_ARGS[@]}")
  if [[ -n "$LIMIT" ]]; then
    gedit_cmd+=(--limit "$LIMIT")
    imgedit_cmd+=(--limit "$LIMIT")
  fi
  experiment_run "${gedit_cmd[@]}"
  experiment_run "${imgedit_cmd[@]}"
fi

if (( RUN_SCORE )); then
  experiment_run bash scripts/score_gedit.sh "${COMMON_EVAL_ARGS[@]}"
  experiment_run bash scripts/score_imgedit.sh "${COMMON_EVAL_ARGS[@]}"
fi

echo "Edit model suite finished for model_type=$MODEL_TYPE model_name=$MODEL_NAME${CHECKPOINT:+ checkpoint=$CHECKPOINT}"
