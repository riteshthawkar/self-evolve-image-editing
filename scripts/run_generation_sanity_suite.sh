#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
. "$ROOT/scripts/experiment_common.sh"

usage() {
  cat <<'EOF'
Usage: bash scripts/run_generation_sanity_suite.sh [options]

Options:
  --model-type TYPE             base, lora, or full. Default: base
  --checkpoint PATH             Optional checkpoint used for lora or full evaluation.
  --model-name NAME             Override benchmark model_name.
  --limit N                     Limit prompt count for exports.
  --skip-export                 Skip benchmark export.
  --skip-score                  Skip benchmark scoring.
  --dry-run                     Print the composed commands without executing them.
  --set-eval KEY=VALUE          Extra evaluation override. Repeatable.
  -h, --help                    Show this message.

Examples:
  bash scripts/run_generation_sanity_suite.sh --limit 32
  bash scripts/run_generation_sanity_suite.sh --model-type full --checkpoint outputs/checkpoints/qwen_image_full/latest.safetensors --model-name qwen_image_full_v2
EOF
}

MODEL_TYPE="base"
CHECKPOINT=""
MODEL_NAME=""
LIMIT=""
RUN_EXPORT=1
RUN_SCORE=1
DRY_RUN=0
EVAL_OVERRIDES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-type)
      MODEL_TYPE="$2"
      shift
      ;;
    --checkpoint)
      CHECKPOINT="$2"
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
    --skip-export)
      RUN_EXPORT=0
      ;;
    --skip-score)
      RUN_SCORE=0
      ;;
    --dry-run)
      DRY_RUN=1
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

if [[ "$MODEL_TYPE" != "base" && -z "$CHECKPOINT" ]]; then
  if (( DRY_RUN )); then
    CHECKPOINT="outputs/checkpoints/${MODEL_TYPE}/dry_run_checkpoint.safetensors"
  else
    echo "--checkpoint is required when --model-type is lora or full." >&2
    exit 1
  fi
fi

if [[ -n "$CHECKPOINT" ]]; then
  experiment_require_file "$CHECKPOINT"
fi

if [[ -z "$MODEL_NAME" ]]; then
  if [[ "$MODEL_TYPE" == "base" ]]; then
    MODEL_NAME="qwen_image_base_experiment"
  else
    checkpoint_stem="$(basename "${CHECKPOINT%.safetensors}")"
    MODEL_NAME="qwen_image_${MODEL_TYPE}_${checkpoint_stem}"
  fi
fi

COMMON_ARGS=(
  --set "model.model_type=$MODEL_TYPE"
  --set "model.model_name=$MODEL_NAME"
)
if [[ -n "$CHECKPOINT" ]]; then
  COMMON_ARGS+=(--set "model.checkpoint_path=$CHECKPOINT")
fi
for override in "${EVAL_OVERRIDES[@]:-}"; do
  [[ -n "$override" ]] || continue
  COMMON_ARGS+=(--set "$override")
done

if (( RUN_EXPORT )); then
  geneval_cmd=(bash scripts/export_geneval.sh "${COMMON_ARGS[@]}")
  dpg_cmd=(bash scripts/export_dpgbench.sh "${COMMON_ARGS[@]}")
  oneig_cmd=(bash scripts/export_oneig_bench.sh "${COMMON_ARGS[@]}")
  if [[ -n "$LIMIT" ]]; then
    geneval_cmd+=(--limit "$LIMIT")
    dpg_cmd+=(--limit "$LIMIT")
    oneig_cmd+=(--limit "$LIMIT")
  fi
  experiment_run "${geneval_cmd[@]}"
  experiment_run "${dpg_cmd[@]}"
  experiment_run "${oneig_cmd[@]}"
fi

if (( RUN_SCORE )); then
  experiment_run bash scripts/score_geneval.sh "${COMMON_ARGS[@]}"
  experiment_run bash scripts/score_dpgbench.sh "${COMMON_ARGS[@]}"
  experiment_run bash scripts/score_oneig_bench.sh "${COMMON_ARGS[@]}"
fi

echo "Generation sanity suite finished for model_name=$MODEL_NAME"
