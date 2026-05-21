#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
. "$ROOT/scripts/experiment_common.sh"

usage() {
  cat <<'EOF'
Usage: bash scripts/run_self_evolve_matrix.sh [options]

Options:
  --variant NAME                base, internal-cepr, naive-self-train, evolmm-style, spatial, cycle, internal, hybrid, hybrid-scalar, delta-ranker, delta-ranker-proxy, delta-grounded, delta-results, pillow-demo, pillow-hybrid, pillow-delta-ranker, or all. Default: all
  --limit N                     Limit number of unlabeled records.
  --images-dir PATH             Override dataset.images_dir.
  --metadata-jsonl PATH         Optional sidecar metadata for directory datasets.
  --output-prefix PATH          Override output roots to PATH/<variant>.
  --dtype DTYPE                 Override editor.model.torch_dtype, for example float16.
  --checkpoint PATH             Starting checkpoint for qwen-based self-evolve variants.
  --checkpoint-dir PATH         Auto-discover the latest checkpoint from this directory.
  --editor-model-type TYPE      Override editor.model.model_type, for example lora or full.
  --launch-training             Set training.trigger=launch.
  --train-config PATH           Override training.base_train_config.
  --dry-run                     Dry-run the selected self-evolve runs.
  --set KEY=VALUE               Extra override passed through to every run. Repeatable.
  -h, --help                    Show this message.

Examples:
  bash scripts/run_self_evolve_matrix.sh --variant hybrid --limit 64 --images-dir data/unlabeled/self_evolve
  bash scripts/run_self_evolve_matrix.sh --variant hybrid --checkpoint-dir outputs/checkpoints/Qwen-Image-Edit-2509_lora --editor-model-type lora
  bash scripts/run_self_evolve_matrix.sh --variant all --limit 32 --dtype float16 --output-prefix outputs/self_evolve/ablation_01
EOF
}

VARIANT="all"
LIMIT=""
IMAGES_DIR=""
METADATA_JSONL=""
OUTPUT_PREFIX=""
DTYPE=""
CHECKPOINT=""
CHECKPOINT_DIR=""
EDITOR_MODEL_TYPE=""
DRY_RUN=0
LAUNCH_TRAINING=0
TRAIN_CONFIG=""
EXTRA_OVERRIDES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --variant)
      VARIANT="$2"
      shift
      ;;
    --limit)
      LIMIT="$2"
      shift
      ;;
    --images-dir)
      IMAGES_DIR="$2"
      shift
      ;;
    --metadata-jsonl)
      METADATA_JSONL="$2"
      shift
      ;;
    --output-prefix)
      OUTPUT_PREFIX="$2"
      shift
      ;;
    --dtype)
      DTYPE="$2"
      shift
      ;;
    --checkpoint)
      CHECKPOINT="$2"
      shift
      ;;
    --checkpoint-dir)
      CHECKPOINT_DIR="$2"
      shift
      ;;
    --editor-model-type)
      EDITOR_MODEL_TYPE="$2"
      shift
      ;;
    --launch-training)
      LAUNCH_TRAINING=1
      ;;
    --train-config)
      TRAIN_CONFIG="$2"
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      ;;
    --set)
      EXTRA_OVERRIDES+=("$2")
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

if [[ -z "$CHECKPOINT" && -n "$CHECKPOINT_DIR" ]]; then
  CHECKPOINT="$(experiment_latest_checkpoint "$CHECKPOINT_DIR" || true)"
fi

if [[ -n "$CHECKPOINT" ]]; then
  experiment_require_file "$CHECKPOINT"
fi

variant_script() {
  case "$1" in
    base) echo "scripts/self_evolve_2509.sh" ;;
    internal-cepr) echo "scripts/self_evolve_2509_internal_cepr.sh" ;;
    naive-self-train) echo "scripts/self_evolve_2509.sh" ;;
    evolmm-style) echo "scripts/self_evolve_2509_evolmm_style.sh" ;;
    spatial) echo "scripts/self_evolve_2509_spatial.sh" ;;
    cycle) echo "scripts/self_evolve_2509_cycle.sh" ;;
    internal) echo "scripts/self_evolve_2509_internal.sh" ;;
    hybrid) echo "scripts/self_evolve_2509_hybrid.sh" ;;
    hybrid-scalar) echo "scripts/self_evolve_2509_hybrid.sh" ;;
    delta-ranker) echo "scripts/self_evolve_2509_delta_ranker.sh" ;;
    delta-ranker-proxy) echo "scripts/self_evolve_2509_delta_ranker.sh" ;;
    delta-grounded) echo "scripts/self_evolve_2509_delta_grounded.sh" ;;
    delta-results) echo "scripts/self_evolve_2509_delta_results.sh" ;;
    pillow-demo) echo "scripts/self_evolve_pillow_demo.sh" ;;
    pillow-hybrid) echo "scripts/self_evolve_pillow_hybrid.sh" ;;
    pillow-delta-ranker) echo "scripts/self_evolve_pillow_delta_ranker.sh" ;;
    *) return 1 ;;
  esac
}

if [[ "$VARIANT" == "all" ]]; then
  variants=(base internal-cepr naive-self-train evolmm-style spatial cycle internal hybrid delta-ranker delta-grounded delta-results)
else
  variants=("$VARIANT")
fi

for current_variant in "${variants[@]}"; do
  run_script="$(variant_script "$current_variant" || true)"
  if [[ -z "$run_script" ]]; then
    echo "Unsupported variant: $current_variant" >&2
    exit 1
  fi

  cmd=(bash "$run_script")
  if (( DRY_RUN )); then
    cmd+=(--dry-run)
  fi
  if [[ -n "$LIMIT" ]]; then
    cmd+=(--limit "$LIMIT")
  fi
  if [[ -n "$IMAGES_DIR" ]]; then
    cmd+=(--set "dataset.images_dir=$IMAGES_DIR")
  fi
  if [[ -n "$METADATA_JSONL" ]]; then
    cmd+=(--set "dataset.metadata_jsonl=$METADATA_JSONL")
  fi
  if [[ -n "$OUTPUT_PREFIX" ]]; then
    cmd+=(--set "output.root_dir=$OUTPUT_PREFIX/$current_variant")
  fi
  case "$current_variant" in
    naive-self-train)
      cmd+=(--set "candidate_generation.samples_per_proposal=1")
      cmd+=(--set "solver.acceptance_threshold=0.60")
      cmd+=(--set "curriculum.proposals_per_image=1")
      ;;
    hybrid-scalar)
      cmd+=(--set "solver.cycle_weight=0.0")
      cmd+=(--set "solver.internal_weight=0.0")
      cmd+=(--set "output.write_evaluator_training=false")
      ;;
    delta-ranker-proxy)
      cmd+=(--set "solver.internal_weight=0.0")
      cmd+=(--set "solver.rank_internal_weight=0.0")
      cmd+=(--set "solver.require_internal_when_weighted=false")
      ;;
  esac
  if [[ -n "$DTYPE" ]]; then
    cmd+=(--set "editor.model.torch_dtype=$DTYPE")
  fi
  if [[ -n "$CHECKPOINT" ]]; then
    cmd+=(--set "editor.model.checkpoint_path=$CHECKPOINT")
    cmd+=(--set "training.current_checkpoint_path=$CHECKPOINT")
  fi
  if [[ -n "$EDITOR_MODEL_TYPE" ]]; then
    cmd+=(--set "editor.model.model_type=$EDITOR_MODEL_TYPE")
  fi
  if (( LAUNCH_TRAINING )); then
    cmd+=(--set "training.trigger=launch")
  fi
  if [[ -n "$TRAIN_CONFIG" ]]; then
    cmd+=(--set "training.base_train_config=$TRAIN_CONFIG")
  fi
  for override in "${EXTRA_OVERRIDES[@]:-}"; do
    [[ -n "$override" ]] || continue
    cmd+=(--set "$override")
  done

  experiment_run "${cmd[@]}"
done

echo "Self-evolve experiment matrix finished for variant=$VARIANT"
