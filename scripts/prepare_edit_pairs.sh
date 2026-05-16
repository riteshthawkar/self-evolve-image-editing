#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

usage() {
  cat <<'EOF'
Usage: bash scripts/prepare_edit_pairs.sh [options]

Downloads a Hugging Face image-editing dataset, filters source-target edit pairs,
stores source/target images, and writes a DiffSynth-compatible training manifest.

Default preset:
  MagicBrush train split: osunlp/MagicBrush

Options:
  --config PATH              YAML config. Default: configs/data/edit_pairs_magicbrush.yaml
  --limit N                  Max dataset rows to scan. Default: full config/dataset split
  --max-selected N           Keep top N accepted pairs after scoring. Default: config value
  --output-root PATH         Root for saved source/target images and score files
  --manifest PATH            DiffSynth manifest JSON output path
  --dataset-path ID          Override HF dataset path
  --dataset-name NAME        Override optional HF dataset name/config
  --dataset-split SPLIT      Override HF dataset split. Default from config
  --source-column NAME       Override source image column
  --target-column NAME       Override target image column
  --instruction-column NAME  Override instruction column
  --id-column NAME           Override stable id column
  --turn-column NAME         Override turn column
  --cache-dir PATH           HF datasets cache directory
  --min-total-score X        Filter threshold for total score
  --min-changed-fraction X   Reject invisible edits below this change fraction
  --max-changed-fraction X   Reject edits that change too much
  --progress-every N         Print progress every N processed rows
  --no-resume                Rebuild outputs from scratch
  --set dotted.key=value     Pass any extra config override. Can repeat.
  -h, --help                 Show this message.

Examples:
  bash scripts/prepare_edit_pairs.sh --limit 1000 --max-selected 800

  bash scripts/prepare_edit_pairs.sh \
    --limit 0 \
    --output-root data/edit_pairs/magicbrush_full \
    --manifest data/manifests/magicbrush_full_filtered.json
EOF
}

CONFIG="configs/data/edit_pairs_magicbrush.yaml"
LIMIT=""
OVERRIDES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift ;;
    --limit) LIMIT="$2"; shift ;;
    --max-selected) OVERRIDES+=(--set "output.max_selected=$2"); shift ;;
    --output-root)
      OUT="${2%/}"
      OVERRIDES+=(--set "output.root_dir=$OUT")
      OVERRIDES+=(--set "output.all_records_jsonl=$OUT/all_records.jsonl")
      OVERRIDES+=(--set "output.selected_records_jsonl=$OUT/selected_records.jsonl")
      OVERRIDES+=(--set "output.rejected_records_jsonl=$OUT/rejected_records.jsonl")
      OVERRIDES+=(--set "output.summary_json=$OUT/summary.json")
      shift
      ;;
    --manifest) OVERRIDES+=(--set "output.manifest_json=$2"); shift ;;
    --dataset-path) OVERRIDES+=(--set "dataset.path=$2"); shift ;;
    --dataset-name) OVERRIDES+=(--set "dataset.name=$2"); shift ;;
    --dataset-split) OVERRIDES+=(--set "dataset.split=$2"); shift ;;
    --source-column) OVERRIDES+=(--set "columns.source_image=$2"); shift ;;
    --target-column) OVERRIDES+=(--set "columns.target_image=$2"); shift ;;
    --instruction-column) OVERRIDES+=(--set "columns.instruction=$2"); shift ;;
    --id-column) OVERRIDES+=(--set "columns.id=$2"); shift ;;
    --turn-column) OVERRIDES+=(--set "columns.turn=$2"); shift ;;
    --cache-dir) OVERRIDES+=(--set "dataset.cache_dir=$2"); shift ;;
    --min-total-score) OVERRIDES+=(--set "filters.min_total_score=$2"); shift ;;
    --min-changed-fraction) OVERRIDES+=(--set "filters.min_changed_fraction=$2"); shift ;;
    --max-changed-fraction) OVERRIDES+=(--set "filters.max_changed_fraction=$2"); shift ;;
    --progress-every) OVERRIDES+=(--set "output.progress_every=$2"); shift ;;
    --no-resume) OVERRIDES+=(--set "output.resume=false");;
    --set) OVERRIDES+=(--set "$2"); shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
  shift
done

cmd=("${PYTHON:-python3}" -m qwen_edit_project.data.prepare_edit_pairs --config "$CONFIG")
if [[ -n "$LIMIT" ]]; then
  cmd+=(--limit "$LIMIT")
fi
cmd+=("${OVERRIDES[@]}")
"${cmd[@]}"
