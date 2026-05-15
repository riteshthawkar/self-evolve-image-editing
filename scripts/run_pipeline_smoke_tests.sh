#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

usage() {
  cat <<'EOF'
Usage: bash scripts/run_pipeline_smoke_tests.sh [options]

Options:
  --keep-artifacts              Keep temporary smoke-test images and outputs.
  -h, --help                    Show this message.
EOF
}

KEEP_ARTIFACTS=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --keep-artifacts)
      KEEP_ARTIFACTS=1
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

TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/qwen_edit_smoke.XXXXXX")"
TMP_IMAGES="$TMP_ROOT/images"
TMP_OUTPUT="$TMP_ROOT/outputs"
mkdir -p "$TMP_IMAGES" "$TMP_OUTPUT"

cleanup() {
  if (( KEEP_ARTIFACTS )); then
    echo "Smoke test artifacts kept at $TMP_ROOT"
    return
  fi
  rm -rf "$TMP_ROOT"
}
trap cleanup EXIT

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

echo "[1/6] Compile Python sources"
"${PYTHON:-python3}" -m compileall src

echo "[2/6] Check shell syntax"
bash -n scripts/*.sh
find scripts/slurm -name '*.sbatch' -print0 | xargs -0 -n1 bash -n

echo "[3/6] Dry-run edit suite commands"
bash scripts/run_edit_model_suite.sh --model-type base --dry-run --limit 2
bash scripts/run_edit_model_suite.sh --model-type lora --train --dry-run --limit 2
bash scripts/run_edit_model_suite.sh --model-type lora --train --resume --dry-run --limit 2
bash scripts/run_edit_model_suite.sh --model-type full --train --resume --dry-run --resume-arg --resume_from_checkpoint --resume-arg outputs/checkpoints/Qwen-Image-Edit-2509_full/dry_run_checkpoint.safetensors --limit 2
bash scripts/run_edit_model_suite.sh --model-type full --train --dry-run --limit 2

echo "[4/6] Dry-run generation suite commands"
bash scripts/run_generation_sanity_suite.sh --dry-run --limit 2
bash scripts/run_generation_sanity_suite.sh --model-type full --dry-run --limit 2

echo "[5/6] Create temporary self-evolve demo images"
"${PYTHON:-python3}" - "$TMP_IMAGES" <<'PY'
from pathlib import Path
import sys

from PIL import Image, ImageDraw

root = Path(sys.argv[1])
root.mkdir(parents=True, exist_ok=True)
for index, color in enumerate(((220, 60, 60), (60, 160, 220), (80, 180, 90)), start=1):
    image = Image.new("RGB", (96, 96), color)
    draw = ImageDraw.Draw(image)
    draw.rectangle((12, 12, 84, 84), outline=(255, 255, 255), width=3)
    draw.text((20, 38), f"S{index}", fill=(255, 255, 255))
    image.save(root / f"sample_{index}.png")
PY

echo "[6/6] Run lightweight source-selection and self-evolve smoke path"
"${PYTHON:-python3}" -m qwen_edit_project.data.select_unlabeled_images \
  --config configs/data/source_image_filter_heuristic.yaml \
  --set input.images_dir="$TMP_IMAGES" \
  --set output.selected_manifest_jsonl="$TMP_OUTPUT/selected/manifest.jsonl" \
  --set output.rejected_manifest_jsonl="$TMP_OUTPUT/selected/rejected.jsonl" \
  --set output.score_jsonl="$TMP_OUTPUT/selected/scores.jsonl" \
  --set output.summary_json="$TMP_OUTPUT/selected/summary.json" \
  --set selection.thresholds.min_short_side=64 \
  --set selection.thresholds.min_total_score=0.20 \
  --set selection.thresholds.min_technical_quality=0.10 \
  --set selection.thresholds.min_editable_content=0.10 \
  --set selection.thresholds.min_preservation_potential=0.10 \
  --limit 3

"${PYTHON:-python3}" -m qwen_edit_project.data.split_source_manifest \
  --input "$TMP_OUTPUT/selected/manifest.jsonl" \
  --output-dir "$TMP_OUTPUT/splits" \
  --pilot-count 1 \
  --main-count 1 \
  --heldout-count 1

bash scripts/run_self_evolve_matrix.sh \
  --variant hybrid \
  --dry-run \
  --limit 2 \
  --images-dir "$TMP_IMAGES" \
  --checkpoint "$TMP_IMAGES/sample_1.png" \
  --editor-model-type lora \
  --output-prefix "$TMP_OUTPUT/self_evolve_dry_run"

bash scripts/run_self_evolve_matrix.sh \
  --variant pillow-hybrid \
  --limit 3 \
  --images-dir "$TMP_IMAGES" \
  --output-prefix "$TMP_OUTPUT/self_evolve"

bash scripts/run_self_evolve_matrix.sh \
  --variant pillow-delta-ranker \
  --limit 3 \
  --images-dir "$TMP_IMAGES" \
  --output-prefix "$TMP_OUTPUT/self_evolve"

echo "Pipeline smoke tests passed."
