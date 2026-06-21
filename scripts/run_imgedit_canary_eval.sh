#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -z "${SLURM_JOB_ID:-}" && "${ALLOW_LOGIN_NODE:-0}" != "1" ]]; then
  echo "Refusing to run ImgEdit canary export/scoring outside a Slurm allocation. Start this inside an srun/sbatch resource session." >&2
  exit 1
fi

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${PYTHON:-/share_6/users/ritesh_thawkar/condaenvs/qedit/bin/python}"

usage() {
  cat <<'EOF'
Usage: bash scripts/run_imgedit_canary_eval.sh --checkpoint PATH [options]

Runs a small ImgEdit canary evaluation without touching the full benchmark outputs.
The script reuses already exported baseline images for the same keys, exports the
candidate checkpoint on the subset, optionally scores both folders, and writes a
comparison JSON.

Options:
  --checkpoint PATH        Required LoRA checkpoint to evaluate.
  --model-name NAME        Candidate model name. Default: rubric_cepr_v1_64_r4_canary
  --baseline-name NAME     Existing full baseline image folder. Default: qwen_edit_2509_baseline_imgedit
  --limit N               Number of ImgEdit records. Default: 32
  --offset N              Starting record offset. Default: 0
  --steps N               Inference steps. Default: 40
  --lora-scale X          Optional global LoRA scale for inference
  --device DEVICE         Export device. Default: cuda
  --no-score              Export images only; skip OpenAI scorer.
  -h, --help              Show this message.
EOF
}

CHECKPOINT=""
MODEL_NAME="rubric_cepr_v1_64_r4_canary"
BASELINE_NAME="qwen_edit_2509_baseline_imgedit"
LIMIT=32
OFFSET=0
STEPS=40
LORA_SCALE=""
DEVICE="cuda"
RUN_SCORE=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint) CHECKPOINT="$2"; shift ;;
    --model-name) MODEL_NAME="$2"; shift ;;
    --baseline-name) BASELINE_NAME="$2"; shift ;;
    --limit) LIMIT="$2"; shift ;;
    --offset) OFFSET="$2"; shift ;;
    --steps) STEPS="$2"; shift ;;
    --lora-scale) LORA_SCALE="$2"; shift ;;
    --device) DEVICE="$2"; shift ;;
    --no-score) RUN_SCORE=0 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
  shift
done

if [[ -z "$CHECKPOINT" ]]; then
  echo "--checkpoint is required" >&2
  exit 1
fi
if [[ ! -f "$CHECKPOINT" && ! -f "$CHECKPOINT/pytorch_lora_weights.safetensors" ]]; then
  echo "Checkpoint not found: $CHECKPOINT" >&2
  exit 1
fi

CANARY_ROOT="outputs/quick_eval/imgedit_canary_o${OFFSET}_n${LIMIT}"
IMAGE_ROOT="$CANARY_ROOT/images"
SCORE_ROOT="$CANARY_ROOT/scores"
SUBSET_JSON="$CANARY_ROOT/imgedit_subset.json"
COMPARISON_JSON="$CANARY_ROOT/${MODEL_NAME}_vs_baseline_comparison.json"
BASELINE_CANARY="${BASELINE_NAME}_canary_o${OFFSET}_n${LIMIT}"

mkdir -p "$CANARY_ROOT" "$IMAGE_ROOT" "$SCORE_ROOT" outputs/logs

"$PYTHON" - "$LIMIT" "$OFFSET" "$BASELINE_NAME" "$BASELINE_CANARY" "$SUBSET_JSON" "$IMAGE_ROOT" <<'PY'
import json
import os
import shutil
import sys
from pathlib import Path

limit = int(sys.argv[1])
offset = int(sys.argv[2])
baseline_name = sys.argv[3]
baseline_canary = sys.argv[4]
subset_json = Path(sys.argv[5])
image_root = Path(sys.argv[6])

source_json = Path("data/processed/benchmark/imgedit/basic_edit.json")
source_images = Path("outputs/benchmark_images/imgedit") / baseline_name
target_images = image_root / baseline_canary

if not source_json.exists():
    raise FileNotFoundError(source_json)
if not source_images.exists():
    raise FileNotFoundError(source_images)

data = json.loads(source_json.read_text(encoding="utf-8"))
items = list(data.items())[offset : offset + limit]
subset = dict(items)
subset_json.parent.mkdir(parents=True, exist_ok=True)
subset_json.write_text(json.dumps(subset, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

target_images.mkdir(parents=True, exist_ok=True)
missing = []
for key, _ in items:
    src = source_images / f"{key}.png"
    dst = target_images / f"{key}.png"
    if not src.exists():
        missing.append(str(src))
        continue
    if dst.exists():
        continue
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)

if missing:
    raise FileNotFoundError("Missing baseline PNGs for canary subset:\n" + "\n".join(missing[:20]))

print(f"Prepared ImgEdit canary subset: {len(items)} keys -> {subset_json}")
print(f"Prepared baseline canary images: {target_images}")
PY

echo "Exporting candidate ImgEdit canary images: model=${MODEL_NAME} limit=${LIMIT} steps=${STEPS}"
bash scripts/export_imgedit.sh \
  --device "$DEVICE" \
  --limit "$LIMIT" \
  --offset 0 \
  --set "model.model_type=lora" \
  --set "model.model_name=$MODEL_NAME" \
  --set "model.checkpoint_path=$CHECKPOINT" \
  --set "model.backend=official_diffusers" \
  --set "generation.num_inference_steps=$STEPS" \
  --set "output.edited_images_dir=$IMAGE_ROOT" \
  --set "output.scores_dir=$SCORE_ROOT" \
  --set "output.summary_path=$SCORE_ROOT/${MODEL_NAME}_summary.json" \
  --set "dataset.edit_json=$SUBSET_JSON" \
  ${LORA_SCALE:+--set "model.lora_scale=$LORA_SCALE"}

if (( RUN_SCORE )); then
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    if [[ -f secret.env ]]; then
      secret_text="$(tr -d '\r\n' < secret.env)"
      if [[ "$secret_text" == OPENAI_API_KEY=* ]]; then
        export OPENAI_API_KEY="${secret_text#OPENAI_API_KEY=}"
      else
        export OPENAI_API_KEY="$secret_text"
      fi
    else
      echo "OPENAI_API_KEY is not set and secret.env is missing; skipping scoring." >&2
      RUN_SCORE=0
    fi
  fi
fi

if (( RUN_SCORE )); then
  for name in "$BASELINE_CANARY" "$MODEL_NAME"; do
    echo "Scoring ImgEdit canary: $name"
    bash scripts/score_imgedit.sh \
      --set "model.model_name=$name" \
      --set "output.edited_images_dir=$IMAGE_ROOT" \
      --set "output.scores_dir=$SCORE_ROOT" \
      --set "output.summary_path=$SCORE_ROOT/${name}_summary.json" \
      --set "dataset.edit_json=$SUBSET_JSON" \
      --set "scoring.num_processes=4" \
      --set "scoring.retry_num_processes=1" \
      --set "scoring.max_retry_rounds=2" \
      --set "scoring.allow_partial=true"
  done

  "$PYTHON" - "$SCORE_ROOT" "$BASELINE_CANARY" "$MODEL_NAME" "$COMPARISON_JSON" <<'PY'
import json
import statistics
import sys
from pathlib import Path

score_root = Path(sys.argv[1])
baseline = sys.argv[2]
model = sys.argv[3]
out_path = Path(sys.argv[4])

def load_scores(name: str) -> dict[str, float]:
    path = score_root / f"{name}_average_score.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(k): float(v) for k, v in data.items()}

base = load_scores(baseline)
candidate = load_scores(model)
common = sorted(set(base) & set(candidate), key=lambda x: int(x) if x.isdigit() else x)
deltas = [candidate[k] - base[k] for k in common]
comparison = {
    "baseline": baseline,
    "candidate": model,
    "count": len(common),
    "baseline_mean": statistics.mean(base[k] for k in common) if common else None,
    "candidate_mean": statistics.mean(candidate[k] for k in common) if common else None,
    "mean_delta": statistics.mean(deltas) if deltas else None,
    "wins": sum(1 for d in deltas if d > 0),
    "ties": sum(1 for d in deltas if d == 0),
    "losses": sum(1 for d in deltas if d < 0),
    "per_key": [
        {
            "key": key,
            "baseline": base[key],
            "candidate": candidate[key],
            "delta": candidate[key] - base[key],
        }
        for key in common
    ],
}
out_path.write_text(json.dumps(comparison, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
print(json.dumps({k: comparison[k] for k in comparison if k != "per_key"}, indent=2))
PY
fi

echo "ImgEdit canary complete: $CANARY_ROOT"
