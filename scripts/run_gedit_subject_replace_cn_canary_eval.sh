#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -z "${SLURM_JOB_ID:-}" && "${ALLOW_LOGIN_NODE:-0}" != "1" ]]; then
  echo "Refusing to run GEdit canary export/scoring outside a Slurm allocation. Start this inside an srun/sbatch resource session." >&2
  exit 1
fi

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${PYTHON:-/share_6/users/ritesh_thawkar/condaenvs/qedit/bin/python}"

usage() {
  cat <<'EOF'
Usage: bash scripts/run_gedit_subject_replace_cn_canary_eval.sh --checkpoint PATH --model-name NAME [options]

Exports and scores a small GEdit subject-replace/cn slice, then compares it
against the existing full baseline CSV on matching keys. The upstream GEdit
statistics script can fail for partial slices; this wrapper still keeps and
compares the raw subject-replace score CSV.

Options:
  --checkpoint PATH    Required LoRA checkpoint.
  --model-name NAME    Required candidate model name.
  --limit N            Number of examples. Default: 32
  --offset N           Starting offset. Default: 0
  --steps N            Inference steps. Default: 40
  --device DEVICE      Export device. Default: cuda
EOF
}

CHECKPOINT=""
MODEL_NAME=""
LIMIT=32
OFFSET=0
STEPS=40
DEVICE="cuda"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint) CHECKPOINT="$2"; shift ;;
    --model-name) MODEL_NAME="$2"; shift ;;
    --limit) LIMIT="$2"; shift ;;
    --offset) OFFSET="$2"; shift ;;
    --steps) STEPS="$2"; shift ;;
    --device) DEVICE="$2"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
  shift
done

if [[ -z "$CHECKPOINT" || -z "$MODEL_NAME" ]]; then
  usage >&2
  exit 1
fi
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Checkpoint not found: $CHECKPOINT" >&2
  exit 1
fi

BASELINE_NAME="qwen_edit_2509_baseline_gedit"
GROUP="subject-replace"
LANGUAGE="cn"
SCORE_DIR="outputs/scores/gedit"
SUMMARY_PATH="${SCORE_DIR}/${MODEL_NAME}_summary.json"
LOG_STEM="outputs/logs/${MODEL_NAME}_gedit_subject_replace_cn_n${LIMIT}_o${OFFSET}"
BASELINE_CSV="${SCORE_DIR}/${BASELINE_NAME}/gpt4o/${BASELINE_NAME}_${GROUP}_all_vie_score.csv"
CANDIDATE_CSV="${SCORE_DIR}/${MODEL_NAME}/gpt4o/${MODEL_NAME}_${GROUP}_all_vie_score.csv"
COMPARISON_JSON="outputs/quick_eval/gedit_subject_replace_cn_n${LIMIT}/${MODEL_NAME}_vs_baseline_subject_replace_cn_n${LIMIT}_comparison.json"

mkdir -p outputs/logs

echo "Exporting GEdit ${GROUP}/${LANGUAGE}: model=${MODEL_NAME} limit=${LIMIT} offset=${OFFSET} steps=${STEPS}"
bash scripts/export_gedit.sh \
  --device "$DEVICE" \
  --limit "$LIMIT" \
  --offset "$OFFSET" \
  --set "model.backend=official_diffusers" \
  --set "model.model_type=lora" \
  --set "model.model_name=${MODEL_NAME}" \
  --set "model.checkpoint_path=${CHECKPOINT}" \
  --set "generation.num_inference_steps=${STEPS}" \
  --set "dataset.task_type=${GROUP}" \
  --set "dataset.instruction_language=${LANGUAGE}" \
  --set "output.summary_path=${SUMMARY_PATH}" \
  2>&1 | tee "${LOG_STEM}_export.log"

echo "Scoring GEdit ${GROUP}/${LANGUAGE}: model=${MODEL_NAME}"
set +e
bash scripts/score_gedit.sh \
  --set "model.model_name=${MODEL_NAME}" \
  --set "dataset.task_type=${GROUP}" \
  --set "dataset.instruction_language=${LANGUAGE}" \
  --set "scoring.allow_partial=true" \
  --set "output.summary_path=${SUMMARY_PATH}" \
  2>&1 | tee "${LOG_STEM}_score.log"
score_status=${PIPESTATUS[0]}
set -e

if [[ ! -f "$CANDIDATE_CSV" ]]; then
  echo "Candidate CSV was not produced: $CANDIDATE_CSV" >&2
  exit "$score_status"
fi

"$PYTHON" scripts/compare_gedit_quick_csv.py \
  --baseline-csv "$BASELINE_CSV" \
  --candidate-csv "$CANDIDATE_CSV" \
  --baseline-name "$BASELINE_NAME" \
  --candidate-name "$MODEL_NAME" \
  --group "$GROUP" \
  --language "$LANGUAGE" \
  --output "$COMPARISON_JSON"

echo "GEdit canary comparison written: $COMPARISON_JSON"
