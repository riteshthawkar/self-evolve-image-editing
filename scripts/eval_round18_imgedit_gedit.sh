#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/share_6/users/ritesh_thawkar/self-evolve-image-editing}"
RUN_DIR="${RUN_DIR:-$ROOT/outputs/self_evolve/qwen_edit_2509_conservative_pairwise_full_v1_20260604T183826}"
CHECKPOINT="${CHECKPOINT:-$RUN_DIR/round_18/training_output/pytorch_lora_weights.safetensors}"
CONDA_ENV="${CONDA_ENV:-/share_6/users/ritesh_thawkar/condaenvs/qedit}"
DEVICE="${DEVICE:-cuda}"
STEPS="${STEPS:-40}"
IMGEDIT_MODEL="${IMGEDIT_MODEL:-self_evolve_pairwise_r18_imgedit}"
GEDIT_MODEL="${GEDIT_MODEL:-self_evolve_pairwise_r18_gedit}"
GEDIT_SHARD_SIZE="${GEDIT_SHARD_SIZE:-128}"
GEDIT_TOTAL="${GEDIT_TOTAL:-1212}"

if [[ -z "${SLURM_JOB_ID:-}" && "${ALLOW_LOGIN_NODE:-0}" != "1" ]]; then
  echo "[eval-r18] Refusing to run eval outside a Slurm allocation."
  exit 2
fi

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "[eval-r18] Missing checkpoint: $CHECKPOINT"
  exit 1
fi

cd "$ROOT"

if [[ -f "$HOME/.bashrc" ]]; then
  # shellcheck disable=SC1090
  source "$HOME/.bashrc" || true
fi
if [[ -d "$CONDA_ENV/bin" ]]; then
  export PATH="$CONDA_ENV/bin:$PATH"
fi

if [[ ! -x "$CONDA_ENV/bin/python" ]]; then
  echo "[eval-r18] Missing qedit python: $CONDA_ENV/bin/python"
  exit 1
fi

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/share_6/users/ritesh_thawkar/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

if [[ -z "${OPENAI_API_KEY:-}" && -f secret.env ]]; then
  secret_text="$(tr -d '\r\n' < secret.env)"
  if [[ "$secret_text" == OPENAI_API_KEY=* ]]; then
    export OPENAI_API_KEY="${secret_text#OPENAI_API_KEY=}"
  else
    export OPENAI_API_KEY="$secret_text"
  fi
fi

mkdir -p outputs/logs outputs/scores/imgedit outputs/scores/gedit

IMGEDIT_LOG="outputs/logs/${IMGEDIT_MODEL}_full_eval.log"
GEDIT_LOG="outputs/logs/${GEDIT_MODEL}_full_eval.log"

echo "[eval-r18] Starting ImgEdit export for $IMGEDIT_MODEL"
bash scripts/export_imgedit.sh \
  --device "$DEVICE" \
  --set "model.backend=official_diffusers" \
  --set "model.model_type=lora" \
  --set "model.model_name=$IMGEDIT_MODEL" \
  --set "model.checkpoint_path=$CHECKPOINT" \
  --set "generation.num_inference_steps=$STEPS" \
  --set "output.edited_images_dir=outputs/benchmark_images/imgedit" \
  --set "output.scores_dir=outputs/scores/imgedit" \
  --set "output.summary_path=outputs/scores/imgedit/${IMGEDIT_MODEL}_export_summary.json" \
  2>&1 | tee -a "$IMGEDIT_LOG"

echo "[eval-r18] Starting ImgEdit scoring for $IMGEDIT_MODEL"
bash scripts/score_imgedit.sh \
  --set "model.model_name=$IMGEDIT_MODEL" \
  --set "output.edited_images_dir=outputs/benchmark_images/imgedit" \
  --set "output.scores_dir=outputs/scores/imgedit" \
  --set "output.summary_path=outputs/scores/imgedit/${IMGEDIT_MODEL}_score_summary.json" \
  --set "scoring.num_processes=4" \
  --set "scoring.retry_num_processes=1" \
  --set "scoring.max_retry_rounds=5" \
  --set "scoring.allow_partial=true" \
  2>&1 | tee -a "$IMGEDIT_LOG"

echo "[eval-r18] Starting GEdit export for $GEDIT_MODEL"
bash scripts/run_gedit_export_shards.sh \
  --model-type lora \
  --model-name "$GEDIT_MODEL" \
  --checkpoint "$CHECKPOINT" \
  --shard-size "$GEDIT_SHARD_SIZE" \
  --total "$GEDIT_TOTAL" \
  --steps "$STEPS" \
  --device "$DEVICE" \
  --set "output.edited_images_dir=outputs/benchmark_images/gedit" \
  --set "output.scores_dir=outputs/scores/gedit" \
  --set "output.summary_path=outputs/scores/gedit/${GEDIT_MODEL}_export_summary.json" \
  2>&1 | tee -a "$GEDIT_LOG"

echo "[eval-r18] Starting GEdit scoring for $GEDIT_MODEL"
bash scripts/score_gedit.sh \
  --set "model.model_name=$GEDIT_MODEL" \
  --set "output.edited_images_dir=outputs/benchmark_images/gedit" \
  --set "output.scores_dir=outputs/scores/gedit" \
  --set "output.summary_path=outputs/scores/gedit/${GEDIT_MODEL}_score_summary.json" \
  --set "scoring.save_dir=outputs/scores/gedit" \
  --set "scoring.scorer_secret_env_path=secret.env" \
  2>&1 | tee -a "$GEDIT_LOG"

echo "[eval-r18] Completed ImgEdit and GEdit eval for checkpoint: $CHECKPOINT"
