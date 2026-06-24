#!/usr/bin/env bash
# Reward-component ablation matrix (offline, free 24 GB GPU; no experiment machine).
#
# Each arm re-runs the real reward-discrimination harness on the same probe set
# but knocks out exactly one component. The reviewer-facing claim is that every
# component earns its place: removing a gate must either reintroduce FALSE-ACCEPTS
# (noop/corrupt/wrong accepted) or cost RECALL (good rejected), while the surgical
# conservative_region relaxation recovers recall WITHOUT releasing no-ops.
#
# All arms reuse the reward-only pipeline (transformer skipped, ~17 GB), so this
# does NOT need the experiment GPU. Each full 210-pair arm with production gates +
# fast-judge takes a few hours; run overnight or subset with --limit-per-type.
#
# Usage:
#   scripts/run_reward_ablation_matrix.sh                 # all arms, full probe set
#   LIMIT=5 scripts/run_reward_ablation_matrix.sh         # quick subset (5/type)
#   ARMS="A1_full A3_no_conservative" scripts/run_reward_ablation_matrix.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

PY="${PYTHON:-.venv_reward/bin/python}"
CFG="${CFG:-configs/self_evolve/qwen_edit_2509_balanced_cepr_v2.yaml}"
PROBE="${PROBE:-data/probe/anyedit_pairs}"
OUT_ROOT="${OUT_ROOT:-outputs/analysis/reward_ablation}"
LIMIT="${LIMIT:-0}"   # 0 = all pairs
HARNESS="scripts/run_reward_discrimination_study.py"

run_arm() {
  local name="$1"; shift
  local out_dir="$OUT_ROOT/$name"
  if [[ -f "$out_dir/report.json" && "${FORCE:-0}" != "1" ]]; then
    echo "== [$name] already has report.json (set FORCE=1 to rerun); skipping =="
    return 0
  fi
  echo "== [$name] running: $* =="
  mkdir -p "$out_dir"
  "$PY" "$HARNESS" \
    --probe-dir "$PROBE" \
    --out "$out_dir" \
    --limit-per-type "$LIMIT" \
    "$@" \
    2>&1 | tee "$out_dir/run.log"
}

# --- Arm definitions ------------------------------------------------------- #
# A0 establishes the gameable baseline; A1 is the production reference; A2..A5
# remove one gate each; A6 is the surgical fix (relax outside-change sub-gate but
# KEEP target-change so no-ops stay caught).
declare -A ARM_ARGS
ARM_ARGS[A0_embedding_only]=""   # no --evaluator-config -> lean default, all gates off
ARM_ARGS[A1_full_production]="--evaluator-config $CFG"
ARM_ARGS[A2_no_rubric_forbidden]="--evaluator-config $CFG --set rubric_forbidden_threshold=0.0"
ARM_ARGS[A3_no_conservative]="--evaluator-config $CFG --set conservative_region_reward_enabled=false"
ARM_ARGS[A4_no_object_detector]="--evaluator-config $CFG --set object_detector_enabled=false"
ARM_ARGS[A5_no_vlm_judge]="--evaluator-config $CFG --set internal_vlm_judge.enabled=false"
ARM_ARGS[A6_conservative_relax_outside]="--evaluator-config $CFG --set conservative_region_max_outside_change=0.140 --set conservative_region_max_outside_changed_fraction=0.75 --set conservative_region_min_outside_preservation=0.30 --set conservative_region_min_localization_precision=0.0"

DEFAULT_ARMS="A0_embedding_only A1_full_production A2_no_rubric_forbidden A3_no_conservative A4_no_object_detector A5_no_vlm_judge A6_conservative_relax_outside"
ARMS="${ARMS:-$DEFAULT_ARMS}"

for arm in $ARMS; do
  args="${ARM_ARGS[$arm]:-}"
  # shellcheck disable=SC2086
  run_arm "$arm" $args
done

echo "== aggregating arms -> comparison table =="
"$PY" scripts/summarize_reward_ablation.py \
  --arms-root "$OUT_ROOT" \
  --order $ARMS \
  --out "$OUT_ROOT/ablation_summary"
