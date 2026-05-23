#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-outputs/self_evolve/final_cepr_weighted_magicbrush_1024/internal-cepr-trainable-proposer}"
shift || true

PYTHONPATH="${PYTHONPATH:-src}" "${PYTHON:-python3}" -m qwen_edit_project.self_evolve.monitor \
  --root "$ROOT" \
  "$@"
