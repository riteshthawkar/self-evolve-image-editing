#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"${PYTHON:-python3}" -m qwen_edit_project.eval.run_dpgbench_score --config configs/eval/dpgbench.yaml "$@"
