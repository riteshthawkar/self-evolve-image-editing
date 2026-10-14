#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="${SOURCE_PATH_GPU:-0}"
exec python run.py --runtime dedicated generate "$@"
