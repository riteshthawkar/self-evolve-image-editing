#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$ROOT/scripts/prepare_edit_pairs.sh" --config configs/data/edit_pairs_magicbrush.yaml "$@"
