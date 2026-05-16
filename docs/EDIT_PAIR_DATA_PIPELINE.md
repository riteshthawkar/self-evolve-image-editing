# Edit-Pair Data Pipeline

This pipeline is for supervised edit-pair data such as MagicBrush. It downloads
source-target edit triples from Hugging Face, filters unusable pairs, stores the
images locally, and writes a DiffSynth-compatible training manifest.

## Why Use This

Raw source-image pools are useful for self-evolving, but they do not provide
ground-truth edited targets. Edit-pair datasets are useful for:

- warm-starting the editor LoRA before self-evolve
- validating that the training stack works before expensive self-evolve runs
- building supervised baselines for the paper story
- filtering high-quality edit instructions and edit types for proposer/evaluator prompts

## Remote Setup

Run this from the repo root on the H200 machine:

```bash
conda activate /share_6/users/ritesh_thawkar/condaenvs/qedit
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
```

If `datasets` is missing:

```bash
pip install -e .
```

## MagicBrush Pilot

Use this first to verify the pipeline:

```bash
bash scripts/prepare_magicbrush.sh \
  --limit 1000 \
  --max-selected 800 \
  --output-root data/edit_pairs/magicbrush_pilot \
  --manifest data/manifests/magicbrush_pilot_filtered.json \
  --progress-every 100
```

Outputs:

```text
data/edit_pairs/magicbrush_pilot/source/
data/edit_pairs/magicbrush_pilot/target/
data/edit_pairs/magicbrush_pilot/all_records.jsonl
data/edit_pairs/magicbrush_pilot/selected_records.jsonl
data/edit_pairs/magicbrush_pilot/rejected_records.jsonl
data/edit_pairs/magicbrush_pilot/summary.json
data/manifests/magicbrush_pilot_filtered.json
```

Check the result:

```bash
python -m qwen_edit_project.data.validate_manifest \
  --manifest data/manifests/magicbrush_pilot_filtered.json

python - <<'PY'
import json
from pathlib import Path
summary = json.loads(Path("data/edit_pairs/magicbrush_pilot/summary.json").read_text())
print(json.dumps(summary, indent=2))
PY
```

## MagicBrush Full Run

After the pilot validates:

```bash
bash scripts/prepare_magicbrush.sh \
  --limit 0 \
  --max-selected 0 \
  --output-root data/edit_pairs/magicbrush_full \
  --manifest data/manifests/magicbrush_full_filtered.json \
  --progress-every 100
```

`--limit 0` means scan the full split. `--max-selected 0` means keep every pair
that passes the filters.

## Stricter Filtering

Use this if too many low-value edits pass:

```bash
bash scripts/prepare_magicbrush.sh \
  --limit 0 \
  --max-selected 5000 \
  --output-root data/edit_pairs/magicbrush_strict \
  --manifest data/manifests/magicbrush_strict_filtered.json \
  --min-total-score 0.58 \
  --min-changed-fraction 0.03 \
  --max-changed-fraction 0.70
```

## Resuming

The script streams `all_records.jsonl` while it runs. If the job is interrupted,
rerun the same command and it will skip already processed keys.

To rebuild from scratch:

```bash
bash scripts/prepare_magicbrush.sh \
  --limit 1000 \
  --output-root data/edit_pairs/magicbrush_pilot \
  --manifest data/manifests/magicbrush_pilot_filtered.json \
  --no-resume
```

## Generic HF Edit Dataset

For another Hugging Face dataset with source image, target image, and instruction
columns:

```bash
bash scripts/prepare_edit_pairs.sh \
  --dataset-path YOUR_ORG/YOUR_DATASET \
  --dataset-split train \
  --source-column SOURCE_IMAGE_COLUMN \
  --target-column TARGET_IMAGE_COLUMN \
  --instruction-column INSTRUCTION_COLUMN \
  --id-column ID_COLUMN \
  --output-root data/edit_pairs/YOUR_DATASET_NAME \
  --manifest data/manifests/YOUR_DATASET_NAME_filtered.json \
  --limit 0
```

If the dataset has no stable id or turn column, omit `--id-column` or
`--turn-column`.

## Train On The Filtered Manifest

For a supervised warm-start LoRA:

```bash
bash scripts/train_lora_2509.sh \
  --set dataset.dataset_base_path=. \
  --set dataset.dataset_metadata_path=data/manifests/magicbrush_pilot_filtered.json \
  --set output.output_path=outputs/checkpoints/qwen_edit_magicbrush_pilot_lora
```

For the full filtered set:

```bash
bash scripts/train_lora_2509.sh \
  --set dataset.dataset_base_path=. \
  --set dataset.dataset_metadata_path=data/manifests/magicbrush_full_filtered.json \
  --set output.output_path=outputs/checkpoints/qwen_edit_magicbrush_full_lora
```

## Recommended Order

1. Run `magicbrush_pilot`.
2. Validate the manifest.
3. Train a short LoRA warm-start on the pilot manifest.
4. Run the full MagicBrush filter.
5. Use the full supervised LoRA as a baseline or initializer before self-evolve.
