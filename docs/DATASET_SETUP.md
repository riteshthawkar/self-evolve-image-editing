# Dataset Setup

This document describes the dataset and asset layout expected by the current repo.

It covers four different surfaces:

1. supervised edit finetuning data
2. unlabeled self-evolve images
3. edit benchmark assets
4. generation benchmark assets

## Recommended directory layout

```text
data/
  manifests/
    train_metadata_qwen_edit.json
  processed/
    train/
      source/
      target/
    benchmark/
      imgedit/
        basic_edit.json
        original_images/
  unlabeled/
    self_evolve/
    demo/
  external/
    geneval_detector/
third_party/
  diffsynth-studio/
  imgedit/
  geneval/
  ella/
  oneig-bench/
```

## 1. Supervised edit finetuning data

The training config expects:

- dataset base path: `data/processed/train`
- manifest path: `data/manifests/train_metadata_qwen_edit.json`

The canonical manifest schema is documented in [docs/DATA_FORMAT.md](/Users/ritesh.thawkar/Ritesh/neurips-project/docs/DATA_FORMAT.md).

Each record must contain:

- `prompt`
- `image`
- `edit_image`

Where:

- `image` is the edited target image
- `edit_image` is the source image, or a list of source images for multi-image edits

### Recommended source layout

Use a simple pair layout under `data/processed/train`:

```text
data/processed/train/
  source/
    example_0001.png
    example_0002_a.png
    example_0002_b.png
  target/
    example_0001.png
    example_0002.png
```

### Build the manifest

Prepare a raw records file, for example:

```json
[
  {
    "prompt": "Replace the red mug with a blue ceramic mug.",
    "image": "target/example_0001.png",
    "edit_image": "source/example_0001.png"
  },
  {
    "prompt": "Use Figure 2 as the dress color reference for Figure 1.",
    "image": "target/example_0002.png",
    "edit_image": "source/example_0002_a.png|source/example_0002_b.png"
  }
]
```

Then run:

```bash
python -m qwen_edit_project.data.build_diffsynth_manifest \
  --input data/raw/train_records.json \
  --output data/manifests/train_metadata_qwen_edit.json \
  --base-dir data/processed/train \
  --relativize-paths
```

Validate it:

```bash
python -m qwen_edit_project.data.validate_manifest \
  --manifest data/manifests/train_metadata_qwen_edit.json
```

### Download and filter MagicBrush automatically

For a ready supervised edit-pair dataset, use the MagicBrush pipeline:

```bash
bash scripts/prepare_magicbrush.sh \
  --limit 1000 \
  --max-selected 800 \
  --output-root data/edit_pairs/magicbrush_pilot \
  --manifest data/manifests/magicbrush_pilot_filtered.json
```

Then train using:

```bash
bash scripts/train_lora_2509.sh \
  --set dataset.dataset_base_path=. \
  --set dataset.dataset_metadata_path=data/manifests/magicbrush_pilot_filtered.json \
  --set output.output_path=outputs/checkpoints/qwen_edit_magicbrush_pilot_lora
```

See [EDIT_PAIR_DATA_PIPELINE.md](/Users/ritesh.thawkar/Ritesh/neurips-project/docs/EDIT_PAIR_DATA_PIPELINE.md) for full commands, resume behavior, and generic Hugging Face dataset usage.

## 2. Unlabeled self-evolve pool

The self-evolve configs default to directory mode.

Expected path:

```text
data/unlabeled/self_evolve/
```

You can nest folders freely. The loader scans recursively for:

- `.png`
- `.jpg`
- `.jpeg`
- `.webp`
- `.bmp`

### Optional metadata sidecar

You can attach captions or metadata using a JSONL file and point `dataset.metadata_jsonl` at it.

Each line must contain at least:

- `key`

Optional fields:

- `caption`
- any additional metadata fields

Important detail:

- the `key` must match the image path relative to `images_dir`, with the file suffix removed and path separators replaced by `__`

Example:

```text
images_dir: data/unlabeled/self_evolve
image path: animals/dog_01.png
key: animals__dog_01
```

Example metadata line:

```json
{"key": "animals__dog_01", "caption": "A small brown dog on grass", "source": "web"}
```

### Source image selection

For the results-first delta path, filter the raw unlabeled pool before self-evolve:

```bash
bash scripts/select_unlabeled_images.sh \
  --set input.images_dir=data/unlabeled/raw \
  --limit 1000
```

Then run self-evolve from the selected manifest:

```bash
bash scripts/run_self_evolve_matrix.sh \
  --variant delta-results \
  --set dataset.source=jsonl \
  --set dataset.manifest_jsonl=data/unlabeled/selected/manifest.jsonl \
  --limit 128
```

See [SOURCE_IMAGE_SELECTION.md](/Users/ritesh.thawkar/Ritesh/neurips-project/docs/SOURCE_IMAGE_SELECTION.md) for the full filtering pipeline.

For remote data preparation on the single-GPU Slurm machine, use the bounded download/filter/split pipeline:

```bash
bash scripts/prepare_remote_data.sh \
  --stage all \
  --download-limit 20000 \
  --filter-limit 20000 \
  --max-selected 5000
```

This writes raw images, VLM-filtered selected manifests, and pilot/main/heldout splits under `data/unlabeled/`. See [REMOTE_DATA_PIPELINE.md](/Users/ritesh.thawkar/Ritesh/neurips-project/docs/REMOTE_DATA_PIPELINE.md).

## 3. Edit benchmark assets

Prepare benchmark assets with:

```bash
bash scripts/prepare_benchmark_data.sh --stage all
```

See [BENCHMARK_DATA_SETUP.md](/Users/ritesh.thawkar/Ritesh/neurips-project/docs/BENCHMARK_DATA_SETUP.md) for stage-by-stage setup and checks.

### GEdit-Bench

Default config:

- source: Hugging Face dataset
- dataset name: `stepfun-ai/GEdit-Bench`

So the default setup is minimal. Once the Python environment is ready, export reads it directly from Hugging Face.

If you want an offline local copy instead, switch the config to disk mode:

```yaml
dataset:
  source: disk
  local_path: path/to/saved_hf_dataset
```

### ImgEdit

This benchmark is **not** fully automatic. The current config expects:

- `data/processed/benchmark/imgedit/basic_edit.json`
- `data/processed/benchmark/imgedit/original_images/`
- `third_party/imgedit/Benchmark/Basic/prompts.json`

So after bootstrap, you still need to place the benchmark edit JSON and original images under:

```text
data/processed/benchmark/imgedit/
```

Use exactly these names unless you also override the config:

- `basic_edit.json`
- `original_images/`

## 4. Generation benchmark assets

### GenEval

After bootstrap:

- prompts metadata comes from `third_party/geneval/prompts/evaluation_metadata.jsonl`

You still need the object detector assets used by the scorer at:

```text
data/external/geneval_detector/
```

If you place them elsewhere, override:

```yaml
scoring:
  object_detector_root: /your/path
```

### DPG-Bench

After bootstrap:

- prompts CSV comes from `third_party/ella/dpg_bench/dpg_bench.csv`

No extra local data directory is required beyond whatever the upstream scorer needs in its own environment.

### OneIG-Bench

After bootstrap:

- CSVs come from `third_party/oneig-bench/OneIG-Bench.csv`
- and `third_party/oneig-bench/OneIG-Bench-ZH.csv`

The scorer stack has additional external model downloads. Follow the upstream README after bootstrap.

## Bootstrap first

Before any benchmark setup, clone the upstream repos:

```bash
bash scripts/bootstrap.sh
```

This is required because several configs point into `third_party/`.

## High-level experiment runners

After the datasets and assets are in place, use these higher-level scripts:

### Edit LoRA plus edit benchmarks

```bash
bash scripts/run_edit_model_suite.sh --model-type lora --train --limit 64
```

### Generation sanity suite

```bash
bash scripts/run_generation_sanity_suite.sh --limit 32
```

### Self-evolve ablations or matrix

```bash
bash scripts/run_self_evolve_matrix.sh --variant all --limit 32 --images-dir data/unlabeled/self_evolve
```

## Quick checklist

- `third_party/` bootstrapped
- editable package installed
- `data/manifests/train_metadata_qwen_edit.json` exists and validates
- `data/unlabeled/splits/<source>/pilot_manifest.jsonl` exists before pilot self-evolve
- `data/unlabeled/splits/<source>/main_manifest.jsonl` exists before main self-evolve
- `data/unlabeled/self_evolve/` exists for self-evolve runs
- `data/processed/benchmark/imgedit/` populated for ImgEdit
- `data/external/geneval_detector/` populated for GenEval scoring
